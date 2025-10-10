#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <glm/gtc/constants.hpp>

#include <viskores/Types.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ColorTable.h>
#include <viskores/cont/DataSet.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/Initialize.h>
#include <viskores/rendering/Actor.h>
#include <viskores/rendering/Camera.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/rendering/Color.h>
#include <viskores/rendering/MapperVolume.h>
#include <viskores/rendering/Scene.h>
#include <viskores/rendering/View3D.h>

#include "ViskoresVolumeExample.hpp"

#include <DirectSend/Base/DirectSendBase.hpp>

#include <Common/Color.hpp>
#include <Common/ImageRGBAUByteColorFloatDepth.hpp>
#include <Common/ImageSparse.hpp>
#include <Common/SavePPM.hpp>
#include <Common/YamlWriter.hpp>

namespace {

struct Options {
  int width = 512;
  int height = 512;
  int trials = 1;
  int samplesPerAxis = 64;
  std::string yamlOutput;
};

void printUsage() {
  std::cout << "Usage: ViskoresVolumeExample [--width W] [--height H] "
               "[--trials N] [--samples S] [--yaml-output FILE]\n"
            << "  --width W        Image width (default: 512)\n"
            << "  --height H       Image height (default: 512)\n"
            << "  --trials N       Number of render trials (default: 1)\n"
            << "  --samples S      Samples per axis for the volume (default: 64)\n"
            << "  --yaml-output F  Write timing information to YAML file F\n"
            << "  -h, --help       Show this help message\n";
}

Options parseOptions(int argc, char** argv, int rank, bool& exitEarly) {
  Options options;
  exitEarly = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--width") {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for --width");
      }
      options.width = std::stoi(argv[++i]);
      if (options.width <= 0) {
        throw std::runtime_error("image width must be positive");
      }
    } else if (arg == "--height") {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for --height");
      }
      options.height = std::stoi(argv[++i]);
      if (options.height <= 0) {
        throw std::runtime_error("image height must be positive");
      }
    } else if (arg == "--trials") {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for --trials");
      }
      options.trials = std::stoi(argv[++i]);
      if (options.trials <= 0) {
        throw std::runtime_error("number of trials must be positive");
      }
    } else if (arg == "--samples") {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for --samples");
      }
      options.samplesPerAxis = std::stoi(argv[++i]);
      if (options.samplesPerAxis < 2) {
        throw std::runtime_error("samples per axis must be at least 2");
      }
    } else if (arg == "--yaml-output") {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for --yaml-output");
      }
      options.yamlOutput = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      if (rank == 0) {
        printUsage();
      }
      exitEarly = true;
      return options;
    } else {
      std::ostringstream message;
      message << "unknown option '" << arg << "'";
      throw std::runtime_error(message.str());
    }
  }

  return options;
}

struct MpiGroupGuard {
  MpiGroupGuard() : group(MPI_GROUP_NULL) {}
  ~MpiGroupGuard() {
    if (group != MPI_GROUP_NULL) {
      MPI_Group_free(&group);
    }
  }

  MPI_Group group;
};

constexpr const char kDensityFieldName[] = "density";
constexpr unsigned int kDefaultCameraSeed = 91021u;

// Custom view that skips on-screen annotations to avoid colorbars.
class View3DNoColorBar : public viskores::rendering::View3D {
 public:
  using viskores::rendering::View3D::View3D;

  void RenderScreenAnnotations() override {}
  void RenderWorldAnnotations() override {}
};

class VolumePainterViskores {
 public:
  VolumePainterViskores();
  ~VolumePainterViskores() = default;

  double paint(const std::vector<ViskoresVolumeExample::VolumeBox>& boxes,
               const ViskoresVolumeExample::VolumeBounds& bounds,
               int samplesPerAxis,
               int rank,
               int numProcs,
               ImageFull& image,
               const ViskoresVolumeExample::CameraParameters& camera);

 private:
  viskores::cont::DataSet boxesToDataSet(
      const std::vector<ViskoresVolumeExample::VolumeBox>& boxes,
      const ViskoresVolumeExample::VolumeBounds& bounds,
      int samplesPerAxis) const;

  viskores::cont::ColorTable buildColorTable(int numProcs) const;

  void setupCamera(viskores::rendering::View3D& view,
                   const ViskoresVolumeExample::CameraParameters& camera);

  void canvasToImage(const viskores::rendering::Canvas& canvas,
                     ImageFull& image) const;
};

VolumePainterViskores::VolumePainterViskores() {
  static bool viskoresInitialized = false;
  if (!viskoresInitialized) {
    viskores::cont::Initialize();
    viskoresInitialized = true;
  }

}

viskores::cont::DataSet VolumePainterViskores::boxesToDataSet(
    const std::vector<ViskoresVolumeExample::VolumeBox>& boxes,
    const ViskoresVolumeExample::VolumeBounds& bounds,
    int samplesPerAxis) const {
  const glm::vec3 minCorner = bounds.minCorner;
  const glm::vec3 maxCorner = bounds.maxCorner;
  const glm::vec3 span = maxCorner - minCorner;

  viskores::cont::DataSetBuilderUniform builder;
  const viskores::Id3 dimensions(samplesPerAxis,
                                 samplesPerAxis,
                                 samplesPerAxis);
  const viskores::Vec3f_32 origin(minCorner.x, minCorner.y, minCorner.z);

  const auto safeSpacing = [&](float component) -> float {
    if (samplesPerAxis <= 1) {
      return 1.0f;
    }
    const float length = std::max(component, 1e-5f);
    return length / static_cast<float>(samplesPerAxis - 1);
  };

  const viskores::Vec3f_32 spacing(safeSpacing(span.x),
                                   safeSpacing(span.y),
                                   safeSpacing(span.z));

  viskores::cont::DataSet dataset = builder.Create(dimensions, origin, spacing);

  const std::size_t pointCount =
      static_cast<std::size_t>(samplesPerAxis) *
      static_cast<std::size_t>(samplesPerAxis) *
      static_cast<std::size_t>(samplesPerAxis);

  std::vector<viskores::Float32> pointScalars(pointCount, 0.0f);
  for (int z = 0; z < samplesPerAxis; ++z) {
    const float posZ =
        minCorner.z + static_cast<float>(z) * spacing[2];

    for (int y = 0; y < samplesPerAxis; ++y) {
      const float posY =
          minCorner.y + static_cast<float>(y) * spacing[1];

      for (int x = 0; x < samplesPerAxis; ++x) {
        const float posX =
            minCorner.x + static_cast<float>(x) * spacing[0];

        float sampleValue = 0.0f;
        for (const auto& box : boxes) {
          const bool inside =
              posX >= (box.minCorner.x - 1e-4f) &&
              posX <= (box.maxCorner.x + 1e-4f) &&
              posY >= (box.minCorner.y - 1e-4f) &&
              posY <= (box.maxCorner.y + 1e-4f) &&
              posZ >= (box.minCorner.z - 1e-4f) &&
              posZ <= (box.maxCorner.z + 1e-4f);
          if (inside) {
            sampleValue = std::max(sampleValue, box.scalarValue);
          }
        }

        const std::size_t linearIndex =
            static_cast<std::size_t>(z) * samplesPerAxis * samplesPerAxis +
            static_cast<std::size_t>(y) * samplesPerAxis +
            static_cast<std::size_t>(x);
        pointScalars[linearIndex] = sampleValue;
      }
    }
  }

  dataset.AddPointField(
      kDensityFieldName,
      viskores::cont::make_ArrayHandle(pointScalars, viskores::CopyFlag::On));

  return dataset;
}

viskores::cont::ColorTable VolumePainterViskores::buildColorTable(
    int numProcs) const {
  viskores::cont::ColorTable colorTable(viskores::ColorSpace::RGB);
  colorTable.AddPoint(0.0f, viskores::Vec3f_32(0.0f, 0.0f, 0.0f));
  colorTable.AddPointAlpha(0.0f, 0.0f);

  constexpr std::array<viskores::Vec3f_32, 8> kBaseColors = {
      viskores::Vec3f_32(0.941f, 0.341f, 0.200f),
      viskores::Vec3f_32(0.235f, 0.702f, 0.443f),
      viskores::Vec3f_32(0.192f, 0.494f, 0.773f),
      viskores::Vec3f_32(0.984f, 0.686f, 0.250f),
      viskores::Vec3f_32(0.414f, 0.353f, 0.804f),
      viskores::Vec3f_32(0.098f, 0.686f, 0.815f),
      viskores::Vec3f_32(0.596f, 0.306f, 0.639f),
      viskores::Vec3f_32(0.129f, 0.588f, 0.953f)};

  const int rankSamples = std::max(numProcs, 1);
  for (int proc = 0; proc < rankSamples; ++proc) {
    const float scalarValue =
        (static_cast<float>(proc) + 1.0f) /
        (static_cast<float>(rankSamples) + 1.0f);
    const viskores::Vec3f_32 color =
        kBaseColors[static_cast<std::size_t>(proc) % kBaseColors.size()];
    colorTable.AddPoint(scalarValue, color);
    colorTable.AddPointAlpha(scalarValue, 0.85f);
  }

  colorTable.AddPoint(1.0f, kBaseColors[0]);
  colorTable.AddPointAlpha(1.0f, 0.9f);
  return colorTable;
}

void VolumePainterViskores::setupCamera(
    viskores::rendering::View3D& targetView,
    const ViskoresVolumeExample::CameraParameters& cameraParams) {
  viskores::rendering::Camera camera;

  camera.SetPosition(viskores::Vec3f_32(cameraParams.eye.x,
                                        cameraParams.eye.y,
                                        cameraParams.eye.z));
  camera.SetLookAt(viskores::Vec3f_32(cameraParams.lookAt.x,
                                      cameraParams.lookAt.y,
                                      cameraParams.lookAt.z));
  camera.SetViewUp(viskores::Vec3f_32(cameraParams.up.x,
                                      cameraParams.up.y,
                                      cameraParams.up.z));
  camera.SetFieldOfView(cameraParams.fovYDegrees);
  camera.SetClippingRange(cameraParams.nearPlane, cameraParams.farPlane);

  targetView.SetCamera(camera);
}

void VolumePainterViskores::canvasToImage(
    const viskores::rendering::Canvas& viskoresCanvas,
    ImageFull& image) const {
  const auto colorPortal = viskoresCanvas.GetColorBuffer().ReadPortal();
  const auto depthPortal = viskoresCanvas.GetDepthBuffer().ReadPortal();

  const int width = image.getWidth();
  const int height = image.getHeight();

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int pixelIndex = y * width + x;
      const viskores::Vec4f color = colorPortal.Get(pixelIndex);
      const float depth = depthPortal.Get(pixelIndex);
      image.setColor(x,
                     y,
                     Color(color[0], color[1], color[2], color[3]));
      image.setDepth(x, y, depth);
    }
  }
}

double VolumePainterViskores::paint(
    const std::vector<ViskoresVolumeExample::VolumeBox>& boxes,
    const ViskoresVolumeExample::VolumeBounds& bounds,
    int samplesPerAxis,
    int rank,
    int numProcs,
    ImageFull& image,
    const ViskoresVolumeExample::CameraParameters& camera) {
  try {
    viskores::cont::DataSet dataset =
        boxesToDataSet(boxes, bounds, samplesPerAxis);
    viskores::cont::ColorTable colorTable = buildColorTable(numProcs);

    const glm::vec3 span = bounds.maxCorner - bounds.minCorner;
    const float minSpacing = std::max(
        1e-4f,
        std::min(span.x, std::min(span.y, span.z)) /
            static_cast<float>(std::max(samplesPerAxis - 1, 1)));

    viskores::rendering::CanvasRayTracer localCanvas(image.getWidth(),
                                                     image.getHeight());
    viskores::rendering::MapperVolume localMapper;
    localMapper.SetSampleDistance(minSpacing * 0.5f);
    localMapper.SetCompositeBackground(false);
    localMapper.SetActiveColorTable(colorTable);

    viskores::rendering::Scene localScene;
    viskores::rendering::Actor actor(dataset.GetCellSet(),
                                     dataset.GetCoordinateSystem(),
                                     dataset.GetField(kDensityFieldName),
                                     colorTable);
    localScene.AddActor(actor);

    View3DNoColorBar localView(localScene,
                               localMapper,
                               localCanvas,
                               viskores::rendering::Color(0.0f,
                                                          0.0f,
                                                          0.0f,
                                                          1.0f));
    localView.SetWorldAnnotationsEnabled(false);

    setupCamera(localView, camera);

    const auto startTime = std::chrono::high_resolution_clock::now();
    localView.Paint();
    const auto endTime = std::chrono::high_resolution_clock::now();

    canvasToImage(localCanvas, image);

    const double seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(endTime -
                                                                  startTime)
            .count();

    if (rank == 0) {
      std::cout << "VolumePainterViskores: Rendered volume with "
                << samplesPerAxis << "^3 samples across " << numProcs
                << " ranks" << std::endl;
    }

    return seconds;
  } catch (const std::exception& error) {
    std::cerr << "VolumePainterViskores error on rank " << rank << ": "
              << error.what() << std::endl;
    throw;
  }
}

}  // namespace

ViskoresVolumeExample::ViskoresVolumeExample() : rank(0), numProcs(1) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
}

void ViskoresVolumeExample::initialize() const {
  if (rank == 0) {
    std::cout << "ViskoresVolumeExample: Using Viskores volume mapper on "
              << numProcs << " ranks" << std::endl;
  }
}

std::vector<ViskoresVolumeExample::VolumeBox>
ViskoresVolumeExample::createRankSpecificBoxes(
    VolumeBounds& globalBounds) const {
  constexpr int boxesX = 2;
  constexpr int boxesY = 2;
  constexpr int boxesZ = 2;
  constexpr int totalBoxes = boxesX * boxesY * boxesZ;
  constexpr float spacing = 2.0f;
  constexpr float boxScale = 0.8f;

  const int ranks = std::max(numProcs, 1);
  const int boxesPerRank = totalBoxes / ranks;
  const int remainder = totalBoxes % ranks;
  const int localBoxCount = boxesPerRank + ((rank < remainder) ? 1 : 0);
  const int firstBoxIndex =
      boxesPerRank * rank + std::min(rank, remainder);

  std::vector<VolumeBox> boxes;
  boxes.reserve(std::max(localBoxCount, 0));

  glm::vec3 localMin(std::numeric_limits<float>::max());
  glm::vec3 localMax(-std::numeric_limits<float>::max());
  const glm::vec3 halfExtent(boxScale * 0.5f);

  for (int localIndex = 0; localIndex < localBoxCount; ++localIndex) {
    const int boxIndex = firstBoxIndex + localIndex;
    if (boxIndex >= totalBoxes) {
      break;
    }

    const int layer = boxIndex / (boxesX * boxesY);
    const int inLayerIndex = boxIndex % (boxesX * boxesY);
    const int row = inLayerIndex / boxesX;
    const int col = inLayerIndex % boxesX;

    glm::vec3 offset(0.0f);
    offset.x = (static_cast<float>(col) -
                (static_cast<float>(boxesX) - 1.0f) * 0.5f) *
               spacing;
    offset.y = ((static_cast<float>(boxesY) - 1.0f) * 0.5f -
                static_cast<float>(row)) *
               spacing;
    offset.z = (static_cast<float>(layer) -
                (static_cast<float>(boxesZ) - 1.0f) * 0.5f) *
               spacing;

    VolumeBox box;
    box.minCorner = offset - halfExtent;
    box.maxCorner = offset + halfExtent;
    box.scalarValue = (static_cast<float>(rank) + 1.0f) /
                      (static_cast<float>(numProcs) + 1.0f);

    boxes.push_back(box);

    localMin.x = std::min(localMin.x, box.minCorner.x);
    localMin.y = std::min(localMin.y, box.minCorner.y);
    localMin.z = std::min(localMin.z, box.minCorner.z);
    localMax.x = std::max(localMax.x, box.maxCorner.x);
    localMax.y = std::max(localMax.y, box.maxCorner.y);
    localMax.z = std::max(localMax.z, box.maxCorner.z);
  }

  if (boxes.empty()) {
    localMin = glm::vec3(std::numeric_limits<float>::max());
    localMax = glm::vec3(-std::numeric_limits<float>::max());
  }

  std::array<float, 3> localMinArray = {localMin.x, localMin.y, localMin.z};
  std::array<float, 3> localMaxArray = {localMax.x, localMax.y, localMax.z};
  std::array<float, 3> globalMinArray = {
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max()};
  std::array<float, 3> globalMaxArray = {
      -std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max()};

  MPI_Allreduce(localMinArray.data(),
                globalMinArray.data(),
                3,
                MPI_FLOAT,
                MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(localMaxArray.data(),
                globalMaxArray.data(),
                3,
                MPI_FLOAT,
                MPI_MAX,
                MPI_COMM_WORLD);

  glm::vec3 minCorner(globalMinArray[0],
                      globalMinArray[1],
                      globalMinArray[2]);
  glm::vec3 maxCorner(globalMaxArray[0],
                      globalMaxArray[1],
                      globalMaxArray[2]);

  if (minCorner.x > maxCorner.x || minCorner.y > maxCorner.y ||
      minCorner.z > maxCorner.z) {
    minCorner = glm::vec3(-1.0f);
    maxCorner = glm::vec3(1.0f);
  }

  const glm::vec3 padding(spacing * 0.35f);
  globalBounds.minCorner = minCorner - padding;
  globalBounds.maxCorner = maxCorner + padding;

  return boxes;
}

double ViskoresVolumeExample::paint(
    const std::vector<VolumeBox>& boxes,
    const VolumeBounds& bounds,
    int samplesPerAxis,
    ImageFull& image,
    const CameraParameters& camera) {
  static VolumePainterViskores painter;
  return painter.paint(
      boxes, bounds, samplesPerAxis, rank, numProcs, image, camera);
}

Compositor* ViskoresVolumeExample::getCompositor() {
  static DirectSendBase compositor;
  return &compositor;
}

int ViskoresVolumeExample::run(int argc, char** argv) {
  bool exitEarly = false;
  ::Options options;

  try {
    options = parseOptions(argc, argv, rank, exitEarly);
  } catch (const std::exception& error) {
    if (rank == 0) {
      std::cerr << "Error parsing options: " << error.what() << std::endl;
      std::cerr << "Use --help to list available options." << std::endl;
    }
    return 1;
  }

  if (exitEarly) {
    return 0;
  }

  initialize();

  VolumeBounds bounds;
  std::vector<VolumeBox> boxes = createRankSpecificBoxes(bounds);

  Compositor* compositor = getCompositor();
  if (compositor == nullptr) {
    if (rank == 0) {
      std::cerr << "No compositor available for the volume example."
                << std::endl;
    }
    return 1;
  }

  std::ofstream yamlFile;
  std::ostringstream yamlBuffer;
  std::ostream* yamlStream = nullptr;
  if (!options.yamlOutput.empty()) {
    yamlFile.open(options.yamlOutput.c_str());
    if (!yamlFile.good()) {
      if (rank == 0) {
        std::cerr << "Failed to open YAML output file '"
                  << options.yamlOutput << "'." << std::endl;
      }
      return 1;
    }
    yamlStream = &yamlFile;
  } else {
    yamlStream = &yamlBuffer;
  }
  YamlWriter yaml(*yamlStream);

  MpiGroupGuard groupGuard;
  MPI_Comm_group(MPI_COMM_WORLD, &groupGuard.group);

  const glm::vec3 center = 0.5f * (bounds.minCorner + bounds.maxCorner);
  const glm::vec3 halfExtent =
      0.5f * (bounds.maxCorner - bounds.minCorner);
  const float boundingRadius = glm::length(halfExtent);

  double accumulatedMaxPaintSeconds = 0.0;
  for (int trial = 0; trial < options.trials; ++trial) {
    ImageRGBAUByteColorFloatDepth localImage(options.width, options.height);

    const float aspect =
        static_cast<float>(options.width) / static_cast<float>(options.height);
    const float fovY = glm::pi<float>() / 4.0f;
    const float cameraDistance =
        boundingRadius / std::tan(fovY * 0.5f) + boundingRadius * 1.5f;

    std::mt19937 trialRng(kDefaultCameraSeed +
                          static_cast<unsigned int>(trial));
    std::uniform_real_distribution<float> angleDistribution(
        0.0f, glm::two_pi<float>());
    const float angle = angleDistribution(trialRng);

    const glm::vec3 eye(center.x + cameraDistance * std::sin(angle),
                        center.y + cameraDistance * 0.35f,
                        center.z + cameraDistance * std::cos(angle));

    const float nearPlane = 0.1f;
    const float farPlane = cameraDistance * 4.0f;
    CameraParameters camera{
        eye,
        center,
        glm::vec3(0.0f, 1.0f, 0.0f),
        fovY * 180.0f / glm::pi<float>(),
        nearPlane,
        farPlane};

    const double localPaintSeconds =
        paint(boxes,
              bounds,
              options.samplesPerAxis,
              localImage,
              camera);

    double maxPaintSeconds = 0.0;
    MPI_Allreduce(&localPaintSeconds,
                  &maxPaintSeconds,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    accumulatedMaxPaintSeconds += maxPaintSeconds;

    if (rank == 0) {
      std::cout << "Trial " << trial
                << ": volume paint time (max across ranks) = "
                << maxPaintSeconds << " s" << std::endl;
    }

    std::unique_ptr<Image> compositedImage =
        compositor->compose(&localImage, groupGuard.group, MPI_COMM_WORLD, yaml);

    if (compositedImage) {
      std::unique_ptr<ImageFull> fullImage;
      if (auto asFull = dynamic_cast<ImageFull*>(compositedImage.get())) {
        fullImage.reset(asFull);
        compositedImage.release();
      } else if (auto asSparse =
                     dynamic_cast<ImageSparse*>(compositedImage.get())) {
        fullImage = asSparse->uncompress();
      } else {
        if (rank == 0) {
          std::cerr << "Unsupported image type returned by compositor."
                    << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      std::unique_ptr<ImageFull> gatheredImage =
          fullImage->Gather(0, MPI_COMM_WORLD);

      if (rank == 0 && gatheredImage) {
        const int pixels = gatheredImage->getNumberOfPixels();
        std::cout << "Trial " << trial << ": composed " << pixels
                  << " pixels on rank 0" << std::endl;

        std::ostringstream filename;
        filename << "viskores-volume-trial-" << trial << ".ppm";
        if (SavePPM(*gatheredImage, filename.str())) {
          std::cout << "Saved trial " << trial
                    << " volume composited image to '" << filename.str()
                    << "'" << std::endl;
        } else {
          std::cerr << "Failed to save trial " << trial
                    << " composited image to '" << filename.str() << "'"
                    << std::endl;
        }
      }
    }
  }

  if (rank == 0 && options.trials > 0) {
    const double averageMaxPaintSeconds =
        accumulatedMaxPaintSeconds / static_cast<double>(options.trials);
    std::cout << "Average volume paint time (max across ranks): "
              << averageMaxPaintSeconds << " s" << std::endl;
  }

  return 0;
}

int main(int argc, char* argv[]) {
  std::vector<std::string> originalArgs(argv, argv + argc);
  std::vector<char*> argvCopy;
  argvCopy.reserve(originalArgs.size());
  for (std::size_t i = 0; i < originalArgs.size(); ++i) {
    argvCopy.push_back(const_cast<char*>(originalArgs[i].c_str()));
  }

  MPI_Init(&argc, &argv);

  int exitCode = 0;
  try {
    ViskoresVolumeExample example;
    const int argcCopy = static_cast<int>(argvCopy.size());
    exitCode = example.run(argcCopy, argvCopy.data());
  } catch (const std::exception& error) {
    int commRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    std::cerr << "Error on rank " << commRank << ": " << error.what()
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return exitCode;
}
