#include <algorithm>
#include <array>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <random>
#include <sstream>
#include <utility>
#include <string>
#include <vector>
#include <numeric>
#include <fstream>
#include <iomanip>

#include <viskores/Math.h>
#include <viskores/Matrix.h>
#include <viskores/VectorAnalysis.h>
#include <viskores/rendering/MatrixHelpers.h>

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
#include <Common/ImageRGBAFloatColorDepthSort.hpp>
#include <Common/ImageSparse.hpp>
#include <Common/SavePPM.hpp>
#include <Common/LayeredImageInterface.hpp>

namespace {

using Vec3 = viskores::Vec3f_32;
using Vec4 = viskores::Vec4f_32;
using Matrix4x4 = viskores::Matrix<viskores::Float32, 4, 4>;

Vec3 componentMin(const Vec3& a, const Vec3& b) {
  Vec3 result;
  result[0] = viskores::Min(a[0], b[0]);
  result[1] = viskores::Min(a[1], b[1]);
  result[2] = viskores::Min(a[2], b[2]);
  return result;
}

Vec3 componentMax(const Vec3& a, const Vec3& b) {
  Vec3 result;
  result[0] = viskores::Max(a[0], b[0]);
  result[1] = viskores::Max(a[1], b[1]);
  result[2] = viskores::Max(a[2], b[2]);
  return result;
}

Matrix4x4 makePerspectiveMatrix(float fovYDegrees,
                                float aspect,
                                float nearPlane,
                                float farPlane) {
  Matrix4x4 matrix;
  viskores::MatrixIdentity(matrix);

  const float fovTangent = viskores::Tan(fovYDegrees * viskores::Pi_180f() * 0.5f);
  const float size = nearPlane * fovTangent;
  const float left = -size * aspect;
  const float right = size * aspect;
  const float bottom = -size;
  const float top = size;

  matrix(0, 0) = 2.0f * nearPlane / (right - left);
  matrix(1, 1) = 2.0f * nearPlane / (top - bottom);
  matrix(0, 2) = (right + left) / (right - left);
  matrix(1, 2) = (top + bottom) / (top - bottom);
  matrix(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
  matrix(3, 2) = -1.0f;
  matrix(2, 3) = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
  matrix(3, 3) = 0.0f;

  return matrix;
}

struct Options {
  int width = 512;
  int height = 512;
  int trials = 1;
  int samplesPerAxis = 64;
  float boxTransparency = 0.0f;
  bool useVisibilityGraph = false;
};

void printUsage() {
  std::cout << "Usage: ViskoresVolumeExample [--width W] [--height H] "
               "[--trials N] [--samples S]\n"
            << "  --width W        Image width (default: 512)\n"
            << "  --height H       Image height (default: 512)\n"
            << "  --trials N       Number of render trials (default: 1)\n"
            << "  --samples S      Samples per axis for the volume (default: 64)\n"
            << "  --box-transparency T  Transparency factor per box in [0,1] "
               "(default: 0)\n"
            << "  --visibility-graph  Enable topological ordering using a visibility "
               "graph\n"
            << "  -h, --help       Show this help message\n";
}

Options parseOptions(int argc, char** argv, int rank, bool& exitEarly) {
  Options options;
  exitEarly = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    const auto requireValue = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + flag);
      }
      return argv[++i];
    };

    if (arg == "--width") {
      options.width = std::stoi(requireValue(arg));
      if (options.width <= 0) {
        throw std::runtime_error("image width must be positive");
      }
    } else if (arg == "--height") {
      options.height = std::stoi(requireValue(arg));
      if (options.height <= 0) {
        throw std::runtime_error("image height must be positive");
      }
    } else if (arg == "--trials") {
      options.trials = std::stoi(requireValue(arg));
      if (options.trials <= 0) {
        throw std::runtime_error("number of trials must be positive");
      }
    } else if (arg == "--samples") {
      options.samplesPerAxis = std::stoi(requireValue(arg));
      if (options.samplesPerAxis < 2) {
        throw std::runtime_error("samples per axis must be at least 2");
      }
    } else if (arg == "--box-transparency") {
      options.boxTransparency = std::stof(requireValue(arg));
      if (options.boxTransparency < 0.0f ||
          options.boxTransparency > 1.0f) {
        throw std::runtime_error(
            "box transparency must be between 0 and 1");
      }
    } else if (arg == "--visibility-graph") {
      options.useVisibilityGraph = true;
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

  void paint(const std::vector<ViskoresVolumeExample::VolumeBox>& boxes,
             const ViskoresVolumeExample::VolumeBounds& bounds,
             int samplesPerAxis,
             int rank,
             int numProcs,
             float boxTransparency,
             ImageFull& image,
             const ViskoresVolumeExample::CameraParameters& camera,
             const Vec3* colorOverride = nullptr);

 private:
  viskores::cont::DataSet boxesToDataSet(
      const std::vector<ViskoresVolumeExample::VolumeBox>& boxes,
      const ViskoresVolumeExample::VolumeBounds& bounds,
      int samplesPerAxis) const;

  viskores::cont::ColorTable buildColorTable(int numProcs,
                                             float alphaScale) const;

  void setupCamera(viskores::rendering::View3D& view,
                   const ViskoresVolumeExample::CameraParameters& camera);

  void canvasToImage(const viskores::rendering::Canvas& canvas,
                     ImageFull& image,
                     const Vec3* colorOverride) const;
};

class LayeredVolumeImage : public Image, public LayeredImageInterface {
 public:
  LayeredVolumeImage(
      int width,
      int height,
      std::vector<std::unique_ptr<ImageRGBAFloatColorDepthSort>> layersIn,
      std::vector<float> depthHintsIn,
      std::unique_ptr<ImageRGBAFloatColorDepthSort> prototypeIn)
      : Image(width, height),
        layers(std::move(layersIn)),
        depthHints(std::move(depthHintsIn)),
        prototype(std::move(prototypeIn)) {}

  ~LayeredVolumeImage() override = default;

  int getLayerCount() const override {
    return static_cast<int>(layers.size());
  }

  Image* getLayer(int layerIndex) override {
    return layers[static_cast<std::size_t>(layerIndex)].get();
  }

  const Image* getLayer(int layerIndex) const override {
    return layers[static_cast<std::size_t>(layerIndex)].get();
  }

  float getLayerDepthHint(int layerIndex) const override {
    return depthHints[static_cast<std::size_t>(layerIndex)];
  }

  std::unique_ptr<Image> createEmptyLayer(int regionBegin,
                                          int regionEnd) const override {
    std::unique_ptr<Image> empty =
        prototype->createNew(regionBegin, regionEnd);
    empty->clear(Color(0.0f, 0.0f, 0.0f, 0.0f));
    return empty;
  }

 protected:
  void clearImpl(const Color& color, float depth) override {
    for (auto& layer : layers) {
      layer->clear(color, depth);
    }
  }

  std::unique_ptr<Image> createNewImpl(int,
                                       int,
                                       int,
                                       int) const override {
    throw std::logic_error("LayeredVolumeImage::createNewImpl not supported");
  }

  std::unique_ptr<const Image> shallowCopyImpl() const override {
    throw std::logic_error(
        "LayeredVolumeImage::shallowCopyImpl not supported");
  }

  std::unique_ptr<Image> copySubrange(int, int) const override {
    throw std::logic_error(
        "LayeredVolumeImage::copySubrange not supported");
  }

  std::unique_ptr<const Image> window(int, int) const override {
    throw std::logic_error("LayeredVolumeImage::window not supported");
  }

  std::vector<MPI_Request> ISend(int, MPI_Comm) const override {
    throw std::logic_error("LayeredVolumeImage::ISend not supported");
  }

  std::vector<MPI_Request> IReceive(int, MPI_Comm) override {
    throw std::logic_error("LayeredVolumeImage::IReceive not supported");
  }

 public:
  std::unique_ptr<Image> blend(const Image&) const override {
    throw std::logic_error("LayeredVolumeImage::blend not supported");
  }

  bool blendIsOrderDependent() const override { return true; }

 private:
  std::vector<std::unique_ptr<ImageRGBAFloatColorDepthSort>> layers;
  std::vector<float> depthHints;
  std::unique_ptr<ImageRGBAFloatColorDepthSort> prototype;
};

float computeBoxDepthHint(const ViskoresVolumeExample::VolumeBox& box,
                          const ViskoresVolumeExample::CameraParameters& camera) {
  const Vec3 viewDir = viskores::Normal(camera.lookAt - camera.eye);
  float minDepth = std::numeric_limits<float>::infinity();
  for (int cornerIndex = 0; cornerIndex < 8; ++cornerIndex) {
    const Vec3 corner((cornerIndex & 1) ? box.maxCorner[0] : box.minCorner[0],
                      (cornerIndex & 2) ? box.maxCorner[1] : box.minCorner[1],
                      (cornerIndex & 4) ? box.maxCorner[2] : box.minCorner[2]);
    minDepth = std::min(minDepth, viskores::Dot(corner - camera.eye, viewDir));
  }
  return minDepth;
}

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
  const Vec3 minCorner = bounds.minCorner;
  const Vec3 maxCorner = bounds.maxCorner;
  const Vec3 span = maxCorner - minCorner;

  viskores::cont::DataSetBuilderUniform builder;
  const viskores::Id3 dimensions(samplesPerAxis,
                                 samplesPerAxis,
                                 samplesPerAxis);
  const viskores::Vec3f_32 origin(minCorner[0], minCorner[1], minCorner[2]);

  const auto safeSpacing = [&](float component) -> float {
    if (samplesPerAxis <= 1) {
      return 1.0f;
    }
    const float length = std::max(component, 1e-5f);
    return length / static_cast<float>(samplesPerAxis - 1);
  };

  const viskores::Vec3f_32 spacing(safeSpacing(span[0]),
                                   safeSpacing(span[1]),
                                   safeSpacing(span[2]));

  viskores::cont::DataSet dataset = builder.Create(dimensions, origin, spacing);

  const std::size_t pointCount =
      static_cast<std::size_t>(samplesPerAxis) *
      static_cast<std::size_t>(samplesPerAxis) *
      static_cast<std::size_t>(samplesPerAxis);

  std::vector<viskores::Float32> pointScalars(pointCount, 0.0f);
  for (int z = 0; z < samplesPerAxis; ++z) {
    const float posZ =
        minCorner[2] + static_cast<float>(z) * spacing[2];

    for (int y = 0; y < samplesPerAxis; ++y) {
      const float posY =
          minCorner[1] + static_cast<float>(y) * spacing[1];

      for (int x = 0; x < samplesPerAxis; ++x) {
        const float posX =
            minCorner[0] + static_cast<float>(x) * spacing[0];

        float sampleValue = 0.0f;
        for (const auto& box : boxes) {
          const bool inside =
              posX >= (box.minCorner[0] - 1e-4f) &&
              posX <= (box.maxCorner[0] + 1e-4f) &&
              posY >= (box.minCorner[1] - 1e-4f) &&
              posY <= (box.maxCorner[1] + 1e-4f) &&
              posZ >= (box.minCorner[2] - 1e-4f) &&
              posZ <= (box.maxCorner[2] + 1e-4f);
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
    int numProcs, float alphaScale) const {
  (void)numProcs;
  const float clampedScale = std::clamp(alphaScale, 0.0f, 1.0f);
  viskores::cont::ColorTable colorTable("inferno");

  const std::array<float, 6> positions = {
      0.0f, 0.15f, 0.35f, 0.6f, 0.85f, 1.0f};
  const std::array<float, 6> alphaValues = {
      0.0f, 0.02f, 0.05f, 0.1f, 0.18f, 0.3f};

  for (std::size_t i = 0; i < positions.size(); ++i) {
    const float scaledAlpha =
        std::clamp(alphaValues[i] * clampedScale, 0.0f, 1.0f);
    colorTable.AddPointAlpha(positions[i], scaledAlpha);
  }

  return colorTable;
}

void VolumePainterViskores::setupCamera(
    viskores::rendering::View3D& targetView,
    const ViskoresVolumeExample::CameraParameters& cameraParams) {
  viskores::rendering::Camera camera;

  camera.SetPosition(cameraParams.eye);
  camera.SetLookAt(cameraParams.lookAt);
  camera.SetViewUp(cameraParams.up);
  camera.SetFieldOfView(cameraParams.fovYDegrees);
  camera.SetClippingRange(cameraParams.nearPlane, cameraParams.farPlane);

  targetView.SetCamera(camera);
}

void VolumePainterViskores::canvasToImage(
    const viskores::rendering::Canvas& viskoresCanvas,
    ImageFull& image,
    const Vec3* colorOverride) const {
  auto* depthSortedImage =
      dynamic_cast<ImageRGBAFloatColorDepthSort*>(&image);
  if (depthSortedImage == nullptr) {
    throw std::runtime_error(
        "VolumePainterViskores expects ImageRGBAFloatColorDepthSort images.");
  }

  const auto colorPortal = viskoresCanvas.GetColorBuffer().ReadPortal();
  const auto depthPortal = viskoresCanvas.GetDepthBuffer().ReadPortal();

  const int width = image.getWidth();
  const int height = image.getHeight();
  const auto clampUnit = [](float value) {
    return std::clamp(value, 0.0f, 1.0f);
  };

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int pixelIndex = y * width + x;
      const viskores::Vec4f color = colorPortal.Get(pixelIndex);
      const float alpha = clampUnit(static_cast<float>(color[3]));
      float red = clampUnit(static_cast<float>(color[0]));
      float green = clampUnit(static_cast<float>(color[1]));
      float blue = clampUnit(static_cast<float>(color[2]));
      if (colorOverride != nullptr) {
        const float intensity = (red + green + blue) / 3.0f;
        red = clampUnit(intensity * (*colorOverride)[0]);
        green = clampUnit(intensity * (*colorOverride)[1]);
        blue = clampUnit(intensity * (*colorOverride)[2]);
      }
      depthSortedImage->setColor(x, y, Color(red, green, blue, alpha));
      float depth = depthPortal.Get(pixelIndex);
      if (!std::isfinite(depth) || alpha <= 0.0f) {
        depth = std::numeric_limits<float>::infinity();
      }
      depthSortedImage->setDepthHint(x, y, depth);
    }
  }
}

void VolumePainterViskores::paint(
    const std::vector<ViskoresVolumeExample::VolumeBox>& boxes,
    const ViskoresVolumeExample::VolumeBounds& bounds,
    int samplesPerAxis,
    int rank,
    int numProcs,
    float boxTransparency,
    ImageFull& image,
    const ViskoresVolumeExample::CameraParameters& camera,
    const Vec3* colorOverride) {
  try {
    viskores::cont::DataSet dataset =
        boxesToDataSet(boxes, bounds, samplesPerAxis);
    const float alphaScale =
        std::clamp(1.0f - boxTransparency, 0.0f, 1.0f);
    viskores::cont::ColorTable colorTable =
        buildColorTable(numProcs, alphaScale);

    const Vec3 span = bounds.maxCorner - bounds.minCorner;
    const float minSpan =
        std::min({span[0], span[1], span[2]});
    const float minSpacing = std::max(
        1e-4f,
        minSpan / static_cast<float>(std::max(samplesPerAxis - 1, 1)));

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

    localView.Paint();

    canvasToImage(localCanvas, image, colorOverride);

    if (rank == 0) {
      std::cout << "VolumePainterViskores: Rendered volume with "
                << samplesPerAxis << "^3 samples across " << numProcs
                << " ranks" << std::endl;
    }
  } catch (const std::exception& error) {
    std::cerr << "VolumePainterViskores error on rank " << rank << ": "
              << error.what() << std::endl;
    throw;
  }
}

}  // namespace

ViskoresVolumeExample::ViskoresVolumeExample()
    : rank(0),
      numProcs(1),
      localCentroid(0.0f),
      localBoundsMin(0.0f),
      localBoundsMax(0.0f),
      hasLocalData(false) {
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
  constexpr float boxScale = 0.8f;
  constexpr float spacing = boxScale;  // centers are one box width apart
  const std::array<Vec3, 8> kPalette = {
      Vec3(0.894f, 0.102f, 0.110f),  // red
      Vec3(0.216f, 0.494f, 0.722f),  // blue
      Vec3(0.302f, 0.686f, 0.290f),  // green
      Vec3(0.596f, 0.306f, 0.639f),  // purple
      Vec3(1.000f, 0.498f, 0.000f),  // orange
      Vec3(1.000f, 0.929f, 0.435f),  // yellow
      Vec3(0.651f, 0.337f, 0.157f),  // brown
      Vec3(0.969f, 0.506f, 0.749f)};  // pink

  const int ranks = std::max(numProcs, 1);
  const int boxesPerRank = totalBoxes / ranks;
  const int remainder = totalBoxes % ranks;
  const int localBoxCount = boxesPerRank + ((rank < remainder) ? 1 : 0);
  const int firstBoxIndex =
      boxesPerRank * rank + std::min(rank, remainder);

  std::vector<VolumeBox> boxes;
  boxes.reserve(std::max(localBoxCount, 0));

  Vec3 localMin(std::numeric_limits<float>::max());
  Vec3 localMax(-std::numeric_limits<float>::max());
  const Vec3 halfExtent(boxScale * 0.5f);

  for (int localIndex = 0; localIndex < localBoxCount; ++localIndex) {
    const int boxIndex = firstBoxIndex + localIndex;
    if (boxIndex >= totalBoxes) {
      break;
    }

    const int layer = boxIndex / (boxesX * boxesY);
    const int inLayerIndex = boxIndex % (boxesX * boxesY);
    const int row = inLayerIndex / boxesX;
    const int col = inLayerIndex % boxesX;

    Vec3 offset(0.0f);
    offset[0] = (static_cast<float>(col) -
                 (static_cast<float>(boxesX) - 1.0f) * 0.5f) *
                spacing;
    offset[1] = ((static_cast<float>(boxesY) - 1.0f) * 0.5f -
                 static_cast<float>(row)) *
                spacing;
    offset[2] = (static_cast<float>(layer) -
                 (static_cast<float>(boxesZ) - 1.0f) * 0.5f) *
                spacing;

    VolumeBox box;
    box.minCorner = offset - halfExtent;
    box.maxCorner = offset + halfExtent;
    box.scalarValue = (static_cast<float>(boxIndex) + 1.0f) /
                      (static_cast<float>(totalBoxes) + 1.0f);
    box.color = kPalette[static_cast<std::size_t>(boxIndex) % kPalette.size()];

    boxes.push_back(box);

    localMin = componentMin(localMin, box.minCorner);
    localMax = componentMax(localMax, box.maxCorner);
  }

  if (boxes.empty()) {
    localMin = Vec3(std::numeric_limits<float>::max());
    localMax = Vec3(-std::numeric_limits<float>::max());
  }

  std::array<float, 3> localMinArray = {localMin[0], localMin[1], localMin[2]};
  std::array<float, 3> localMaxArray = {localMax[0], localMax[1], localMax[2]};
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

  Vec3 minCorner(globalMinArray[0],
                 globalMinArray[1],
                 globalMinArray[2]);
  Vec3 maxCorner(globalMaxArray[0],
                 globalMaxArray[1],
                 globalMaxArray[2]);

  if (minCorner[0] > maxCorner[0] || minCorner[1] > maxCorner[1] ||
      minCorner[2] > maxCorner[2]) {
    minCorner = Vec3(-1.0f);
    maxCorner = Vec3(1.0f);
  }

  const Vec3 padding(spacing * 0.35f);
  globalBounds.minCorner = minCorner - padding;
  globalBounds.maxCorner = maxCorner + padding;

  if (!boxes.empty()) {
    localCentroid = 0.5f * (localMin + localMax);
    localBoundsMin = localMin;
    localBoundsMax = localMax;
    hasLocalData = true;
  } else {
    localCentroid = Vec3(0.0f);
    localBoundsMin = Vec3(0.0f);
    localBoundsMax = Vec3(0.0f);
    hasLocalData = false;
  }

  return boxes;
}

void ViskoresVolumeExample::paint(
    const std::vector<VolumeBox>& boxes,
    const VolumeBounds& bounds,
    int samplesPerAxis,
    float boxTransparency,
    ImageFull& image,
    const CameraParameters& camera,
    const Vec3* colorOverride) {
  if (boxes.empty()) {
    image.clear(Color(0.0f, 0.0f, 0.0f, 0.0f));
    return;
  }

  static VolumePainterViskores painter;
  painter.paint(
      boxes,
      bounds,
      samplesPerAxis,
      rank,
      numProcs,
      boxTransparency,
      image,
      camera,
      colorOverride);
}

Compositor* ViskoresVolumeExample::getCompositor() {
  static DirectSendBase compositor;
  return &compositor;
}

MPI_Group ViskoresVolumeExample::buildVisibilityOrderedGroup(
    const CameraParameters& camera,
    float aspect,
    MPI_Group baseGroup,
    bool useVisibilityGraph,
    const std::vector<VolumeBox>& localBoxes) const {
  const Matrix4x4 modelview =
      viskores::rendering::MatrixHelpers::ViewMatrix(camera.eye, camera.lookAt, camera.up);
  const Matrix4x4 projection =
      makePerspectiveMatrix(camera.fovYDegrees, aspect, camera.nearPlane, camera.farPlane);

  int localHasDataFlag = hasLocalData ? 1 : 0;
  std::vector<int> allHasData(numProcs, 0);
  MPI_Allgather(&localHasDataFlag,
                1,
                MPI_INT,
                allHasData.data(),
                1,
                MPI_INT,
                MPI_COMM_WORLD);

  std::array<float, 6> localBoundsArray = {localBoundsMin[0],
                                           localBoundsMin[1],
                                           localBoundsMin[2],
                                           localBoundsMax[0],
                                           localBoundsMax[1],
                                           localBoundsMax[2]};
  std::vector<float> allBounds(static_cast<std::size_t>(numProcs) * 6, 0.0f);
  MPI_Allgather(localBoundsArray.data(),
                6,
                MPI_FLOAT,
                allBounds.data(),
                6,
                MPI_FLOAT,
                MPI_COMM_WORLD);

  struct DepthInfo {
    float minDepth;
    float maxDepth;
    int rank;
  };

  std::vector<DepthInfo> depthByRank(static_cast<std::size_t>(numProcs));
  for (int proc = 0; proc < numProcs; ++proc) {
    DepthInfo info{};
    info.rank = proc;

    if (allHasData[proc]) {
      const std::size_t base = static_cast<std::size_t>(proc) * 6;
      const Vec3 boundsMin(allBounds[base + 0],
                           allBounds[base + 1],
                           allBounds[base + 2]);
      const Vec3 boundsMax(allBounds[base + 3],
                           allBounds[base + 4],
                           allBounds[base + 5]);

      float minDepth = std::numeric_limits<float>::infinity();
      float maxDepth = -std::numeric_limits<float>::infinity();
      for (int cornerIndex = 0; cornerIndex < 8; ++cornerIndex) {
        const Vec3 corner((cornerIndex & 1) ? boundsMax[0] : boundsMin[0],
                          (cornerIndex & 2) ? boundsMax[1] : boundsMin[1],
                          (cornerIndex & 4) ? boundsMax[2] : boundsMin[2]);
        const Vec4 homogeneousCorner(corner[0], corner[1], corner[2], 1.0f);
        const Vec4 viewSpace =
            viskores::MatrixMultiply(modelview, homogeneousCorner);
        const Vec4 clipSpace =
            viskores::MatrixMultiply(projection, viewSpace);
        if (clipSpace[3] != 0.0f) {
          const float normalizedDepth = clipSpace[2] / clipSpace[3];
          minDepth = std::min(minDepth, normalizedDepth);
          maxDepth = std::max(maxDepth, normalizedDepth);
        }
      }

      if (!std::isfinite(minDepth) || !std::isfinite(maxDepth)) {
        minDepth = std::numeric_limits<float>::infinity();
        maxDepth = std::numeric_limits<float>::infinity();
      }
      info.minDepth = minDepth;
      info.maxDepth = maxDepth;
    } else {
      info.minDepth = std::numeric_limits<float>::infinity();
      info.maxDepth = std::numeric_limits<float>::infinity();
    }

    depthByRank[static_cast<std::size_t>(proc)] = info;
  }

  auto depthCompare = [](const DepthInfo& a, const DepthInfo& b) {
    const bool aFinite = std::isfinite(a.minDepth);
    const bool bFinite = std::isfinite(b.minDepth);
    if (aFinite != bFinite) {
      return aFinite && !bFinite;
    }
    if (a.minDepth == b.minDepth) {
      if (a.maxDepth == b.maxDepth) {
        return a.rank < b.rank;
      }
      return a.maxDepth < b.maxDepth;
    }
    return a.minDepth < b.minDepth;
  };

  std::vector<DepthInfo> sortedDepth = depthByRank;
  std::sort(sortedDepth.begin(), sortedDepth.end(), depthCompare);

  auto fallbackOrder = [this, &sortedDepth]() {
    std::vector<int> order(static_cast<std::size_t>(this->numProcs));
    for (int i = 0; i < this->numProcs; ++i) {
      order[static_cast<std::size_t>(i)] =
          sortedDepth[static_cast<std::size_t>(i)].rank;
    }
    return order;
  };

  auto attemptVisibilityGraphOrdering =
      [&]() -> std::pair<bool, std::vector<int>> {
    const int localBoxCount =
        static_cast<int>(localBoxes.size());
    std::vector<int> allBoxCounts(static_cast<std::size_t>(numProcs), 0);
    MPI_Allgather(&localBoxCount,
                  1,
                  MPI_INT,
                  allBoxCounts.data(),
                  1,
                  MPI_INT,
                  MPI_COMM_WORLD);

    const int totalBoxes =
        std::accumulate(allBoxCounts.begin(), allBoxCounts.end(), 0);
    if (totalBoxes <= 0) {
      return {true, fallbackOrder()};
    }

    std::vector<int> boxDispls(static_cast<std::size_t>(numProcs), 0);
    for (int i = 1; i < numProcs; ++i) {
      boxDispls[static_cast<std::size_t>(i)] =
          boxDispls[static_cast<std::size_t>(i) - 1] +
          allBoxCounts[static_cast<std::size_t>(i) - 1];
    }

    std::vector<float> localBoxBounds(
        static_cast<std::size_t>(localBoxCount) * 6, 0.0f);
    for (int i = 0; i < localBoxCount; ++i) {
      const VolumeBox& box = localBoxes[static_cast<std::size_t>(i)];
      const std::size_t base = static_cast<std::size_t>(i) * 6;
      localBoxBounds[base + 0] = box.minCorner[0];
      localBoxBounds[base + 1] = box.minCorner[1];
      localBoxBounds[base + 2] = box.minCorner[2];
      localBoxBounds[base + 3] = box.maxCorner[0];
      localBoxBounds[base + 4] = box.maxCorner[1];
      localBoxBounds[base + 5] = box.maxCorner[2];
    }

    std::vector<float> allBoxBounds(
        static_cast<std::size_t>(totalBoxes) * 6, 0.0f);
    std::vector<int> countsBounds(static_cast<std::size_t>(numProcs), 0);
    std::vector<int> displsBounds(static_cast<std::size_t>(numProcs), 0);
    for (int i = 0; i < numProcs; ++i) {
      countsBounds[static_cast<std::size_t>(i)] =
          allBoxCounts[static_cast<std::size_t>(i)] * 6;
      displsBounds[static_cast<std::size_t>(i)] =
          boxDispls[static_cast<std::size_t>(i)] * 6;
    }

    MPI_Allgatherv(localBoxBounds.data(),
                   localBoxCount * 6,
                   MPI_FLOAT,
                   allBoxBounds.data(),
                   countsBounds.data(),
                   displsBounds.data(),
                   MPI_FLOAT,
                   MPI_COMM_WORLD);

    std::vector<int> localOwners(static_cast<std::size_t>(localBoxCount),
                                 this->rank);
    std::vector<int> allOwners(static_cast<std::size_t>(totalBoxes), 0);
    MPI_Allgatherv(localOwners.data(),
                   localBoxCount,
                   MPI_INT,
                   allOwners.data(),
                   allBoxCounts.data(),
                   boxDispls.data(),
                   MPI_INT,
                   MPI_COMM_WORLD);

    struct BoxInfo {
      Vec3 minCorner;
      Vec3 maxCorner;
      int ownerRank = -1;
      float minDepth = std::numeric_limits<float>::infinity();
      float maxDepth = std::numeric_limits<float>::infinity();
    };

    std::vector<BoxInfo> globalBoxes(static_cast<std::size_t>(totalBoxes));

    auto computeDepthRange = [&](const Vec3& minCorner,
                                 const Vec3& maxCorner) {
      float minDepth = std::numeric_limits<float>::infinity();
      float maxDepth = -std::numeric_limits<float>::infinity();
      for (int cornerIndex = 0; cornerIndex < 8; ++cornerIndex) {
        const Vec3 corner((cornerIndex & 1) ? maxCorner[0] : minCorner[0],
                          (cornerIndex & 2) ? maxCorner[1] : minCorner[1],
                          (cornerIndex & 4) ? maxCorner[2] : minCorner[2]);
        const Vec4 homogeneousCorner(corner[0],
                                     corner[1],
                                     corner[2],
                                     1.0f);
        const Vec4 viewSpace =
            viskores::MatrixMultiply(modelview, homogeneousCorner);
        const Vec4 clipSpace =
            viskores::MatrixMultiply(projection, viewSpace);
        if (clipSpace[3] != 0.0f) {
          const float normalizedDepth = clipSpace[2] / clipSpace[3];
          minDepth = std::min(minDepth, normalizedDepth);
          maxDepth = std::max(maxDepth, normalizedDepth);
        }
      }

      if (!std::isfinite(minDepth) || !std::isfinite(maxDepth)) {
        minDepth = std::numeric_limits<float>::infinity();
        maxDepth = std::numeric_limits<float>::infinity();
      }
      return std::make_pair(minDepth, maxDepth);
    };

    for (int boxIndex = 0; boxIndex < totalBoxes; ++boxIndex) {
      const std::size_t base = static_cast<std::size_t>(boxIndex) * 6;
      BoxInfo info;
      info.minCorner = Vec3(allBoxBounds[base + 0],
                            allBoxBounds[base + 1],
                            allBoxBounds[base + 2]);
      info.maxCorner = Vec3(allBoxBounds[base + 3],
                            allBoxBounds[base + 4],
                            allBoxBounds[base + 5]);
      info.ownerRank = allOwners[static_cast<std::size_t>(boxIndex)];
      const auto depthRange =
          computeDepthRange(info.minCorner, info.maxCorner);
      info.minDepth = depthRange.first;
      info.maxDepth = depthRange.second;
      globalBoxes[static_cast<std::size_t>(boxIndex)] = info;
    }

    Vec3 viewDir = camera.lookAt - camera.eye;
    const float viewDirMagnitude = viskores::Magnitude(viewDir);
    if (viewDirMagnitude > 0.0f) {
      viewDir = viewDir * (1.0f / viewDirMagnitude);
    } else {
      viewDir = Vec3(0.0f, 0.0f, -1.0f);
    }

    auto nearlyEqual = [](float a, float b) {
      const float scale = std::max({1.0f, std::fabs(a), std::fabs(b)});
      return std::fabs(a - b) <= 1e-5f * scale;
    };

    auto overlaps = [](float aMin, float aMax, float bMin, float bMax) {
      const float overlapMin = std::max(aMin, bMin);
      const float overlapMax = std::min(aMax, bMax);
      const float scale = std::max(
          {1.0f,
           std::fabs(aMin),
           std::fabs(aMax),
           std::fabs(bMin),
           std::fabs(bMax),
           std::fabs(overlapMin),
           std::fabs(overlapMax)});
      return (overlapMax - overlapMin) > 1e-5f * scale;
    };

    std::vector<std::vector<int>> adjacency(
        static_cast<std::size_t>(totalBoxes));
    std::vector<int> indegree(static_cast<std::size_t>(totalBoxes), 0);

    auto addEdge = [&](int from, int to) {
      if (from == to) {
        return;
      }
      auto& edges = adjacency[static_cast<std::size_t>(from)];
      if (std::find(edges.begin(), edges.end(), to) == edges.end()) {
        edges.push_back(to);
        ++indegree[static_cast<std::size_t>(to)];
      }
    };

    for (int i = 0; i < totalBoxes; ++i) {
      const BoxInfo& a = globalBoxes[static_cast<std::size_t>(i)];
      for (int j = i + 1; j < totalBoxes; ++j) {
        const BoxInfo& b = globalBoxes[static_cast<std::size_t>(j)];

        for (int axis = 0; axis < 3; ++axis) {
          const int axis1 = (axis + 1) % 3;
          const int axis2 = (axis + 2) % 3;
          const bool overlapAxis1 = overlaps(a.minCorner[axis1],
                                             a.maxCorner[axis1],
                                             b.minCorner[axis1],
                                             b.maxCorner[axis1]);
          const bool overlapAxis2 = overlaps(a.minCorner[axis2],
                                             a.maxCorner[axis2],
                                             b.minCorner[axis2],
                                             b.maxCorner[axis2]);
          if (!overlapAxis1 || !overlapAxis2) {
            continue;
          }

          const float dirComponent = viewDir[axis];
          constexpr float kDirectionTolerance = 1e-6f;

          if (nearlyEqual(a.maxCorner[axis], b.minCorner[axis])) {
            if (dirComponent > kDirectionTolerance) {
              addEdge(j, i);
            } else if (dirComponent < -kDirectionTolerance) {
              addEdge(i, j);
            } else {
              const float depthA = a.minDepth;
              const float depthB = b.minDepth;
              if (depthA <= depthB) {
                addEdge(j, i);
              } else {
                addEdge(i, j);
              }
            }
          } else if (nearlyEqual(b.maxCorner[axis], a.minCorner[axis])) {
            if (dirComponent > kDirectionTolerance) {
              addEdge(i, j);
            } else if (dirComponent < -kDirectionTolerance) {
              addEdge(j, i);
            } else {
              const float depthA = a.minDepth;
              const float depthB = b.minDepth;
              if (depthA <= depthB) {
                addEdge(i, j);
              } else {
                addEdge(j, i);
              }
            }
          }
        }
      }
    }

    if (rank == 0) {
      static int graphFileCounter = 0;
      std::ostringstream filename;
      filename << "visibility_graph_" << graphFileCounter++ << ".dot";
      std::ofstream dotFile(filename.str());
      if (dotFile) {
        dotFile << "digraph VisibilityGraph {\n";
        dotFile << "  rankdir=LR;\n";
        dotFile << std::fixed << std::setprecision(6);
        for (int idx = 0; idx < totalBoxes; ++idx) {
          const BoxInfo& info = globalBoxes[static_cast<std::size_t>(idx)];
          dotFile << "  box" << idx << " [label=\"box " << idx
                  << "\\nrank " << info.ownerRank
                  << "\\nminDepth " << info.minDepth
                  << "\\nmaxDepth " << info.maxDepth << "\"];\n";
        }
        for (int from = 0; from < totalBoxes; ++from) {
          for (int to : adjacency[static_cast<std::size_t>(from)]) {
            dotFile << "  box" << from << " -> box" << to << ";\n";
          }
        }
        dotFile << "}\n";
        std::cout << "Wrote visibility graph to '" << filename.str() << "'"
                  << std::endl;
      } else {
        std::cerr << "Failed to write visibility graph to '" << filename.str()
                  << "'" << std::endl;
      }
    }

    auto compareBoxes = [&globalBoxes](int lhs, int rhs) {
      const BoxInfo& a = globalBoxes[static_cast<std::size_t>(lhs)];
      const BoxInfo& b = globalBoxes[static_cast<std::size_t>(rhs)];
      const bool aFinite = std::isfinite(a.minDepth);
      const bool bFinite = std::isfinite(b.minDepth);
      if (aFinite != bFinite) {
        return aFinite && !bFinite;
      }
      if (a.minDepth == b.minDepth) {
        if (a.maxDepth == b.maxDepth) {
          if (a.ownerRank == b.ownerRank) {
            return lhs < rhs;
          }
          return a.ownerRank < b.ownerRank;
        }
        return a.maxDepth < b.maxDepth;
      }
      return a.minDepth < b.minDepth;
    };

    std::vector<int> ready;
    ready.reserve(static_cast<std::size_t>(totalBoxes));
    for (int boxIndex = 0; boxIndex < totalBoxes; ++boxIndex) {
      if (indegree[static_cast<std::size_t>(boxIndex)] == 0) {
        ready.push_back(boxIndex);
      }
    }

    std::vector<int> topoOrder;
    topoOrder.reserve(static_cast<std::size_t>(totalBoxes));

    auto sortReady = [&]() {
      std::sort(ready.begin(), ready.end(), compareBoxes);
    };

    sortReady();
    while (!ready.empty()) {
      const int current = ready.front();
      ready.erase(ready.begin());
      topoOrder.push_back(current);

      for (int next : adjacency[static_cast<std::size_t>(current)]) {
        int& in = indegree[static_cast<std::size_t>(next)];
        --in;
        if (in == 0) {
          ready.push_back(next);
        }
      }
      sortReady();
    }

    if (static_cast<int>(topoOrder.size()) != totalBoxes) {
      return {false, {}};
    }

    std::vector<int> rankVisited(static_cast<std::size_t>(numProcs), 0);
    std::vector<int> rankOrder;
    rankOrder.reserve(static_cast<std::size_t>(numProcs));
    for (int boxIndex : topoOrder) {
      const int owner = globalBoxes[static_cast<std::size_t>(boxIndex)].ownerRank;
      if (owner >= 0 &&
          rankVisited[static_cast<std::size_t>(owner)] == 0) {
        rankVisited[static_cast<std::size_t>(owner)] = 1;
        rankOrder.push_back(owner);
      }
    }

    for (const DepthInfo& info : sortedDepth) {
      const int owner = info.rank;
      if (rankVisited[static_cast<std::size_t>(owner)] == 0) {
        rankVisited[static_cast<std::size_t>(owner)] = 1;
        rankOrder.push_back(owner);
      }
    }

    return {true, std::move(rankOrder)};
  };

  std::vector<int> rankOrder;
  if (useVisibilityGraph) {
    auto result = attemptVisibilityGraphOrdering();
    if (result.first) {
      rankOrder = std::move(result.second);
    } else {
      static bool warnedGraphFailure = false;
      if (!warnedGraphFailure && rank == 0) {
        std::cerr << "Visibility graph ordering failed; "
                     "falling back to depth sorting."
                  << std::endl;
      }
      warnedGraphFailure = true;
      rankOrder = fallbackOrder();
    }
  } else {
    rankOrder = fallbackOrder();
  }

  MPI_Group orderedGroup = MPI_GROUP_NULL;
  MPI_Group_incl(baseGroup, numProcs, rankOrder.data(), &orderedGroup);
  return orderedGroup;
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

  MpiGroupGuard groupGuard;
  MPI_Comm_group(MPI_COMM_WORLD, &groupGuard.group);

  const Vec3 center = 0.5f * (bounds.minCorner + bounds.maxCorner);
  const Vec3 halfExtent =
      0.5f * (bounds.maxCorner - bounds.minCorner);
  const float boundingRadius = viskores::Magnitude(halfExtent);

  for (int trial = 0; trial < options.trials; ++trial) {
    const float aspect =
        static_cast<float>(options.width) / static_cast<float>(options.height);
    const float fovY = viskores::Pif() * 0.25f;
    const float cameraDistance =
        boundingRadius / std::tan(fovY * 0.5f) + boundingRadius * 1.5f;

    std::mt19937 trialRng(kDefaultCameraSeed +
                          static_cast<unsigned int>(trial));
    std::uniform_real_distribution<float> angleDistribution(
        0.0f, viskores::TwoPif());
    const float angle = angleDistribution(trialRng);

    const Vec3 eye(center[0] + cameraDistance * std::sin(angle),
                   center[1] + cameraDistance * 0.35f,
                   center[2] + cameraDistance * std::cos(angle));

    const float nearPlane = 0.1f;
    const float farPlane = cameraDistance * 4.0f;
    CameraParameters camera{
        eye,
        center,
        Vec3(0.0f, 1.0f, 0.0f),
        fovY * 180.0f / viskores::Pif(),
        nearPlane,
        farPlane};

    std::vector<std::unique_ptr<ImageRGBAFloatColorDepthSort>> localLayers;
    localLayers.reserve(boxes.size());
    std::vector<float> depthHints;
    depthHints.reserve(boxes.size());

    std::vector<VolumeBox> singleBox(1);

    for (std::size_t boxIndex = 0; boxIndex < boxes.size(); ++boxIndex) {
      singleBox[0] = boxes[boxIndex];

      auto layerImage =
          std::make_unique<ImageRGBAFloatColorDepthSort>(options.width,
                                                         options.height);
      paint(singleBox,
            bounds,
            options.samplesPerAxis,
            options.boxTransparency,
            *layerImage,
            camera,
            &boxes[boxIndex].color);
      depthHints.push_back(computeBoxDepthHint(boxes[boxIndex], camera));
      localLayers.push_back(std::move(layerImage));
    }

    auto prototype =
        std::make_unique<ImageRGBAFloatColorDepthSort>(options.width,
                                                       options.height);

    LayeredVolumeImage layeredImage(options.width,
                                    options.height,
                                    std::move(localLayers),
                                    std::move(depthHints),
                                    std::move(prototype));

    MPI_Group orderedGroup =
        buildVisibilityOrderedGroup(camera,
                                    aspect,
                                    groupGuard.group,
                                    options.useVisibilityGraph,
                                    boxes);

    std::unique_ptr<Image> compositedImage =
        compositor->compose(&layeredImage, orderedGroup, MPI_COMM_WORLD);

    MPI_Group_free(&orderedGroup);

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
