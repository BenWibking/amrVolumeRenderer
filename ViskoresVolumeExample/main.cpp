#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <viskores/Math.h>
#include <viskores/Types.h>
#include <viskores/VectorAnalysis.h>

#include "ViskoresVolumeExample.hpp"

#include <DirectSend/Base/DirectSendBase.hpp>

#include <Common/Color.hpp>
#include <Common/ImageRGBAFloatColorDepthSort.hpp>
#include <Common/ImageSparse.hpp>
#include <Common/LayeredVolumeImage.hpp>
#include <Common/SavePPM.hpp>
#include <Common/VisibilityOrdering.hpp>
#include <Common/VolumePainterViskores.hpp>
#include <Common/VolumeTypes.hpp>

namespace {

using Vec3 = viskores::Vec3f_32;
using minigraphics::volume::CameraParameters;
using minigraphics::volume::VolumeBounds;
using minigraphics::volume::VolumeBox;

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

struct ParsedOptions {
  ViskoresVolumeExample::RenderParameters parameters;
  std::string outputFilename = "viskores-volume-trial.ppm";
  bool exitEarly = false;
};

void printUsage() {
  std::cout << "Usage: ViskoresVolumeExample [--width W] [--height H] "
               "[--trials N] [--samples S] [--output FILE]\n"
            << "  --width W        Image width (default: 512)\n"
            << "  --height H       Image height (default: 512)\n"
            << "  --trials N       Number of render trials (default: 1)\n"
            << "  --samples S      Samples per axis for the volume (default: 64)\n"
            << "  --box-transparency T  Transparency factor per box in [0,1] "
               "(default: 0)\n"
            << "  --visibility-graph  Enable topological ordering using a visibility "
               "graph (default)\n"
            << "  --no-visibility-graph  Disable topological ordering using a "
               "visibility graph\n"
            << "  --output FILE    Output filename (default: viskores-volume-trial.ppm)\n"
            << "  -h, --help       Show this help message\n";
}

ParsedOptions parseOptions(int argc, char** argv, int rank) {
  ParsedOptions parsed;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    const auto requireValue = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for " + flag);
      }
      return argv[++i];
    };

    if (arg == "--width") {
      parsed.parameters.width = std::stoi(requireValue(arg));
      if (parsed.parameters.width <= 0) {
        throw std::runtime_error("image width must be positive");
      }
    } else if (arg == "--height") {
      parsed.parameters.height = std::stoi(requireValue(arg));
      if (parsed.parameters.height <= 0) {
        throw std::runtime_error("image height must be positive");
      }
    } else if (arg == "--trials") {
      parsed.parameters.trials = std::stoi(requireValue(arg));
      if (parsed.parameters.trials <= 0) {
        throw std::runtime_error("number of trials must be positive");
      }
    } else if (arg == "--samples") {
      parsed.parameters.samplesPerAxis = std::stoi(requireValue(arg));
      if (parsed.parameters.samplesPerAxis < 2) {
        throw std::runtime_error("samples per axis must be at least 2");
      }
    } else if (arg == "--box-transparency") {
      parsed.parameters.boxTransparency = std::stof(requireValue(arg));
      if (parsed.parameters.boxTransparency < 0.0f ||
          parsed.parameters.boxTransparency > 1.0f) {
        throw std::runtime_error(
            "box transparency must be between 0 and 1");
      }
    } else if (arg == "--visibility-graph") {
      parsed.parameters.useVisibilityGraph = true;
    } else if (arg == "--no-visibility-graph") {
      parsed.parameters.useVisibilityGraph = false;
    } else if (arg == "--output") {
      parsed.outputFilename = requireValue(arg);
      if (parsed.outputFilename.empty()) {
        throw std::runtime_error("output filename must not be empty");
      }
    } else if (arg == "--help" || arg == "-h") {
      if (rank == 0) {
        printUsage();
      }
      parsed.exitEarly = true;
      return parsed;
    } else {
      std::ostringstream message;
      message << "unknown option '" << arg << "'";
      throw std::runtime_error(message.str());
    }
  }

  return parsed;
}

std::string buildTrialFilename(const std::string& base,
                               int trialIndex,
                               int totalTrials) {
  if (totalTrials <= 1) {
    return base;
  }

  const std::string suffix = "-trial-" + std::to_string(trialIndex);
  const std::size_t dot = base.find_last_of('.');
  if (dot == std::string::npos || dot == 0) {
    return base + suffix;
  }
  return base.substr(0, dot) + suffix + base.substr(dot);
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

float computeBoxDepthHint(const VolumeBox& box,
                          const CameraParameters& camera) {
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

}  // namespace

ViskoresVolumeExample::ViskoresVolumeExample()
    : rank(0),
      numProcs(1) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
}

void ViskoresVolumeExample::initialize() const {
  if (rank == 0) {
    std::cout << "ViskoresVolumeExample: Using Viskores volume mapper on "
              << numProcs << " ranks" << std::endl;
  }
}

ViskoresVolumeExample::SceneGeometry
ViskoresVolumeExample::createRankSpecificGeometry() const {
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

  SceneGeometry scene;
  scene.localBoxes.reserve(std::max(localBoxCount, 0));

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

    scene.localBoxes.push_back(box);

    localMin = componentMin(localMin, box.minCorner);
    localMax = componentMax(localMax, box.maxCorner);
  }

  if (scene.localBoxes.empty()) {
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

  const Vec3 padding(spacing * 0.35f);
  if (minCorner[0] > maxCorner[0] || minCorner[1] > maxCorner[1] ||
      minCorner[2] > maxCorner[2]) {
    scene.explicitBounds.minCorner = Vec3(-1.0f) - padding;
    scene.explicitBounds.maxCorner = Vec3(1.0f) + padding;
  } else {
    scene.explicitBounds.minCorner = minCorner - padding;
    scene.explicitBounds.maxCorner = maxCorner + padding;
  }

  scene.hasExplicitBounds = true;
  return scene;
}

ViskoresVolumeExample::VolumeBounds ViskoresVolumeExample::computeGlobalBounds(
    const std::vector<VolumeBox>& boxes,
    bool hasExplicitBounds,
    const VolumeBounds& explicitBounds) const {
  if (hasExplicitBounds) {
    return explicitBounds;
  }

  Vec3 localMin(std::numeric_limits<float>::max());
  Vec3 localMax(-std::numeric_limits<float>::max());

  for (const auto& box : boxes) {
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

  VolumeBounds bounds;
  bounds.minCorner = Vec3(globalMinArray[0],
                          globalMinArray[1],
                          globalMinArray[2]);
  bounds.maxCorner = Vec3(globalMaxArray[0],
                          globalMaxArray[1],
                          globalMaxArray[2]);

  const bool invalidBounds =
      bounds.minCorner[0] > bounds.maxCorner[0] ||
      bounds.minCorner[1] > bounds.maxCorner[1] ||
      bounds.minCorner[2] > bounds.maxCorner[2];

  if (invalidBounds) {
    bounds.minCorner = Vec3(-1.0f);
    bounds.maxCorner = Vec3(1.0f);
    return bounds;
  }

  const Vec3 extent = bounds.maxCorner - bounds.minCorner;
  const float maxExtent =
      viskores::Max(extent[0], viskores::Max(extent[1], extent[2]));
  const float padding = (maxExtent > 0.0f) ? maxExtent * 0.05f : 0.5f;
  const Vec3 paddingVec(padding);

  bounds.minCorner -= paddingVec;
  bounds.maxCorner += paddingVec;
  return bounds;
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
  return BuildVisibilityOrderedGroup(camera,
                                     aspect,
                                     baseGroup,
                                     rank,
                                     numProcs,
                                     useVisibilityGraph,
                                     localBoxes,
                                     MPI_COMM_WORLD);
}

int ViskoresVolumeExample::renderScene(
    const std::string& outputFilenameBase,
    const RenderParameters& parameters,
    const SceneGeometry& geometry) {
  if (parameters.width <= 0 || parameters.height <= 0) {
    throw std::invalid_argument("image dimensions must be positive");
  }
  if (parameters.trials <= 0) {
    throw std::invalid_argument("number of trials must be positive");
  }
  if (parameters.samplesPerAxis < 2) {
    throw std::invalid_argument("samples per axis must be at least 2");
  }
  if (parameters.boxTransparency < 0.0f ||
      parameters.boxTransparency > 1.0f) {
    throw std::invalid_argument(
        "box transparency must be between 0 and 1");
  }

  initialize();

  VolumeBounds bounds =
      computeGlobalBounds(geometry.localBoxes,
                          geometry.hasExplicitBounds,
                          geometry.explicitBounds);

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
  const Vec3 halfExtent = 0.5f * (bounds.maxCorner - bounds.minCorner);
  float boundingRadius = viskores::Magnitude(halfExtent);
  if (boundingRadius <= 0.0f) {
    boundingRadius = 1.0f;
  }

  for (int trial = 0; trial < parameters.trials; ++trial) {
    const float aspect = static_cast<float>(parameters.width) /
                         static_cast<float>(parameters.height);
    const float fovY = viskores::Pif() * 0.25f;
    const float cameraDistance =
        boundingRadius / std::tan(fovY * 0.5f) + boundingRadius * 1.5f;

    std::mt19937 trialRng(parameters.cameraSeed +
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
    localLayers.reserve(geometry.localBoxes.size());
    std::vector<float> depthHints;
    depthHints.reserve(geometry.localBoxes.size());

    std::vector<VolumeBox> singleBox(1);

    for (std::size_t boxIndex = 0; boxIndex < geometry.localBoxes.size();
         ++boxIndex) {
      singleBox[0] = geometry.localBoxes[boxIndex];

      auto layerImage =
          std::make_unique<ImageRGBAFloatColorDepthSort>(parameters.width,
                                                         parameters.height);
      paint(singleBox,
            bounds,
            parameters.samplesPerAxis,
            parameters.boxTransparency,
            *layerImage,
            camera,
            &geometry.localBoxes[boxIndex].color);
      depthHints.push_back(computeBoxDepthHint(geometry.localBoxes[boxIndex],
                                               camera));
      localLayers.push_back(std::move(layerImage));
    }

    auto prototype =
        std::make_unique<ImageRGBAFloatColorDepthSort>(parameters.width,
                                                       parameters.height);

    LayeredVolumeImage layeredImage(parameters.width,
                                    parameters.height,
                                    std::move(localLayers),
                                    std::move(depthHints),
                                    std::move(prototype));

    MPI_Group orderedGroup =
        buildVisibilityOrderedGroup(camera,
                                    aspect,
                                    groupGuard.group,
                                    parameters.useVisibilityGraph,
                                    geometry.localBoxes);

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

        const std::string trialFilename =
            buildTrialFilename(outputFilenameBase, trial, parameters.trials);
        if (SavePPM(*gatheredImage, trialFilename)) {
          std::cout << "Saved trial " << trial
                    << " volume composited image to '" << trialFilename
                    << "'" << std::endl;
        } else {
          std::cerr << "Failed to save trial " << trial
                    << " composited image to '" << trialFilename << "'"
                    << std::endl;
        }
      }
    }
  }

  return 0;
}

int ViskoresVolumeExample::run(int argc, char** argv) {
  ParsedOptions options;

  try {
    options = parseOptions(argc, argv, rank);
  } catch (const std::exception& error) {
    if (rank == 0) {
      std::cerr << "Error parsing options: " << error.what() << std::endl;
      std::cerr << "Use --help to list available options." << std::endl;
    }
    return 1;
  }

  if (options.exitEarly) {
    return 0;
  }

  SceneGeometry geometry = createRankSpecificGeometry();
  return renderScene(options.outputFilename,
                     options.parameters,
                     geometry);
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
