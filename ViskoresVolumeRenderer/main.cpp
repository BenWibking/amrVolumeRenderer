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
#include <viskores/Matrix.h>
#include <viskores/Types.h>
#include <viskores/VectorAnalysis.h>
#include <viskores/rendering/MatrixHelpers.h>

#include "ViskoresVolumeRenderer.hpp"

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
using Matrix4x4 = viskores::Matrix<viskores::Float32, 4, 4>;
using viskores::Id;
using minigraphics::volume::CameraParameters;
using minigraphics::volume::VolumeBounds;
using minigraphics::volume::AmrBox;

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

  const float fovTangent =
      viskores::Tan(fovYDegrees * viskores::Pi_180f() * 0.5f);
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

void clearDepthHints(ImageRGBAFloatColorDepthSort& image, float depth) {
  const int totalPixels = image.getNumberOfPixels();
  for (int pixelIndex = 0; pixelIndex < totalPixels; ++pixelIndex) {
    image.setDepthHint(pixelIndex, depth);
  }
}

void renderBoundingBoxLayer(const VolumeBounds& bounds,
                            const CameraParameters& camera,
                            int sqrtAntialiasing,
                            ImageRGBAFloatColorDepthSort& layer) {
  const int width = layer.getWidth();
  const int height = layer.getHeight();
  if (width <= 0 || height <= 0) {
    return;
  }

  const float aspect =
      static_cast<float>(width) / static_cast<float>(std::max(height, 1));

  const Matrix4x4 view = viskores::rendering::MatrixHelpers::ViewMatrix(
      camera.eye, camera.lookAt, camera.up);
  const Matrix4x4 projection =
      makePerspectiveMatrix(camera.fovYDegrees,
                            aspect,
                            camera.nearPlane,
                            camera.farPlane);

  Vec3 forward = camera.lookAt - camera.eye;
  const float forwardLength = viskores::Magnitude(forward);
  if (forwardLength > 0.0f) {
    forward = forward / forwardLength;
  } else {
    forward = Vec3(0.0f, 0.0f, -1.0f);
  }

  struct ScreenCorner {
    Vec3 world = Vec3(0.0f);
    float x = 0.0f;
    float y = 0.0f;
    float depth = std::numeric_limits<float>::infinity();
    bool valid = false;
  };

  std::array<ScreenCorner, 8> projectedCorners;

  const float widthScale =
      (width > 1) ? static_cast<float>(width - 1) : 0.0f;
  const float heightScale =
      (height > 1) ? static_cast<float>(height - 1) : 0.0f;

  for (int index = 0; index < 8; ++index) {
    Vec3 corner((index & 1) ? bounds.maxCorner[0] : bounds.minCorner[0],
                (index & 2) ? bounds.maxCorner[1] : bounds.minCorner[1],
                (index & 4) ? bounds.maxCorner[2] : bounds.minCorner[2]);

    const viskores::Vec4f_32 homogeneousCorner(corner[0],
                                               corner[1],
                                               corner[2],
                                               1.0f);
    const viskores::Vec4f_32 viewSpace =
        viskores::MatrixMultiply(view, homogeneousCorner);
    const viskores::Vec4f_32 clipSpace =
        viskores::MatrixMultiply(projection, viewSpace);
    const float w = clipSpace[3];
    ScreenCorner projected;
    projected.world = corner;

    if (w <= 0.0f || !std::isfinite(w)) {
      projectedCorners[static_cast<std::size_t>(index)] = projected;
      continue;
    }

    const float invW = 1.0f / w;
    const float ndcX = clipSpace[0] * invW;
    const float ndcY = clipSpace[1] * invW;
    const float ndcZ = clipSpace[2] * invW;

    if (!std::isfinite(ndcX) || !std::isfinite(ndcY) ||
        !std::isfinite(ndcZ)) {
      projectedCorners[static_cast<std::size_t>(index)] = projected;
      continue;
    }

    projected.x = (ndcX * 0.5f + 0.5f) * widthScale;
    projected.y = (ndcY * 0.5f + 0.5f) * heightScale;
    projected.depth = viskores::Dot(corner - camera.eye, forward);
    if (!std::isfinite(projected.depth)) {
      projected.depth = camera.farPlane;
    }
    projected.valid = true;
    projectedCorners[static_cast<std::size_t>(index)] = projected;
  }

  constexpr std::array<std::pair<int, int>, 12> edges = {
      std::pair<int, int>{0, 1}, std::pair<int, int>{1, 3},
      std::pair<int, int>{3, 2}, std::pair<int, int>{2, 0},
      std::pair<int, int>{4, 5}, std::pair<int, int>{5, 7},
      std::pair<int, int>{7, 6}, std::pair<int, int>{6, 4},
      std::pair<int, int>{0, 4}, std::pair<int, int>{1, 5},
      std::pair<int, int>{2, 6}, std::pair<int, int>{3, 7}};

  const Color baseLineColor(1.0f, 1.0f, 1.0f, 1.0f);

  const auto blendSample = [&](int pixelX,
                               int pixelY,
                               float coverage,
                               float depth) {
    if (pixelX < 0 || pixelX >= width || pixelY < 0 || pixelY >= height) {
      return;
    }

    const float clampedCoverage =
        std::clamp(coverage, 0.0f, 1.0f);
    if (clampedCoverage <= 0.0f) {
      return;
    }

    const int pixelIndex = layer.pixelIndex(pixelX, pixelY);
    auto* buffer = layer.getColorBuffer(pixelIndex);

    const float srcAlpha = clampedCoverage;
    const float srcRed = baseLineColor.Components[0] * srcAlpha;
    const float srcGreen = baseLineColor.Components[1] * srcAlpha;
    const float srcBlue = baseLineColor.Components[2] * srcAlpha;

    buffer[0] = srcRed + buffer[0] * (1.0f - srcAlpha);
    buffer[1] = srcGreen + buffer[1] * (1.0f - srcAlpha);
    buffer[2] = srcBlue + buffer[2] * (1.0f - srcAlpha);
    buffer[3] = srcAlpha + buffer[3] * (1.0f - srcAlpha);
    buffer[4] = std::min(buffer[4], depth);
  };

  const auto lerp = [](float v0, float v1, float t) {
    return v0 + (v1 - v0) * t;
  };

  const float pixelRadius =
      0.5f * static_cast<float>(std::max(sqrtAntialiasing, 1));
  const float influenceRadius = pixelRadius + 0.5f;

  for (const auto& edge : edges) {
    const ScreenCorner& start =
        projectedCorners[static_cast<std::size_t>(edge.first)];
    const ScreenCorner& end =
        projectedCorners[static_cast<std::size_t>(edge.second)];
    if (!start.valid || !end.valid) {
      continue;
    }

    const float minX =
        std::min(start.x, end.x) - influenceRadius;
    const float maxX =
        std::max(start.x, end.x) + influenceRadius;
    const float minY =
        std::min(start.y, end.y) - influenceRadius;
    const float maxY =
        std::max(start.y, end.y) + influenceRadius;

    const int xBegin = std::max(
        0, static_cast<int>(std::floor(minX)));
    const int xEnd = std::min(
        width - 1, static_cast<int>(std::ceil(maxX)));
    const int yBegin = std::max(
        0, static_cast<int>(std::floor(minY)));
    const int yEnd = std::min(
        height - 1, static_cast<int>(std::ceil(maxY)));

    const float edgeDx = end.x - start.x;
    const float edgeDy = end.y - start.y;
    const float edgeLenSquared = edgeDx * edgeDx + edgeDy * edgeDy;
    if (!(edgeLenSquared > 0.0f)) {
      const float depth = std::min(start.depth, end.depth);
      blendSample(static_cast<int>(std::lround(start.x)),
                  static_cast<int>(std::lround(start.y)),
                  1.0f,
                  depth);
      continue;
    }

    for (int py = yBegin; py <= yEnd; ++py) {
      const float sampleY = static_cast<float>(py) + 0.5f;
      for (int px = xBegin; px <= xEnd; ++px) {
        const float sampleX = static_cast<float>(px) + 0.5f;

        const float apx = sampleX - start.x;
        const float apy = sampleY - start.y;
        float t = 0.0f;
        if (edgeLenSquared > 0.0f) {
          t = (apx * edgeDx + apy * edgeDy) / edgeLenSquared;
        }
        t = std::clamp(t, 0.0f, 1.0f);

        const float closestX = lerp(start.x, end.x, t);
        const float closestY = lerp(start.y, end.y, t);
        const float distX = sampleX - closestX;
        const float distY = sampleY - closestY;
        const float distance =
            std::sqrt(distX * distX + distY * distY);

        const float coverage =
            std::clamp(pixelRadius + 0.5f - distance, 0.0f, 1.0f);
        if (coverage <= 0.0f) {
          continue;
        }

        Vec3 worldPoint =
            start.world + (end.world - start.world) * t;
        float depth = viskores::Dot(worldPoint - camera.eye, forward);
        if (!std::isfinite(depth)) {
          depth = lerp(start.depth, end.depth, t);
        }
        if (!std::isfinite(depth)) {
          depth = camera.farPlane;
        }

        blendSample(px, py, coverage, depth);
      }
    }
  }
}

struct ParsedOptions {
  ViskoresVolumeRenderer::RenderParameters parameters;
  std::string outputFilename = "viskores-volume-trial.ppm";
  bool exitEarly = false;
};

void printUsage() {
  std::cout << "Usage: ViskoresVolumeRenderer [--width W] [--height H] "
               "[--trials N] [--antialiasing A] [--output FILE]\n"
            << "  --width W        Image width (default: 512)\n"
            << "  --height H       Image height (default: 512)\n"
            << "  --trials N       Number of render trials (default: 1)\n"
            << "  --antialiasing A Supersampling factor (positive integer square, default: 1)\n"
            << "  --box-transparency T  Transparency factor per box in [0,1] "
               "(default: 0)\n"
            << "  --visibility-graph  Enable topological ordering using a visibility "
               "graph (default)\n"
            << "  --no-visibility-graph  Disable topological ordering using a "
               "visibility graph\n"
            << "  --write-visibility-graph  Export visibility graph DOT files (default: disabled)\n"
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
    } else if (arg == "--box-transparency") {
      parsed.parameters.boxTransparency = std::stof(requireValue(arg));
      if (parsed.parameters.boxTransparency < 0.0f ||
          parsed.parameters.boxTransparency > 1.0f) {
        throw std::runtime_error(
            "box transparency must be between 0 and 1");
      }
    } else if (arg == "--antialiasing") {
      parsed.parameters.antialiasing = std::stoi(requireValue(arg));
      if (parsed.parameters.antialiasing <= 0) {
        throw std::runtime_error("antialiasing must be positive");
      }
    } else if (arg == "--visibility-graph") {
      parsed.parameters.useVisibilityGraph = true;
    } else if (arg == "--no-visibility-graph") {
      parsed.parameters.useVisibilityGraph = false;
    } else if (arg == "--write-visibility-graph") {
      parsed.parameters.writeVisibilityGraph = true;
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

std::unique_ptr<ImageFull> downsampleImage(const ImageFull& source,
                                           int targetWidth,
                                           int targetHeight,
                                           int sqrtAA) {
  const int blockSize = std::max(sqrtAA, 1);
  if (blockSize <= 1) {
    throw std::invalid_argument(
        "downsampleImage expects sqrtAA > 1 for downsampling");
  }

  const auto* typedSource =
      dynamic_cast<const ImageRGBAFloatColorDepthSort*>(&source);
  if (typedSource == nullptr) {
    throw std::runtime_error(
        "downsampleImage expects ImageRGBAFloatColorDepthSort input.");
  }

  auto downsampled =
      std::make_unique<ImageRGBAFloatColorDepthSort>(targetWidth,
                                                     targetHeight);
  const float invSamples =
      1.0f / static_cast<float>(blockSize * blockSize);

  for (int y = 0; y < targetHeight; ++y) {
    for (int x = 0; x < targetWidth; ++x) {
      float sumR = 0.0f;
      float sumG = 0.0f;
      float sumB = 0.0f;
      float sumA = 0.0f;
      for (int dy = 0; dy < blockSize; ++dy) {
        const int srcY = y * blockSize + dy;
        for (int dx = 0; dx < blockSize; ++dx) {
          const int srcX = x * blockSize + dx;
          const Color sample = typedSource->getColor(srcX, srcY);
          sumR += sample.Components[0];
          sumG += sample.Components[1];
          sumB += sample.Components[2];
          sumA += sample.Components[3];
        }
      }

      Color averaged(sumR * invSamples,
                     sumG * invSamples,
                     sumB * invSamples,
                     sumA * invSamples);
      downsampled->setColor(x, y, averaged);
      downsampled->setDepthHint(
          x, y, std::numeric_limits<float>::infinity());
    }
  }

  return downsampled;
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

float computeBoxDepthHint(const AmrBox& box,
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

ViskoresVolumeRenderer::ViskoresVolumeRenderer()
    : rank(0),
      numProcs(1) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
}

void ViskoresVolumeRenderer::validateRenderParameters(
    const RenderParameters& parameters) const {
  if (parameters.width <= 0 || parameters.height <= 0) {
    throw std::invalid_argument("image dimensions must be positive");
  }
  if (parameters.trials <= 0) {
    throw std::invalid_argument("number of trials must be positive");
  }
  if (parameters.boxTransparency < 0.0f ||
      parameters.boxTransparency > 1.0f) {
    throw std::invalid_argument(
        "box transparency must be between 0 and 1");
  }
  if (parameters.antialiasing <= 0) {
    throw std::invalid_argument("antialiasing must be positive");
  }
  const int sqrtAA = static_cast<int>(std::lround(std::sqrt(parameters.antialiasing)));
  if (sqrtAA * sqrtAA != parameters.antialiasing) {
    throw std::invalid_argument(
        "antialiasing must be a perfect square (1, 4, 9, ...)");
  }
}

void ViskoresVolumeRenderer::initialize() const {
  if (rank == 0) {
    std::cout << "ViskoresVolumeRenderer: Using Viskores volume mapper on "
              << numProcs << " ranks" << std::endl;
  }
}

ViskoresVolumeRenderer::SceneGeometry
ViskoresVolumeRenderer::createRankSpecificGeometry() const {
  constexpr int boxesX = 2;
  constexpr int boxesY = 2;
  constexpr int boxesZ = 2;
  constexpr int totalBoxes = boxesX * boxesY * boxesZ;
  constexpr float boxScale = 0.8f;
  constexpr float spacing = boxScale;  // centers are one box width apart

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

    AmrBox box;
    box.minCorner = offset - halfExtent;
    box.maxCorner = offset + halfExtent;

    const int refinementCycle = boxIndex % 3;
    const Id baseResolution = 8;
    const Id cellsPerAxis = baseResolution + static_cast<Id>(refinementCycle * 4);
    box.cellDimensions =
        viskores::Id3(cellsPerAxis, cellsPerAxis, cellsPerAxis);

    const float cellValue = static_cast<float>(boxIndex);
    const std::size_t valueCount =
        static_cast<std::size_t>(box.cellDimensions[0]) *
        static_cast<std::size_t>(box.cellDimensions[1]) *
        static_cast<std::size_t>(box.cellDimensions[2]);
    box.cellValues.assign(valueCount, cellValue);

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

ViskoresVolumeRenderer::VolumeBounds ViskoresVolumeRenderer::computeGlobalBounds(
    const std::vector<AmrBox>& boxes,
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

ViskoresVolumeRenderer::VolumeBounds ViskoresVolumeRenderer::computeTightBounds(
    const std::vector<AmrBox>& boxes,
    const VolumeBounds& fallback) const {
  Vec3 localMin(std::numeric_limits<float>::max());
  Vec3 localMax(-std::numeric_limits<float>::max());
  bool hasLocalBoxes = false;

  for (const auto& box : boxes) {
    localMin = componentMin(localMin, box.minCorner);
    localMax = componentMax(localMax, box.maxCorner);
    hasLocalBoxes = true;
  }

  if (!hasLocalBoxes) {
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

  int localHasAny = hasLocalBoxes ? 1 : 0;
  int globalHasAny = 0;
  MPI_Allreduce(&localHasAny,
                &globalHasAny,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);

  if (globalHasAny <= 0) {
    return fallback;
  }

  VolumeBounds tight;
  tight.minCorner = Vec3(globalMinArray[0],
                         globalMinArray[1],
                         globalMinArray[2]);
  tight.maxCorner = Vec3(globalMaxArray[0],
                         globalMaxArray[1],
                         globalMaxArray[2]);
  return tight;
}

std::pair<float, float> ViskoresVolumeRenderer::computeGlobalScalarRange(
    const std::vector<AmrBox>& boxes) const {
  float localMin = std::numeric_limits<float>::infinity();
  float localMax = -std::numeric_limits<float>::infinity();

  for (const auto& box : boxes) {
    if (!box.cellValues.empty()) {
      const auto [minIt, maxIt] =
          std::minmax_element(box.cellValues.begin(), box.cellValues.end());
      localMin = std::min(localMin, *minIt);
      localMax = std::max(localMax, *maxIt);
    }
  }

  if (!std::isfinite(localMin) || !std::isfinite(localMax)) {
    localMin = 0.0f;
    localMax = 0.0f;
  }

  float globalMin = 0.0f;
  float globalMax = 0.0f;
  MPI_Allreduce(&localMin, &globalMin, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&localMax, &globalMax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

  if (globalMin == globalMax) {
    globalMax = globalMin + 1.0f;
  }

  return {globalMin, globalMax};
}

void ViskoresVolumeRenderer::paint(const AmrBox& box,
                                   const VolumeBounds& bounds,
                                   const std::pair<float, float>& scalarRange,
                                   float boxTransparency,
                                   int antialiasing,
                                   ImageFull& image,
                                   const CameraParameters& camera) {
  static VolumePainterViskores painter;
  painter.paint(box,
                bounds,
                scalarRange,
                rank,
                numProcs,
                boxTransparency,
                antialiasing,
                image,
                camera);
}

Compositor* ViskoresVolumeRenderer::getCompositor() {
  static DirectSendBase compositor;
  return &compositor;
}

MPI_Group ViskoresVolumeRenderer::buildVisibilityOrderedGroup(
    const CameraParameters& camera,
    float aspect,
    MPI_Group baseGroup,
    bool useVisibilityGraph,
    bool writeVisibilityGraph,
    const std::vector<AmrBox>& localBoxes) const {
  return BuildVisibilityOrderedGroup(camera,
                                     aspect,
                                     baseGroup,
                                     rank,
                                     numProcs,
                                     useVisibilityGraph,
                                     writeVisibilityGraph,
                                     localBoxes,
                                     MPI_COMM_WORLD);
}

int ViskoresVolumeRenderer::renderScene(
    const std::string& outputFilenameBase,
    const RenderParameters& parameters,
    const SceneGeometry& geometry) {
  validateRenderParameters(parameters);
  initialize();

  VolumeBounds bounds =
      computeGlobalBounds(geometry.localBoxes,
                          geometry.hasExplicitBounds,
                          geometry.explicitBounds);
  const std::pair<float, float> scalarRange =
      computeGlobalScalarRange(geometry.localBoxes);

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

  const float fovY = viskores::Pif() * 0.25f;
  const float maxAltitude = viskores::Pif() * 0.25f;

  for (int trial = 0; trial < parameters.trials; ++trial) {
    const float cameraDistance =
        boundingRadius / std::tan(fovY * 0.5f) + boundingRadius * 1.5f;

    std::mt19937 trialRng(parameters.cameraSeed +
                          static_cast<unsigned int>(trial));
    std::uniform_real_distribution<float> azimuthDistribution(
        0.0f, viskores::TwoPif());
    std::uniform_real_distribution<float> altitudeDistribution(
        -maxAltitude, maxAltitude);
    const float azimuth = azimuthDistribution(trialRng);
    const float altitude = altitudeDistribution(trialRng);
    const float cosAltitude = std::cos(altitude);

    const Vec3 eye(
        center[0] + cameraDistance * cosAltitude * std::sin(azimuth),
        center[1] + cameraDistance * std::sin(altitude),
        center[2] + cameraDistance * cosAltitude * std::cos(azimuth));

    const float nearPlane = 0.1f;
    const float farPlane = cameraDistance * 4.0f;
    CameraParameters camera{
        eye,
        center,
        Vec3(0.0f, 1.0f, 0.0f),
        fovY * 180.0f / viskores::Pif(),
        nearPlane,
        farPlane};

    const std::string trialFilename =
        buildTrialFilename(outputFilenameBase, trial, parameters.trials);
    const int result = renderSingleTrial(trialFilename,
                                         parameters,
                                         geometry,
                                         bounds,
                                         scalarRange,
                                         compositor,
                                         groupGuard.group,
                                         camera,
                                         trial);
    if (result != 0) {
      return result;
    }
  }

  return 0;
}

int ViskoresVolumeRenderer::renderScene(
    const std::string& outputFilenameBase,
    const RenderParameters& parameters,
    const SceneGeometry& geometry,
    const CameraParameters& camera) {
  validateRenderParameters(parameters);
  initialize();

  VolumeBounds bounds =
      computeGlobalBounds(geometry.localBoxes,
                          geometry.hasExplicitBounds,
                          geometry.explicitBounds);
  const std::pair<float, float> scalarRange =
      computeGlobalScalarRange(geometry.localBoxes);

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

  for (int trial = 0; trial < parameters.trials; ++trial) {
    const std::string trialFilename =
        buildTrialFilename(outputFilenameBase, trial, parameters.trials);
    const int result = renderSingleTrial(trialFilename,
                                         parameters,
                                         geometry,
                                         bounds,
                                         scalarRange,
                                         compositor,
                                         groupGuard.group,
                                         camera,
                                         trial);
    if (result != 0) {
      return result;
    }
  }

  return 0;
}

int ViskoresVolumeRenderer::renderSingleTrial(
    const std::string& outputFilename,
    const RenderParameters& parameters,
    const SceneGeometry& geometry,
  const VolumeBounds& bounds,
  const std::pair<float, float>& scalarRange,
  Compositor* compositor,
  MPI_Group baseGroup,
  const CameraParameters& camera,
    int trialIndex) {
  const float aspect = static_cast<float>(parameters.width) /
                       static_cast<float>(parameters.height);
  const int sqrtAntialiasing =
      static_cast<int>(std::lround(std::sqrt(parameters.antialiasing)));
  const int renderWidth =
      parameters.width * std::max(sqrtAntialiasing, 1);
  const int renderHeight =
      parameters.height * std::max(sqrtAntialiasing, 1);

  const VolumeBounds tightBounds =
      computeTightBounds(geometry.localBoxes, bounds);

  std::vector<std::unique_ptr<ImageRGBAFloatColorDepthSort>> localLayers;
  localLayers.reserve(geometry.localBoxes.size() + 1);
  std::vector<float> depthHints;
  depthHints.reserve(geometry.localBoxes.size() + 1);

  for (std::size_t boxIndex = 0; boxIndex < geometry.localBoxes.size();
       ++boxIndex) {
    const AmrBox& box = geometry.localBoxes[boxIndex];

    auto layerImage =
        std::make_unique<ImageRGBAFloatColorDepthSort>(renderWidth,
                                                       renderHeight);
    paint(box,
          bounds,
          scalarRange,
          parameters.boxTransparency,
          parameters.antialiasing,
          *layerImage,
          camera);
    depthHints.push_back(computeBoxDepthHint(box, camera));
    localLayers.push_back(std::move(layerImage));
  }

  auto boundingBoxLayer =
      std::make_unique<ImageRGBAFloatColorDepthSort>(renderWidth,
                                                     renderHeight);
  boundingBoxLayer->clear();
  clearDepthHints(*boundingBoxLayer,
                  std::numeric_limits<float>::infinity());
  renderBoundingBoxLayer(tightBounds,
                         camera,
                         sqrtAntialiasing,
                         *boundingBoxLayer);
  AmrBox boundsBox;
  boundsBox.minCorner = tightBounds.minCorner;
  boundsBox.maxCorner = tightBounds.maxCorner;
  depthHints.push_back(computeBoxDepthHint(boundsBox, camera));
  localLayers.push_back(std::move(boundingBoxLayer));

  auto prototype =
      std::make_unique<ImageRGBAFloatColorDepthSort>(renderWidth,
                                                     renderHeight);

  LayeredVolumeImage layeredImage(renderWidth,
                                  renderHeight,
                                  std::move(localLayers),
                                  std::move(depthHints),
                                  std::move(prototype));

  MPI_Group orderedGroup =
      buildVisibilityOrderedGroup(camera,
                                  aspect,
                                  baseGroup,
                                  parameters.useVisibilityGraph,
                                  parameters.writeVisibilityGraph,
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
      const bool hasTrialInfo = trialIndex >= 0;
      if (hasTrialInfo) {
        std::cout << "Trial " << trialIndex << ": composed " << pixels
                  << " pixels on rank 0" << std::endl;
      } else {
        std::cout << "Render: composed " << pixels
                  << " pixels on rank 0" << std::endl;
      }

      std::unique_ptr<ImageFull> outputImage;
      if (sqrtAntialiasing > 1) {
        outputImage =
            downsampleImage(*gatheredImage,
                            parameters.width,
                            parameters.height,
                            sqrtAntialiasing);
      } else {
        outputImage = std::move(gatheredImage);
      }

      const bool saved = SavePPM(*outputImage, outputFilename);
      if (hasTrialInfo) {
        if (saved) {
          std::cout << "Saved trial " << trialIndex
                    << " volume composited image to '" << outputFilename
                    << "'" << std::endl;
        } else {
          std::cerr << "Failed to save trial " << trialIndex
                    << " composited image to '" << outputFilename << "'"
                    << std::endl;
        }
      } else {
        if (saved) {
          std::cout << "Saved volume composited image to '"
                    << outputFilename << "'" << std::endl;
        } else {
          std::cerr << "Failed to save composited image to '"
                    << outputFilename << "'" << std::endl;
        }
      }
    }
  }

  return 0;
}

int ViskoresVolumeRenderer::run(int argc, char** argv) {
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
    ViskoresVolumeRenderer example;
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
