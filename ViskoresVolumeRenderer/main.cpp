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

#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>

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
  std::string plotfilePath;
  std::string variableName;
  int minLevel = 0;
  int maxLevel = -1;
  bool logScaleInput = false;
  bool exitEarly = false;
};

void printUsage() {
  std::cout << "Usage: ViskoresVolumeRenderer [options] plotfile\n"
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
            << "  --variable NAME  Scalar variable to render (default: first variable in plotfile)\n"
            << "  --max-level L    Finest AMR level to include (default: plotfile finest level)\n"
            << "  --min-level L    Coarsest AMR level to include (default: 0)\n"
            << "  --up-vector X Y Z  Camera up vector components (default: 0 1 0)\n"
            << "  --log-scale      Apply natural log scaling before normalizing the input field\n"
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
    } else if (arg == "--variable") {
      parsed.variableName = requireValue(arg);
      if (parsed.variableName.empty()) {
        throw std::runtime_error("variable name must not be empty");
      }
    } else if (arg == "--min-level") {
      parsed.minLevel = std::stoi(requireValue(arg));
      if (parsed.minLevel < 0) {
        throw std::runtime_error("min level must be non-negative");
      }
    } else if (arg == "--max-level") {
      parsed.maxLevel = std::stoi(requireValue(arg));
      if (parsed.maxLevel < 0) {
        throw std::runtime_error("max level must be non-negative");
      }
    } else if (arg == "--log-scale") {
      parsed.logScaleInput = true;
    } else if (arg == "--up-vector") {
      if (i + 3 >= argc) {
        throw std::runtime_error("--up-vector requires three components");
      }
      const float x = std::stof(argv[++i]);
      const float y = std::stof(argv[++i]);
      const float z = std::stof(argv[++i]);
      Vec3 upVector(x, y, z);
      const float length = viskores::Magnitude(upVector);
      if (!(length > 0.0f) || !std::isfinite(length)) {
        throw std::runtime_error("--up-vector must be non-zero and finite");
      }
      parsed.parameters.cameraUp = upVector / length;
      parsed.parameters.useCustomUp = true;
    } else if (arg == "--plotfile") {
      parsed.plotfilePath = requireValue(arg);
    } else if (arg == "--help" || arg == "-h") {
      if (rank == 0) {
        printUsage();
      }
      parsed.exitEarly = true;
      return parsed;
    } else {
      if (!arg.empty() && arg[0] == '-') {
        std::ostringstream message;
        message << "unknown option '" << arg << "'";
        throw std::runtime_error(message.str());
      }
      if (!parsed.plotfilePath.empty()) {
        std::ostringstream message;
        message << "multiple plot files specified ('" << parsed.plotfilePath
                << "' and '" << arg << "')";
        throw std::runtime_error(message.str());
      }
      parsed.plotfilePath = arg;
    }
  }

  if (parsed.plotfilePath.empty()) {
    throw std::runtime_error("plotfile path is required");
  }
  if (parsed.maxLevel >= 0 && parsed.minLevel > parsed.maxLevel) {
    throw std::runtime_error("min level must not exceed max level");
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

auto ViskoresVolumeRenderer::loadPlotFileGeometry(
    const std::string& plotfilePath,
    const std::string& variableName,
    int requestedMinLevel,
    int requestedMaxLevel,
    bool logScaleInput) const -> SceneGeometry {
  if (plotfilePath.empty()) {
    throw std::invalid_argument("Plotfile path must not be empty.");
  }

  amrex::PlotFileData plotfile(plotfilePath);

  const int spaceDim = plotfile.spaceDim();
  if (spaceDim != 3) {
    std::ostringstream message;
    message << "Plotfile '" << plotfilePath << "' has space dimension "
            << spaceDim << ". The volume renderer currently expects 3D data.";
    throw std::runtime_error(message.str());
  }

  const auto& varNames = plotfile.varNames();
  if (varNames.empty()) {
    throw std::runtime_error("Plotfile contains no cell variables to render.");
  }

  std::string componentName = variableName;
  if (componentName.empty()) {
    componentName = varNames.front();
  } else {
    const auto it =
        std::find(varNames.begin(), varNames.end(), componentName);
    if (it == varNames.end()) {
      std::ostringstream message;
      message << "Variable '" << componentName
              << "' not found in plotfile '" << plotfilePath << "'.";
      throw std::runtime_error(message.str());
    }
  }

  const int finestLevel = plotfile.finestLevel();
  int minLevel = requestedMinLevel;
  if (minLevel < 0) {
    minLevel = 0;
  }
  if (minLevel > finestLevel) {
    minLevel = finestLevel;
  }
  int maxLevel = requestedMaxLevel;
  if (maxLevel < 0 || maxLevel > finestLevel) {
    maxLevel = finestLevel;
  }
  if (minLevel > maxLevel) {
    std::ostringstream message;
    message << "Minimum AMR level " << minLevel
            << " exceeds available maximum level " << maxLevel << ".";
    throw std::runtime_error(message.str());
  }

  SceneGeometry scene;
  scene.localBoxes.reserve(64);

  Vec3 localMinOriginal(std::numeric_limits<float>::max());
  Vec3 localMaxOriginal(-std::numeric_limits<float>::max());
  bool hasLocalBoxes = false;

  float localScalarMin = std::numeric_limits<float>::infinity();
  float localScalarMax = -std::numeric_limits<float>::infinity();
  bool hasLocalScalars = false;
  float localPositiveMin = std::numeric_limits<float>::infinity();
  bool hasLocalPositive = false;

  const amrex::Array<amrex::Real, AMREX_SPACEDIM> probLo = plotfile.probLo();

  amrex::Vector<amrex::MultiFab> levelData;
  levelData.reserve(static_cast<std::size_t>(maxLevel) + 1);
  for (int level = 0; level <= maxLevel; ++level) {
    levelData.emplace_back(plotfile.get(level, componentName));
  }

  amrex::Vector<amrex::MultiFab const*> levelPtrs;
  levelPtrs.reserve(levelData.size());
  for (auto& mf : levelData) {
    levelPtrs.push_back(&mf);
  }

  amrex::Vector<amrex::IntVect> refinementRatios;
  refinementRatios.reserve((maxLevel > 0) ? static_cast<std::size_t>(maxLevel) : 0);
  for (int level = 0; level < maxLevel; ++level) {
    amrex::IntVect ratio(plotfile.refRatio(level));
    for (int id = spaceDim; id < AMREX_SPACEDIM; ++id) {
      ratio[id] = 1;
    }
    refinementRatios.push_back(ratio);
  }

  amrex::Vector<amrex::MultiFab> convexified =
      amrex::convexify(levelPtrs, refinementRatios);

  for (int level = 0; level <= maxLevel; ++level) {
    if (level >= static_cast<int>(convexified.size())) {
      break;
    }

    amrex::MultiFab& mf = convexified[level];
    if (mf.size() == 0) {
      continue;
    }
    if (level < minLevel) {
      continue;
    }

    const amrex::Array<amrex::Real, AMREX_SPACEDIM> cellSize =
        plotfile.cellSize(level);

    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.validbox();
      const amrex::IntVect lo = box.smallEnd();
      const amrex::IntVect hi = box.bigEnd();

      const int nx = box.length(0);
      const int ny = box.length(1);
      const int nz = box.length(2);
      if (nx <= 0 || ny <= 0 || nz <= 0) {
        continue;
      }

      AmrBox amrBox;
      amrBox.cellDimensions =
          viskores::Id3(static_cast<Id>(nx),
                        static_cast<Id>(ny),
                        static_cast<Id>(nz));

      Vec3 minCorner(0.0f);
      Vec3 maxCorner(0.0f);
      minCorner[0] = static_cast<float>(
          probLo[0] + static_cast<double>(lo[0]) * cellSize[0]);
      minCorner[1] = static_cast<float>(
          probLo[1] + static_cast<double>(lo[1]) * cellSize[1]);
      minCorner[2] = static_cast<float>(
          probLo[2] + static_cast<double>(lo[2]) * cellSize[2]);
      maxCorner[0] = static_cast<float>(
          probLo[0] + static_cast<double>(hi[0] + 1) * cellSize[0]);
      maxCorner[1] = static_cast<float>(
          probLo[1] + static_cast<double>(hi[1] + 1) * cellSize[1]);
      maxCorner[2] = static_cast<float>(
          probLo[2] + static_cast<double>(hi[2] + 1) * cellSize[2]);

      localMinOriginal = componentMin(localMinOriginal, minCorner);
      localMaxOriginal = componentMax(localMaxOriginal, maxCorner);
      amrBox.minCorner = minCorner;
      amrBox.maxCorner = maxCorner;
      hasLocalBoxes = true;

      const std::size_t valueCount =
          static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
          static_cast<std::size_t>(nz);
      amrBox.cellValues.resize(valueCount);

      const auto data = mf.const_array(mfi);

      for (int k = 0; k < nz; ++k) {
        const int globalK = lo[2] + k;
        for (int j = 0; j < ny; ++j) {
          const int globalJ = lo[1] + j;
          for (int i = 0; i < nx; ++i) {
            const int globalI = lo[0] + i;
            float value =
                static_cast<float>(data(globalI, globalJ, globalK));
            if (!std::isfinite(value)) {
              value = 0.0f;
            } else {
              if (logScaleInput && value > 0.0f) {
                localPositiveMin = std::min(localPositiveMin, value);
                hasLocalPositive = true;
              }
              localScalarMin = std::min(localScalarMin, value);
              localScalarMax = std::max(localScalarMax, value);
              hasLocalScalars = true;
            }

            const std::size_t index =
                (static_cast<std::size_t>(k) * static_cast<std::size_t>(ny) +
                 static_cast<std::size_t>(j)) *
                    static_cast<std::size_t>(nx) +
                static_cast<std::size_t>(i);
            amrBox.cellValues[index] = value;
          }
        }
      }

      scene.localBoxes.push_back(std::move(amrBox));
    }
  }

  if (!hasLocalBoxes) {
    localMinOriginal = Vec3(std::numeric_limits<float>::max());
    localMaxOriginal = Vec3(-std::numeric_limits<float>::max());
  }

  std::array<float, 3> localMinOriginalArray = {localMinOriginal[0],
                                                localMinOriginal[1],
                                                localMinOriginal[2]};
  std::array<float, 3> localMaxOriginalArray = {localMaxOriginal[0],
                                                localMaxOriginal[1],
                                                localMaxOriginal[2]};
  std::array<float, 3> globalMinOriginalArray = {
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max(),
      std::numeric_limits<float>::max()};
  std::array<float, 3> globalMaxOriginalArray = {
      -std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max(),
      -std::numeric_limits<float>::max()};

  MPI_Allreduce(localMinOriginalArray.data(),
                globalMinOriginalArray.data(),
                3,
                MPI_FLOAT,
                MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(localMaxOriginalArray.data(),
                globalMaxOriginalArray.data(),
                3,
                MPI_FLOAT,
                MPI_MAX,
                MPI_COMM_WORLD);

  Vec3 globalMinOriginal(globalMinOriginalArray[0],
                         globalMinOriginalArray[1],
                         globalMinOriginalArray[2]);
  Vec3 globalMaxOriginal(globalMaxOriginalArray[0],
                         globalMaxOriginalArray[1],
                         globalMaxOriginalArray[2]);

  Vec3 globalExtentOriginal = globalMaxOriginal - globalMinOriginal;
  float minExtent = std::numeric_limits<float>::max();
  for (int component = 0; component < 3; ++component) {
    const float axisLength = std::fabs(globalExtentOriginal[component]);
    if (axisLength > 0.0f && std::isfinite(axisLength)) {
      minExtent = std::min(minExtent, axisLength);
    }
  }

  float globalScale = 1.0f;
  if (minExtent > 0.0f && std::isfinite(minExtent)) {
    globalScale = 1.0f / minExtent;
  }
  if (!std::isfinite(globalScale) || globalScale <= 0.0f) {
    globalScale = 1.0f;
  }

  if (globalScale != 1.0f) {
    for (auto& box : scene.localBoxes) {
      box.minCorner[0] *= globalScale;
      box.minCorner[1] *= globalScale;
      box.minCorner[2] *= globalScale;
      box.maxCorner[0] *= globalScale;
      box.maxCorner[1] *= globalScale;
      box.maxCorner[2] *= globalScale;
    }
    globalMinOriginal[0] *= globalScale;
    globalMinOriginal[1] *= globalScale;
    globalMinOriginal[2] *= globalScale;
    globalMaxOriginal[0] *= globalScale;
    globalMaxOriginal[1] *= globalScale;
    globalMaxOriginal[2] *= globalScale;
  }

  Vec3 localMin(std::numeric_limits<float>::max());
  Vec3 localMax(-std::numeric_limits<float>::max());
  for (const auto& box : scene.localBoxes) {
    localMin = componentMin(localMin, box.minCorner);
    localMax = componentMax(localMax, box.maxCorner);
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

  Vec3 globalMin(globalMinArray[0],
                 globalMinArray[1],
                 globalMinArray[2]);
  Vec3 globalMax(globalMaxArray[0],
                 globalMaxArray[1],
                 globalMaxArray[2]);

  const bool invalidBounds = (globalMin[0] > globalMax[0]) ||
                             (globalMin[1] > globalMax[1]) ||
                             (globalMin[2] > globalMax[2]);
  if (invalidBounds) {
    throw std::runtime_error(
        "Failed to locate any volumetric data within the plotfile.");
  }

  const Vec3 extent = globalMax - globalMin;
  const float maxExtent = std::max(
      extent[0], std::max(extent[1], extent[2]));
  const float paddingAmount =
      (maxExtent > 0.0f) ? maxExtent * 0.05f : 1.0f;
  const Vec3 padding(paddingAmount);

  scene.explicitBounds.minCorner = globalMin - padding;
  scene.explicitBounds.maxCorner = globalMax + padding;
  scene.hasExplicitBounds = true;

  if (logScaleInput) {
    float positiveMinSend =
        hasLocalPositive ? localPositiveMin
                         : std::numeric_limits<float>::infinity();
    float globalPositiveMin = positiveMinSend;
    MPI_Allreduce(&positiveMinSend,
                  &globalPositiveMin,
                  1,
                  MPI_FLOAT,
                  MPI_MIN,
                  MPI_COMM_WORLD);

    int localPositiveCount = hasLocalPositive ? 1 : 0;
    int globalPositiveCount = 0;
    MPI_Allreduce(&localPositiveCount,
                  &globalPositiveCount,
                  1,
                  MPI_INT,
                  MPI_SUM,
                  MPI_COMM_WORLD);

    if (globalPositiveCount <= 0 || !std::isfinite(globalPositiveMin) ||
        !(globalPositiveMin > 0.0f)) {
      throw std::runtime_error(
          "Log scaling requested but no positive scalar values were found.");
    }

    localScalarMin = std::numeric_limits<float>::infinity();
    localScalarMax = -std::numeric_limits<float>::infinity();
    hasLocalScalars = false;

    for (auto& box : scene.localBoxes) {
      for (float& value : box.cellValues) {
        float sanitized = value;
        if (!std::isfinite(sanitized) || !(sanitized > 0.0f)) {
          sanitized = globalPositiveMin;
        } else if (sanitized < globalPositiveMin) {
          sanitized = globalPositiveMin;
        }
        const float logValue = std::log(sanitized);
        value = logValue;
        if (std::isfinite(logValue)) {
          localScalarMin = std::min(localScalarMin, logValue);
          localScalarMax = std::max(localScalarMax, logValue);
          hasLocalScalars = true;
        }
      }
    }
  }

  float scalarMinSend =
      hasLocalScalars ? localScalarMin : std::numeric_limits<float>::infinity();
  float scalarMaxSend =
      hasLocalScalars ? localScalarMax : -std::numeric_limits<float>::infinity();
  float globalScalarMin = scalarMinSend;
  float globalScalarMax = scalarMaxSend;

  MPI_Allreduce(&scalarMinSend,
                &globalScalarMin,
                1,
                MPI_FLOAT,
                MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(&scalarMaxSend,
                &globalScalarMax,
                1,
                MPI_FLOAT,
                MPI_MAX,
                MPI_COMM_WORLD);

  int localHasScalars = hasLocalScalars ? 1 : 0;
  int globalHasScalars = 0;
  MPI_Allreduce(&localHasScalars,
                &globalHasScalars,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);

  if (globalHasScalars <= 0 || !std::isfinite(globalScalarMin) ||
      !std::isfinite(globalScalarMax)) {
    throw std::runtime_error(
        "Failed to compute a valid scalar range from the plotfile.");
  }
  if (globalScalarMin == globalScalarMax) {
    globalScalarMax = globalScalarMin + 1.0f;
  }

  scene.scalarRange = {globalScalarMin, globalScalarMax};
  scene.hasScalarRange = true;

  const float rangeWidth = globalScalarMax - globalScalarMin;
  if (rangeWidth > 0.0f && std::isfinite(rangeWidth)) {
    for (auto& box : scene.localBoxes) {
      for (float& value : box.cellValues) {
        const float normalized =
            (value - globalScalarMin) / rangeWidth;
        value = std::clamp(normalized, 0.0f, 1.0f);
      }
    }
    scene.scalarRange = {0.0f, 1.0f};
  }

  scene.localBoxes.shrink_to_fit();

  if (rank == 0) {
    const int includedLevels = maxLevel - minLevel + 1;
    std::cout << "Loaded plotfile '" << plotfilePath << "' with variable '"
              << componentName << "' across " << includedLevels
              << " level(s)";
    if (minLevel > 0 || maxLevel < finestLevel) {
      std::cout << " (levels " << minLevel << "-" << maxLevel << ")";
    }
    std::cout << "; normalized scalar range [0, 1]";
    if (logScaleInput) {
      std::cout << " (log scaled)";
    }
    std::cout << std::endl;
  }

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
      geometry.hasScalarRange ? geometry.scalarRange
                              : computeGlobalScalarRange(geometry.localBoxes);

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
    const float halfFov = fovY * 0.5f;
    const float minDistance =
        (halfFov > 0.0f)
            ? boundingRadius / static_cast<float>(std::tan(halfFov))
            : boundingRadius;
    const float safetyMargin =
        std::max(0.25f * boundingRadius, 0.5f);
    const float cameraDistance = minDistance + safetyMargin;

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

    Vec3 upVector = parameters.useCustomUp ? parameters.cameraUp
                                           : Vec3(0.0f, 1.0f, 0.0f);
    const Vec3 viewDir = viskores::Normal(center - eye);
    if (viskores::Magnitude(viskores::Cross(viewDir, upVector)) <= 1e-4f) {
      upVector = Vec3(0.0f, 0.0f, 1.0f);
      if (viskores::Magnitude(viskores::Cross(viewDir, upVector)) <= 1e-4f) {
        upVector = Vec3(1.0f, 0.0f, 0.0f);
      }
    }
    upVector = viskores::Normal(upVector);

    const float nearPlane = 0.1f;
    const float farPlane = cameraDistance * 4.0f;
    CameraParameters camera{
        eye,
        center,
        upVector,
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
      geometry.hasScalarRange ? geometry.scalarRange
                              : computeGlobalScalarRange(geometry.localBoxes);

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

  SceneGeometry geometry = loadPlotFileGeometry(options.plotfilePath,
                                                options.variableName,
                                                options.minLevel,
                                                options.maxLevel,
                                                options.logScaleInput);
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
  amrex::Initialize(argc, argv, false, MPI_COMM_WORLD);

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

  amrex::Finalize();
  MPI_Finalize();
  return exitCode;
}
