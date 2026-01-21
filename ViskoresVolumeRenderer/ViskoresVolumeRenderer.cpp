#include "ViskoresVolumeRenderer.hpp"

#include <AMReX.H>
#include <AMReX_MultiFabUtil.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_Utility.H>
#include <AMReX_Math.H>
#include <AMReX_RealVect.H>
#include <AMReX_SmallMatrix.H>

#include <Common/Color.hpp>
#include <Common/ImageRGBAFloatColorDepthSort.hpp>
#include <Common/ImageSparse.hpp>
#include <Common/LayeredVolumeImage.hpp>
#include <Common/SavePNG.hpp>
#include <Common/SavePPM.hpp>
#include <Common/VisibilityOrdering.hpp>
#include <Common/VolumePainterViskores.hpp>
#include <Common/VolumeTypes.hpp>
#include <DirectSend/Base/DirectSendBase.hpp>
#include <algorithm>
#include <chrono>
#include <array>
#include <cmath>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <utility>
#include <vector>

namespace {

using Vec3 = amrex::RealVect;
using Vec4 = amrex::SmallMatrix<float, 4, 1>;
using Matrix4x4 = amrex::SmallMatrix<float, 4, 4>;
using amrVolumeRenderer::volume::AmrBox;
using amrVolumeRenderer::volume::CameraParameters;
using amrVolumeRenderer::volume::VolumeBounds;

constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 2.0f * kPi;

Vec3 safeNormalize(const Vec3& input) {
  const amrex::Real length = input.vectorLength();
  if (length > 0.0 && std::isfinite(static_cast<double>(length))) {
    return input / length;
  }
  return Vec3(0.0, 0.0, -1.0);
}

Matrix4x4 makeViewMatrix(const Vec3& eye,
                         const Vec3& lookAt,
                         const Vec3& up) {
  const Vec3 forward = safeNormalize(lookAt - eye);
  Vec3 right = forward.crossProduct(up);
  const amrex::Real rightLength = right.vectorLength();
  if (rightLength > 0.0 && std::isfinite(static_cast<double>(rightLength))) {
    right /= rightLength;
  } else {
    right = Vec3(1.0, 0.0, 0.0);
  }
  const Vec3 upOrtho = right.crossProduct(forward);

  Matrix4x4 view = Matrix4x4::Identity();
  view(0, 0) = static_cast<float>(right[0]);
  view(1, 0) = static_cast<float>(right[1]);
  view(2, 0) = static_cast<float>(right[2]);
  view(3, 0) = static_cast<float>(-right.dotProduct(eye));

  view(0, 1) = static_cast<float>(upOrtho[0]);
  view(1, 1) = static_cast<float>(upOrtho[1]);
  view(2, 1) = static_cast<float>(upOrtho[2]);
  view(3, 1) = static_cast<float>(-upOrtho.dotProduct(eye));

  view(0, 2) = static_cast<float>(-forward[0]);
  view(1, 2) = static_cast<float>(-forward[1]);
  view(2, 2) = static_cast<float>(-forward[2]);
  view(3, 2) = static_cast<float>(forward.dotProduct(eye));

  view(0, 3) = 0.0f;
  view(1, 3) = 0.0f;
  view(2, 3) = 0.0f;
  view(3, 3) = 1.0f;

  return view;
}

std::string lowercaseExtension(const std::string& filename) {
  const std::size_t dot = filename.find_last_of('.');
  if (dot == std::string::npos) {
    return std::string();
  }

  std::string ext = filename.substr(dot);
  std::transform(ext.begin(),
                 ext.end(),
                 ext.begin(),
                 [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
  return ext;
}

Vec3 componentMin(const Vec3& a, const Vec3& b) {
  Vec3 result;
  result[0] = std::min(a[0], b[0]);
  result[1] = std::min(a[1], b[1]);
  result[2] = std::min(a[2], b[2]);
  return result;
}

Vec3 componentMax(const Vec3& a, const Vec3& b) {
  Vec3 result;
  result[0] = std::max(a[0], b[0]);
  result[1] = std::max(a[1], b[1]);
  result[2] = std::max(a[2], b[2]);
  return result;
}

Matrix4x4 makePerspectiveMatrix(float fovYDegrees,
                                float aspect,
                                float nearPlane,
                                float farPlane) {
  Matrix4x4 matrix = Matrix4x4::Identity();

  const float fovTangent =
      std::tan(fovYDegrees * kPi / 180.0f * 0.5f);
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

  Vec3 forward = safeNormalize(camera.lookAt - camera.eye);
  Vec3 right = forward.crossProduct(camera.up);
  const amrex::Real rightLength = right.vectorLength();
  if (rightLength > 0.0 && std::isfinite(static_cast<double>(rightLength))) {
    right /= rightLength;
  } else {
    right = Vec3(1.0, 0.0, 0.0);
  }
  const Vec3 upOrtho = right.crossProduct(forward);
  const float tanHalfFov =
      std::tan(camera.fovYDegrees * 0.5f * kPi / 180.0f);

  struct ScreenCorner {
    Vec3 world = Vec3(0.0f);
    float x = 0.0f;
    float y = 0.0f;
    float depth = std::numeric_limits<float>::infinity();
    bool valid = false;
  };

  std::array<ScreenCorner, 8> projectedCorners;
  int validCorners = 0;

  const float widthScale = (width > 1) ? static_cast<float>(width - 1) : 0.0f;
  const float heightScale =
      (height > 1) ? static_cast<float>(height - 1) : 0.0f;

  for (int index = 0; index < 8; ++index) {
    Vec3 corner((index & 1) ? bounds.maxCorner[0] : bounds.minCorner[0],
                (index & 2) ? bounds.maxCorner[1] : bounds.minCorner[1],
                (index & 4) ? bounds.maxCorner[2] : bounds.minCorner[2]);

    ScreenCorner projected;
    projected.world = corner;

    const Vec3 relative = corner - camera.eye;
    const float depth = static_cast<float>(relative.dotProduct(forward));
    if (!(depth > 0.0f) || !std::isfinite(depth)) {
      projectedCorners[static_cast<std::size_t>(index)] = projected;
      continue;
    }

    const float xCam = static_cast<float>(relative.dotProduct(right));
    const float yCam = static_cast<float>(relative.dotProduct(upOrtho));
    const float ndcX = xCam / (depth * tanHalfFov * aspect);
    const float ndcY = yCam / (depth * tanHalfFov);
    if (!std::isfinite(ndcX) || !std::isfinite(ndcY)) {
      projectedCorners[static_cast<std::size_t>(index)] = projected;
      continue;
    }

    projected.x = (ndcX * 0.5f + 0.5f) * widthScale;
    projected.y = (ndcY * 0.5f + 0.5f) * heightScale;
    projected.depth = depth;
    projected.valid = true;
    projectedCorners[static_cast<std::size_t>(index)] = projected;
    ++validCorners;
  }

  constexpr std::array<std::pair<int, int>, 12> edges = {
      std::pair<int, int>{0, 1},
      std::pair<int, int>{1, 3},
      std::pair<int, int>{3, 2},
      std::pair<int, int>{2, 0},
      std::pair<int, int>{4, 5},
      std::pair<int, int>{5, 7},
      std::pair<int, int>{7, 6},
      std::pair<int, int>{6, 4},
      std::pair<int, int>{0, 4},
      std::pair<int, int>{1, 5},
      std::pair<int, int>{2, 6},
      std::pair<int, int>{3, 7}};

  const Color baseLineColor(1.0f, 1.0f, 1.0f, 1.0f);

  const float overlayDepth = std::numeric_limits<float>::lowest();
  const auto blendSample =
      [&](int pixelX, int pixelY, float coverage, float /*depth*/) {
        if (pixelX < 0 || pixelX >= width || pixelY < 0 || pixelY >= height) {
          return;
        }

        const float clampedCoverage = std::clamp(coverage, 0.0f, 1.0f);
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
        buffer[4] = overlayDepth;
      };

  const auto lerp = [](float v0, float v1, float t) {
    return v0 + (v1 - v0) * t;
  };

  const float pixelRadius =
      0.5f * static_cast<float>(std::max(sqrtAntialiasing, 1));
  const float influenceRadius = pixelRadius + 0.5f;
  // Scale factor to adjust line thickness for better visual appearance.
  constexpr float coverageScale = 0.6f;

  for (const auto& edge : edges) {
    const ScreenCorner& start =
        projectedCorners[static_cast<std::size_t>(edge.first)];
    const ScreenCorner& end =
        projectedCorners[static_cast<std::size_t>(edge.second)];
    if (!start.valid || !end.valid) {
      continue;
    }

    const float minX = std::min(start.x, end.x) - influenceRadius;
    const float maxX = std::max(start.x, end.x) + influenceRadius;
    const float minY = std::min(start.y, end.y) - influenceRadius;
    const float maxY = std::max(start.y, end.y) + influenceRadius;

    const int xBegin = std::max(0, static_cast<int>(std::floor(minX)));
    const int xEnd = std::min(width - 1, static_cast<int>(std::ceil(maxX)));
    const int yBegin = std::max(0, static_cast<int>(std::floor(minY)));
    const int yEnd = std::min(height - 1, static_cast<int>(std::ceil(maxY)));

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
        const float distance = std::sqrt(distX * distX + distY * distY);

        const float coverage = std::clamp(
            (pixelRadius + 0.5f - distance) * coverageScale, 0.0f, 1.0f);
        if (coverage <= 0.0f) {
          continue;
        }

        Vec3 worldPoint = start.world + (end.world - start.world) * t;
        float depth = static_cast<float>(
            (worldPoint - camera.eye).dotProduct(forward));
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

using ParsedOptions = ViskoresVolumeRenderer::RunOptions;

void printUsage() {
  std::cout
      << "Usage: volume_renderer [options] plotfile\n"
      << "  --width W        Image width (default: 512)\n"
      << "  --height H       Image height (default: 512)\n"
      << "  --antialiasing A Supersampling factor (positive integer square, "
         "default: 1)\n"
      << "  --box-transparency T  Transparency factor per box in [0,1] "
         "(default: 0)\n"
      << "  --visibility-graph  Enable topological ordering using a visibility "
         "graph (default)\n"
      << "  --no-visibility-graph  Disable topological ordering using a "
         "visibility graph\n"
      << "  --write-visibility-graph  Export visibility graph DOT files "
         "(default: disabled)\n"
      << "  --variable NAME  Scalar variable to render (default: first "
         "variable in plotfile)\n"
      << "  --max-level L    Finest AMR level to include (default: plotfile "
         "finest level)\n"
      << "  --min-level L    Coarsest AMR level to include (default: 0)\n"
      << "  --up-vector X Y Z  Camera up vector components (default: 0 1 0)\n"
      << "  --print-camera   Emit the camera parameters selected automatically\n"
      << "  --log-scale      Apply natural log scaling before normalizing the "
         "input field\n"
      << "  --output FILE    Output filename (supports .ppm or .png; default: "
      << "viskores-volume.ppm)\n"
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
    } else if (arg == "--box-transparency") {
      parsed.parameters.boxTransparency = std::stof(requireValue(arg));
      if (parsed.parameters.boxTransparency < 0.0f ||
          parsed.parameters.boxTransparency > 1.0f) {
        throw std::runtime_error("box transparency must be between 0 and 1");
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
      const float length = static_cast<float>(upVector.vectorLength());
      if (!(length > 0.0f) || !std::isfinite(length)) {
        throw std::runtime_error("--up-vector must be non-zero and finite");
      }
      parsed.parameters.cameraUp = upVector / length;
      parsed.parameters.useCustomUp = true;
    } else if (arg == "--print-camera") {
      parsed.parameters.printCamera = true;
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
      std::make_unique<ImageRGBAFloatColorDepthSort>(targetWidth, targetHeight);
  const float invSamples = 1.0f / static_cast<float>(blockSize * blockSize);

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
      downsampled->setDepthHint(x, y, std::numeric_limits<float>::infinity());
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

float computeBoxDepthHint(const AmrBox& box, const CameraParameters& camera) {
  const Vec3 viewDir = safeNormalize(camera.lookAt - camera.eye);
  float minDepth = std::numeric_limits<float>::infinity();
  for (int cornerIndex = 0; cornerIndex < 8; ++cornerIndex) {
    const Vec3 corner((cornerIndex & 1) ? box.maxCorner[0] : box.minCorner[0],
                      (cornerIndex & 2) ? box.maxCorner[1] : box.minCorner[1],
                      (cornerIndex & 4) ? box.maxCorner[2] : box.minCorner[2]);
    minDepth = std::min(minDepth,
                        static_cast<float>(
                            (corner - camera.eye).dotProduct(viewDir)));
  }
  return minDepth;
}

}  // namespace

ViskoresVolumeRenderer::ViskoresVolumeRenderer() : rank(0), numProcs(1) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
}

void ViskoresVolumeRenderer::validateRenderParameters(
    const RenderParameters& parameters) const {
  if (parameters.width <= 0 || parameters.height <= 0) {
    throw std::invalid_argument("image dimensions must be positive");
  }
  if (parameters.boxTransparency < 0.0f || parameters.boxTransparency > 1.0f) {
    throw std::invalid_argument("box transparency must be between 0 and 1");
  }
  if (parameters.antialiasing <= 0) {
    throw std::invalid_argument("antialiasing must be positive");
  }
  const int sqrtAA =
      static_cast<int>(std::lround(std::sqrt(parameters.antialiasing)));
  if (sqrtAA * sqrtAA != parameters.antialiasing) {
    throw std::invalid_argument(
        "antialiasing must be a perfect square (1, 4, 9, ...)");
  }
}

void ViskoresVolumeRenderer::initialize() const {
  if (rank == 0) {
    std::cout << "volume_renderer: Using AMReX volume mapper on "
              << numProcs << " ranks" << std::endl;
  }
}

auto ViskoresVolumeRenderer::loadPlotFileGeometry(
    const std::string& plotfilePath,
    const std::string& variableName,
    int requestedMinLevel,
    int requestedMaxLevel,
    bool logScaleInput,
    bool normalizeToDataRange) const -> SceneGeometry {
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
    const auto it = std::find(varNames.begin(), varNames.end(), componentName);
    if (it == varNames.end()) {
      std::ostringstream message;
      message << "Variable '" << componentName << "' not found in plotfile '"
              << plotfilePath << "'.";
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
  refinementRatios.reserve((maxLevel > 0) ? static_cast<std::size_t>(maxLevel)
                                          : 0);
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
      amrBox.cellDimensions = amrex::IntVect(nx, ny, nz);

      Vec3 minCorner(0.0f);
      Vec3 maxCorner(0.0f);
      minCorner[0] = static_cast<float>(probLo[0] + static_cast<double>(lo[0]) *
                                                        cellSize[0]);
      minCorner[1] = static_cast<float>(probLo[1] + static_cast<double>(lo[1]) *
                                                        cellSize[1]);
      minCorner[2] = static_cast<float>(probLo[2] + static_cast<double>(lo[2]) *
                                                        cellSize[2]);
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

      const std::size_t valueCount = static_cast<std::size_t>(nx) *
                                     static_cast<std::size_t>(ny) *
                                     static_cast<std::size_t>(nz);
      amrBox.cellValues.resize(valueCount);

      const auto data = mf.const_array(mfi);

      for (int k = 0; k < nz; ++k) {
        const int globalK = lo[2] + k;
        for (int j = 0; j < ny; ++j) {
          const int globalJ = lo[1] + j;
          for (int i = 0; i < nx; ++i) {
            const int globalI = lo[0] + i;
            float value = static_cast<float>(data(globalI, globalJ, globalK));
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

  std::array<float, 3> localMinOriginalArray = {
      static_cast<float>(localMinOriginal[0]),
      static_cast<float>(localMinOriginal[1]),
      static_cast<float>(localMinOriginal[2])};
  std::array<float, 3> localMaxOriginalArray = {
      static_cast<float>(localMaxOriginal[0]),
      static_cast<float>(localMaxOriginal[1]),
      static_cast<float>(localMaxOriginal[2])};
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

  std::array<float, 3> localMinArray = {static_cast<float>(localMin[0]),
                                        static_cast<float>(localMin[1]),
                                        static_cast<float>(localMin[2])};
  std::array<float, 3> localMaxArray = {static_cast<float>(localMax[0]),
                                        static_cast<float>(localMax[1]),
                                        static_cast<float>(localMax[2])};
  std::array<float, 3> globalMinArray = {std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max()};
  std::array<float, 3> globalMaxArray = {-std::numeric_limits<float>::max(),
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

  Vec3 globalMin(globalMinArray[0], globalMinArray[1], globalMinArray[2]);
  Vec3 globalMax(globalMaxArray[0], globalMaxArray[1], globalMaxArray[2]);

  const bool invalidBounds = (globalMin[0] > globalMax[0]) ||
                             (globalMin[1] > globalMax[1]) ||
                             (globalMin[2] > globalMax[2]);
  if (invalidBounds) {
    throw std::runtime_error(
        "Failed to locate any volumetric data within the plotfile.");
  }

  const Vec3 extent = globalMax - globalMin;
  const float maxExtent = std::max(extent[0], std::max(extent[1], extent[2]));
  const float paddingAmount = (maxExtent > 0.0f) ? maxExtent * 0.05f : 1.0f;
  const Vec3 padding(paddingAmount);

  scene.explicitBounds.minCorner = globalMin - padding;
  scene.explicitBounds.maxCorner = globalMax + padding;
  scene.hasExplicitBounds = true;

  float originalScalarMinSend = hasLocalScalars
                                    ? localScalarMin
                                    : std::numeric_limits<float>::infinity();
  float originalScalarMaxSend = hasLocalScalars
                                    ? localScalarMax
                                    : -std::numeric_limits<float>::infinity();
  float globalOriginalScalarMin = originalScalarMinSend;
  float globalOriginalScalarMax = originalScalarMaxSend;

  MPI_Allreduce(&originalScalarMinSend,
                &globalOriginalScalarMin,
                1,
                MPI_FLOAT,
                MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(&originalScalarMaxSend,
                &globalOriginalScalarMax,
                1,
                MPI_FLOAT,
                MPI_MAX,
                MPI_COMM_WORLD);

  int localOriginalScalarCount = hasLocalScalars ? 1 : 0;
  int globalOriginalScalarCount = 0;
  MPI_Allreduce(&localOriginalScalarCount,
                &globalOriginalScalarCount,
                1,
                MPI_INT,
                MPI_SUM,
                MPI_COMM_WORLD);

  if (globalOriginalScalarCount > 0 &&
      std::isfinite(globalOriginalScalarMin) &&
      std::isfinite(globalOriginalScalarMax)) {
    if (globalOriginalScalarMin == globalOriginalScalarMax) {
      globalOriginalScalarMax = globalOriginalScalarMin + 1.0f;
    }
    scene.originalScalarRange = {globalOriginalScalarMin,
                                 globalOriginalScalarMax};
    scene.hasOriginalScalarRange = true;
  }

  if (logScaleInput) {
    float positiveMinSend = hasLocalPositive
                                ? localPositiveMin
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
  float scalarMaxSend = hasLocalScalars
                            ? localScalarMax
                            : -std::numeric_limits<float>::infinity();
  float globalScalarMin = scalarMinSend;
  float globalScalarMax = scalarMaxSend;

  MPI_Allreduce(
      &scalarMinSend, &globalScalarMin, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(
      &scalarMaxSend, &globalScalarMax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

  int localHasScalars = hasLocalScalars ? 1 : 0;
  int globalHasScalars = 0;
  MPI_Allreduce(
      &localHasScalars, &globalHasScalars, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (globalHasScalars <= 0 || !std::isfinite(globalScalarMin) ||
      !std::isfinite(globalScalarMax)) {
    throw std::runtime_error(
        "Failed to compute a valid scalar range from the plotfile.");
  }
  if (globalScalarMin == globalScalarMax) {
    globalScalarMax = globalScalarMin + 1.0f;
  }

  scene.processedScalarRange = {globalScalarMin, globalScalarMax};
  scene.hasProcessedScalarRange = true;
  scene.scalarRange = scene.processedScalarRange;
  scene.hasScalarRange = true;

  if (normalizeToDataRange) {
    const float rangeWidth = globalScalarMax - globalScalarMin;
    if (rangeWidth > 0.0f && std::isfinite(rangeWidth)) {
      for (auto& box : scene.localBoxes) {
        for (float& value : box.cellValues) {
          const float normalized = (value - globalScalarMin) / rangeWidth;
          value = std::clamp(normalized, 0.0f, 1.0f);
        }
      }
      scene.scalarRange = {0.0f, 1.0f};
    }
  }

  scene.localBoxes.shrink_to_fit();

  if (rank == 0) {
    const int includedLevels = maxLevel - minLevel + 1;
    std::cout << "Loaded plotfile '" << plotfilePath << "' with variable '"
              << componentName << "' across " << includedLevels << " level(s)";
    if (minLevel > 0 || maxLevel < finestLevel) {
      std::cout << " (levels " << minLevel << "-" << maxLevel << ")";
    }
    if (normalizeToDataRange) {
      std::cout << "; normalized scalar range [0, 1]";
    } else {
      std::cout << "; scalar range [" << globalScalarMin << ", " << globalScalarMax
                << "]";
    }
    if (logScaleInput) {
      std::cout << " (log scaled)";
    }
    std::cout << std::endl;
  }

  return scene;
}

ViskoresVolumeRenderer::VolumeBounds
ViskoresVolumeRenderer::computeGlobalBounds(
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

  std::array<float, 3> localMinArray = {static_cast<float>(localMin[0]),
                                        static_cast<float>(localMin[1]),
                                        static_cast<float>(localMin[2])};
  std::array<float, 3> localMaxArray = {static_cast<float>(localMax[0]),
                                        static_cast<float>(localMax[1]),
                                        static_cast<float>(localMax[2])};
  std::array<float, 3> globalMinArray = {std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max()};
  std::array<float, 3> globalMaxArray = {-std::numeric_limits<float>::max(),
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
  bounds.minCorner =
      Vec3(globalMinArray[0], globalMinArray[1], globalMinArray[2]);
  bounds.maxCorner =
      Vec3(globalMaxArray[0], globalMaxArray[1], globalMaxArray[2]);

  const bool invalidBounds = bounds.minCorner[0] > bounds.maxCorner[0] ||
                             bounds.minCorner[1] > bounds.maxCorner[1] ||
                             bounds.minCorner[2] > bounds.maxCorner[2];

  if (invalidBounds) {
    bounds.minCorner = Vec3(-1.0f);
    bounds.maxCorner = Vec3(1.0f);
    return bounds;
  }

  const Vec3 extent = bounds.maxCorner - bounds.minCorner;
  const float maxExtent =
      std::max(extent[0], std::max(extent[1], extent[2]));
  const float padding = (maxExtent > 0.0f) ? maxExtent * 0.05f : 0.5f;
  const Vec3 paddingVec(padding);

  bounds.minCorner -= paddingVec;
  bounds.maxCorner += paddingVec;
  return bounds;
}

ViskoresVolumeRenderer::VolumeBounds ViskoresVolumeRenderer::computeTightBounds(
    const std::vector<AmrBox>& boxes, const VolumeBounds& fallback) const {
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

  std::array<float, 3> localMinArray = {static_cast<float>(localMin[0]),
                                        static_cast<float>(localMin[1]),
                                        static_cast<float>(localMin[2])};
  std::array<float, 3> localMaxArray = {static_cast<float>(localMax[0]),
                                        static_cast<float>(localMax[1]),
                                        static_cast<float>(localMax[2])};
  std::array<float, 3> globalMinArray = {std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max(),
                                         std::numeric_limits<float>::max()};
  std::array<float, 3> globalMaxArray = {-std::numeric_limits<float>::max(),
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
  MPI_Allreduce(
      &localHasAny, &globalHasAny, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (globalHasAny <= 0) {
    return fallback;
  }

  VolumeBounds tight;
  tight.minCorner =
      Vec3(globalMinArray[0], globalMinArray[1], globalMinArray[2]);
  tight.maxCorner =
      Vec3(globalMaxArray[0], globalMaxArray[1], globalMaxArray[2]);
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

auto ViskoresVolumeRenderer::computeScalarHistogram(
    const std::string& plotfilePath,
    const std::string& variableName,
    int requestedMinLevel,
    int requestedMaxLevel,
    bool logScaleInput,
    int binCount) const -> ScalarHistogram {
  if (binCount <= 0) {
    throw std::invalid_argument("binCount must be positive");
  }

  SceneGeometry geometry = loadPlotFileGeometry(plotfilePath,
                                                variableName,
                                                requestedMinLevel,
                                                requestedMaxLevel,
                                                logScaleInput);

  ScalarHistogram histogram;
  histogram.binCounts.resize(static_cast<std::size_t>(binCount), 0);

  if (geometry.hasScalarRange) {
    histogram.normalizedRange = geometry.scalarRange;
  }

  if (geometry.hasProcessedScalarRange) {
    histogram.processedRange = geometry.processedScalarRange;
    histogram.hasProcessedRange = true;
  }

  if (geometry.hasOriginalScalarRange) {
    histogram.originalRange = geometry.originalScalarRange;
    histogram.hasOriginalRange = true;
  }

  const float rangeMin = histogram.normalizedRange.first;
  const float rangeMax = histogram.normalizedRange.second;
  const float rangeWidth = rangeMax - rangeMin;

  std::vector<std::uint64_t> localCounts(
      static_cast<std::size_t>(binCount), 0);
  std::uint64_t localSamples = 0;

  if (rangeWidth > 0.0f && std::isfinite(rangeWidth)) {
    const float inverseWidth = 1.0f / rangeWidth;
    for (const auto& box : geometry.localBoxes) {
      for (float value : box.cellValues) {
        if (!std::isfinite(value)) {
          continue;
        }
        float clamped = value;
        if (clamped < rangeMin) {
          clamped = rangeMin;
        } else if (clamped > rangeMax) {
          clamped = rangeMax;
        }
        float normalized = (clamped - rangeMin) * inverseWidth;
        normalized = std::clamp(normalized, 0.0f, 1.0f);
        int index =
            static_cast<int>(normalized * static_cast<float>(binCount));
        if (index >= binCount) {
          index = binCount - 1;
        } else if (index < 0) {
          index = 0;
        }
        localCounts[static_cast<std::size_t>(index)] += 1;
        localSamples += 1;
      }
    }
  }

  std::vector<std::uint64_t> globalCounts(
      static_cast<std::size_t>(binCount), 0);
  MPI_Allreduce(localCounts.data(),
                globalCounts.data(),
                binCount,
                MPI_UINT64_T,
                MPI_SUM,
                MPI_COMM_WORLD);

  std::uint64_t globalSamples = 0;
  MPI_Allreduce(
      &localSamples, &globalSamples, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);

  histogram.binCounts = std::move(globalCounts);
  histogram.sampleCount = globalSamples;
  if (!histogram.hasProcessedRange || globalSamples == 0) {
    histogram.binCounts.assign(histogram.binCounts.size(), 0);
  }

  return histogram;
}

void ViskoresVolumeRenderer::paint(const AmrBox& box,
                                   const VolumeBounds& bounds,
                                   const std::pair<float, float>& scalarRange,
                                   float boxTransparency,
                                   int antialiasing,
                                   float referenceSampleDistance,
                                   ImageFull& image,
                                   const CameraParameters& camera,
                                   const ColorMap* colorMap) {
  static VolumePainterViskores painter;
  painter.paint(box,
                bounds,
                scalarRange,
                rank,
                numProcs,
                boxTransparency,
                antialiasing,
                referenceSampleDistance,
                image,
                camera,
                colorMap);
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
    const SceneGeometry& geometry,
    const std::optional<ColorMap>& colorMap) {
  validateRenderParameters(parameters);
  initialize();

  VolumeBounds bounds = computeGlobalBounds(
      geometry.localBoxes, geometry.hasExplicitBounds, geometry.explicitBounds);
  const std::pair<float, float> scalarRange =
      geometry.hasScalarRange ? geometry.scalarRange
                              : computeGlobalScalarRange(geometry.localBoxes);
  const ColorMap* colorMapPtr = colorMap ? &(*colorMap) : nullptr;

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
  float boundingRadius = static_cast<float>(halfExtent.vectorLength());
  if (boundingRadius <= 0.0f) {
    boundingRadius = 1.0f;
  }

  const float fovY = kPi * 0.25f;
  const float maxAltitude = kPi * 0.25f;

  const float halfFov = fovY * 0.5f;
  const float minDistance =
      (halfFov > 0.0f)
          ? boundingRadius / static_cast<float>(std::tan(halfFov))
          : boundingRadius;
  const float safetyMargin = std::max(0.25f * boundingRadius, 0.5f);
  const float cameraDistance = minDistance + safetyMargin;

  std::mt19937 cameraRng(parameters.cameraSeed);
  std::uniform_real_distribution<float> azimuthDistribution(0.0f, kTwoPi);
  std::uniform_real_distribution<float> altitudeDistribution(-maxAltitude,
                                                             maxAltitude);
  const float azimuth = azimuthDistribution(cameraRng);
  const float altitude = altitudeDistribution(cameraRng);
  const float cosAltitude = std::cos(altitude);

  const Vec3 eye(
      center[0] + cameraDistance * cosAltitude * std::sin(azimuth),
      center[1] + cameraDistance * std::sin(altitude),
      center[2] + cameraDistance * cosAltitude * std::cos(azimuth));

  Vec3 upVector =
      parameters.useCustomUp ? parameters.cameraUp : Vec3(0.0f, 1.0f, 0.0f);
  const Vec3 viewDir = safeNormalize(center - eye);
  if (viewDir.crossProduct(upVector).vectorLength() <= 1e-4f) {
    upVector = Vec3(0.0f, 0.0f, 1.0f);
    if (viewDir.crossProduct(upVector).vectorLength() <= 1e-4f) {
      upVector = Vec3(1.0f, 0.0f, 0.0f);
    }
  }
  upVector = safeNormalize(upVector);

  const float nearPlane = 0.1f;
  const float farPlane = cameraDistance * 4.0f;
  CameraParameters camera{eye,
                          center,
                          upVector,
                          fovY * 180.0f / kPi,
                          nearPlane,
                          farPlane};

  if (parameters.printCamera && rank == 0) {
    std::cout << "Camera parameters (automatic):\n"
              << "  eye      = (" << camera.eye[0] << ", " << camera.eye[1]
              << ", " << camera.eye[2] << ")\n"
              << "  look_at  = (" << camera.lookAt[0] << ", " << camera.lookAt[1]
              << ", " << camera.lookAt[2] << ")\n"
              << "  up       = (" << camera.up[0] << ", " << camera.up[1] << ", "
              << camera.up[2] << ")\n"
              << "  fov_y    = " << camera.fovYDegrees << " degrees\n"
              << "  near     = " << camera.nearPlane << "\n"
              << "  far      = " << camera.farPlane << std::endl;
  }

  return renderSingleTrial(outputFilenameBase,
                           parameters,
                           geometry,
                           bounds,
                           scalarRange,
                           compositor,
                           groupGuard.group,
                           camera,
                           colorMapPtr);
}

int ViskoresVolumeRenderer::renderScene(
    const std::string& outputFilenameBase,
    const RenderParameters& parameters,
    const SceneGeometry& geometry,
    const CameraParameters& camera,
    const std::optional<ColorMap>& colorMap) {
  validateRenderParameters(parameters);
  initialize();

  if (parameters.printCamera && rank == 0) {
    std::cout << "Camera parameters (explicit):\n"
              << "  eye      = (" << camera.eye[0] << ", " << camera.eye[1]
              << ", " << camera.eye[2] << ")\n"
              << "  look_at  = (" << camera.lookAt[0] << ", " << camera.lookAt[1]
              << ", " << camera.lookAt[2] << ")\n"
              << "  up       = (" << camera.up[0] << ", " << camera.up[1] << ", "
              << camera.up[2] << ")\n"
              << "  fov_y    = " << camera.fovYDegrees << " degrees\n"
              << "  near     = " << camera.nearPlane << "\n"
              << "  far      = " << camera.farPlane << std::endl;
  }

  VolumeBounds bounds = computeGlobalBounds(
      geometry.localBoxes, geometry.hasExplicitBounds, geometry.explicitBounds);
  const std::pair<float, float> scalarRange =
      geometry.hasScalarRange ? geometry.scalarRange
                              : computeGlobalScalarRange(geometry.localBoxes);
  const ColorMap* colorMapPtr = colorMap ? &(*colorMap) : nullptr;

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

  return renderSingleTrial(outputFilenameBase,
                           parameters,
                           geometry,
                           bounds,
                           scalarRange,
                           compositor,
                           groupGuard.group,
                           camera,
                           colorMapPtr);
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
    const ColorMap* colorMap) {
  const float aspect = static_cast<float>(parameters.width) /
                       static_cast<float>(parameters.height);
  const int sqrtAntialiasing =
      static_cast<int>(std::lround(std::sqrt(parameters.antialiasing)));
  const int renderWidth = parameters.width * std::max(sqrtAntialiasing, 1);
  const int renderHeight = parameters.height * std::max(sqrtAntialiasing, 1);

  const auto reportStageTime = [&](const char* label, double localSeconds) {
    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds,
               &maxSeconds,
               1,
               MPI_DOUBLE,
               MPI_MAX,
               0,
               MPI_COMM_WORLD);
    if (rank == 0) {
      std::ostringstream stream;
      stream << std::fixed << std::setprecision(4);
      stream << "Render: " << label << " took " << maxSeconds << " s";
      std::cout << stream.str() << std::endl;
    }
  };

  float localCoarsestMinSpacing = 0.0f;
  for (const AmrBox& box : geometry.localBoxes) {
    const Vec3 span = box.maxCorner - box.minCorner;
    Vec3 spacing(0.0);
    if (box.cellDimensions[0] > 0) {
      spacing[0] = span[0] / static_cast<float>(box.cellDimensions[0]);
    }
    if (box.cellDimensions[1] > 0) {
      spacing[1] = span[1] / static_cast<float>(box.cellDimensions[1]);
    }
    if (box.cellDimensions[2] > 0) {
      spacing[2] = span[2] / static_cast<float>(box.cellDimensions[2]);
    }

    float minSpacing = std::numeric_limits<float>::max();
    for (int component = 0; component < 3; ++component) {
      if (spacing[component] > 0.0f && spacing[component] < minSpacing &&
          std::isfinite(spacing[component])) {
        minSpacing = spacing[component];
      }
    }

    if (minSpacing > 0.0f && std::isfinite(minSpacing)) {
      localCoarsestMinSpacing = std::max(localCoarsestMinSpacing, minSpacing);
    }
  }

  float globalCoarsestMinSpacing = 0.0f;
  MPI_Allreduce(&localCoarsestMinSpacing,
                &globalCoarsestMinSpacing,
                1,
                MPI_FLOAT,
                MPI_MAX,
                MPI_COMM_WORLD);

  if (!(globalCoarsestMinSpacing > 0.0f &&
        std::isfinite(globalCoarsestMinSpacing))) {
    const Vec3 fallbackSpan = bounds.maxCorner - bounds.minCorner;
    float fallbackMin = std::numeric_limits<float>::max();
    for (int component = 0; component < 3; ++component) {
      const float axisLength = fallbackSpan[component];
      if (axisLength > 0.0f && std::isfinite(axisLength)) {
        fallbackMin = std::min(fallbackMin, axisLength);
      }
    }
    if (!(fallbackMin > 0.0f && std::isfinite(fallbackMin))) {
      fallbackMin = 1.0f;
    }
    globalCoarsestMinSpacing = std::max(1e-4f, fallbackMin * 0.01f);
  }

  const float referenceSampleDistance =
      std::max(globalCoarsestMinSpacing * 0.5f, 1e-5f);

  const VolumeBounds tightBounds =
      computeTightBounds(geometry.localBoxes, bounds);

  std::vector<std::unique_ptr<ImageRGBAFloatColorDepthSort>> localLayers;
  localLayers.reserve(geometry.localBoxes.size() + 1);
  std::vector<float> depthHints;
  depthHints.reserve(geometry.localBoxes.size() + 1);

  const auto renderStart = std::chrono::steady_clock::now();
  for (std::size_t boxIndex = 0; boxIndex < geometry.localBoxes.size();
       ++boxIndex) {
    const AmrBox& box = geometry.localBoxes[boxIndex];

    auto layerImage = std::make_unique<ImageRGBAFloatColorDepthSort>(
        renderWidth, renderHeight);
    paint(box,
          bounds,
          scalarRange,
          parameters.boxTransparency,
          parameters.antialiasing,
          referenceSampleDistance,
          *layerImage,
          camera,
          colorMap);
    depthHints.push_back(computeBoxDepthHint(box, camera));
    localLayers.push_back(std::move(layerImage));
  }
  const auto renderEnd = std::chrono::steady_clock::now();
  const double renderSeconds =
      std::chrono::duration<double>(renderEnd - renderStart).count();
  reportStageTime("per-box rendering", renderSeconds);

  auto prototype =
      std::make_unique<ImageRGBAFloatColorDepthSort>(renderWidth, renderHeight);

  LayeredVolumeImage layeredImage(renderWidth,
                                  renderHeight,
                                  std::move(localLayers),
                                  std::move(depthHints),
                                  std::move(prototype));

  const auto visibilityStart = std::chrono::steady_clock::now();
  MPI_Group orderedGroup =
      buildVisibilityOrderedGroup(camera,
                                  aspect,
                                  baseGroup,
                                  parameters.useVisibilityGraph,
                                  parameters.writeVisibilityGraph,
                                  geometry.localBoxes);
  const auto visibilityEnd = std::chrono::steady_clock::now();
  const double visibilitySeconds =
      std::chrono::duration<double>(visibilityEnd - visibilityStart).count();
  reportStageTime("visibility graph computation", visibilitySeconds);

  const auto compositeStart = std::chrono::steady_clock::now();
  std::unique_ptr<Image> compositedImage =
      compositor->compose(&layeredImage, orderedGroup, MPI_COMM_WORLD);
  const auto compositeEnd = std::chrono::steady_clock::now();
  const double compositeSeconds =
      std::chrono::duration<double>(compositeEnd - compositeStart).count();
  reportStageTime("compositing", compositeSeconds);

  MPI_Group_free(&orderedGroup);

  if (compositedImage) {
    std::unique_ptr<ImageFull> fullImage;
    if (auto asFull = dynamic_cast<ImageFull*>(compositedImage.get())) {
      fullImage.reset(asFull);
      compositedImage.release();
    } else if (auto asSparse =
                   dynamic_cast<ImageSparse*>(compositedImage.get())) {
      fullImage = asSparse->uncompress();
    } else if (auto asColorDepth =
                   dynamic_cast<ImageRGBAFloatColorDepthSort*>(
                       compositedImage.get())) {
      fullImage.reset(static_cast<ImageFull*>(asColorDepth));
      compositedImage.release();
    } else {
      if (rank == 0) {
        std::cerr << "Unsupported image type returned by compositor.";
#if defined(__cpp_rtti)
        if (compositedImage) {
          const Image& imageRef = *compositedImage;
          std::cerr << " type=" << typeid(imageRef).name();
          if (dynamic_cast<ImageRGBAFloatColorDepthSort*>(compositedImage.get())) {
            std::cerr << " (matches ImageRGBAFloatColorDepthSort)";
          }
          if (dynamic_cast<ImageFull*>(compositedImage.get())) {
            std::cerr << " (matches ImageFull)";
          }
          if (dynamic_cast<ImageSparse*>(compositedImage.get())) {
            std::cerr << " (matches ImageSparse)";
          }
        }
#endif
        std::cerr << std::endl;
      }
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::unique_ptr<ImageFull> gatheredImage =
        fullImage->Gather(0, MPI_COMM_WORLD);

    if (rank == 0 && gatheredImage) {
      const int pixels = gatheredImage->getNumberOfPixels();
      std::cout << "Render: composed " << pixels << " pixels on rank 0"
                << std::endl;

      std::unique_ptr<ImageFull> outputImage;
      if (sqrtAntialiasing > 1) {
        outputImage = downsampleImage(*gatheredImage,
                                      parameters.width,
                                      parameters.height,
                                      sqrtAntialiasing);
      } else {
        outputImage = std::move(gatheredImage);
      }

      if (auto* asColorDepth =
              dynamic_cast<ImageRGBAFloatColorDepthSort*>(outputImage.get())) {
        renderBoundingBoxLayer(tightBounds, camera, 1, *asColorDepth);
      }

      const std::string extension = lowercaseExtension(outputFilename);
      if (!extension.empty() && extension != ".ppm" && extension != ".png") {
        std::cerr << "Render: unrecognized image extension '" << extension
                  << "', defaulting to PPM output." << std::endl;
      }

      bool saved = false;
      if (extension == ".png") {
        saved = SavePNG(*outputImage, outputFilename);
      } else {
        saved = SavePPM(*outputImage, outputFilename);
      }
      if (saved) {
        std::cout << "Saved volume composited image to '" << outputFilename
                  << "'" << std::endl;
      } else {
        std::cerr << "Failed to save composited image to '" << outputFilename
                  << "'" << std::endl;
      }
    }
  }

  return 0;
}

int ViskoresVolumeRenderer::run(const RunOptions& providedOptions) {
  RunOptions options = providedOptions;

  validateRenderParameters(options.parameters);

  if (options.outputFilename.empty()) {
    throw std::invalid_argument("output filename must not be empty");
  }
  if (options.plotfilePath.empty()) {
    throw std::runtime_error("plotfile path is required");
  }
  if (options.minLevel < 0) {
    throw std::invalid_argument("min level must be non-negative");
  }
  if (options.maxLevel < -1) {
    throw std::invalid_argument(
        "max level must be non-negative or -1 for all levels");
  }
  if (options.maxLevel >= 0 && options.minLevel > options.maxLevel) {
    throw std::runtime_error("min level must not exceed max level");
  }

  if (options.parameters.useCustomUp) {
    const float length =
        static_cast<float>(options.parameters.cameraUp.vectorLength());
    if (!(length > 0.0f) || !std::isfinite(length)) {
      throw std::invalid_argument(
          "custom up vector must be non-zero and finite");
    }
  }

  if (options.scalarRange) {
    const float rangeMin = options.scalarRange->first;
    const float rangeMax = options.scalarRange->second;
    if (!std::isfinite(rangeMin) || !std::isfinite(rangeMax) ||
        !(rangeMin < rangeMax)) {
      throw std::invalid_argument(
          "scalar range must contain two finite values with min < max");
    }
  }

  if (options.colorMap) {
    const auto& colorMap = *options.colorMap;
    if (colorMap.size() < 2) {
      throw std::invalid_argument(
          "color map must provide at least two control points");
    }
    float previousValue = -std::numeric_limits<float>::infinity();
    for (std::size_t index = 0; index < colorMap.size(); ++index) {
      const auto& point = colorMap[index];
      if (!std::isfinite(point.value)) {
        throw std::invalid_argument(
            "color map control point values must be finite");
      }
      if (point.value <= previousValue) {
        throw std::invalid_argument(
            "color map control point values must be strictly increasing");
      }
      previousValue = point.value;

      const auto validateComponent = [&](float component,
                                         const char* name) -> void {
        if (!std::isfinite(component) || component < 0.0f ||
            component > 1.0f) {
          throw std::invalid_argument(
              std::string("color map ") + name +
              " components must be finite and within [0, 1]");
        }
      };

      validateComponent(point.red, "red");
      validateComponent(point.green, "green");
      validateComponent(point.blue, "blue");
      validateComponent(point.alpha, "alpha");
    }
  }

  if (options.camera) {
    const CameraParameters& camera = *options.camera;
    const auto isFiniteVec3 = [](const Vec3& vector) -> bool {
      return std::isfinite(vector[0]) && std::isfinite(vector[1]) &&
             std::isfinite(vector[2]);
    };

    Vec3 eye(camera.eye);
    Vec3 lookAt(camera.lookAt);
    Vec3 up(camera.up);
    if (!isFiniteVec3(eye) || !isFiniteVec3(lookAt) || !isFiniteVec3(up)) {
      throw std::invalid_argument(
          "camera vectors must have finite components");
    }

    const Vec3 forward = lookAt - eye;
    const float forwardLength = static_cast<float>(forward.vectorLength());
    if (!(forwardLength > 0.0f) || !std::isfinite(forwardLength)) {
      throw std::invalid_argument("camera eye and look-at must be distinct");
    }

    const float upLength = static_cast<float>(up.vectorLength());
    if (!(upLength > 0.0f) || !std::isfinite(upLength)) {
      throw std::invalid_argument("camera up vector must be non-zero");
    }
    const float crossMagnitude =
        static_cast<float>(forward.crossProduct(up).vectorLength());
    if (!(crossMagnitude > 1e-6f)) {
      throw std::invalid_argument(
          "camera up vector must not be parallel to the view direction");
    }

    if (!std::isfinite(camera.fovYDegrees) || !(camera.fovYDegrees > 0.0f) ||
        !(camera.fovYDegrees < 180.0f)) {
      throw std::invalid_argument("camera fov must be in (0, 180) degrees");
    }
    if (!std::isfinite(camera.nearPlane) || !(camera.nearPlane > 0.0f)) {
      throw std::invalid_argument("camera near plane must be > 0");
    }
    if (!std::isfinite(camera.farPlane) ||
        !(camera.farPlane > camera.nearPlane)) {
      throw std::invalid_argument(
          "camera far plane must exceed the near plane");
    }
  }

  if (!amrex::FileSystem::Exists(options.plotfilePath)) {
    throw std::runtime_error("plotfile path '" + options.plotfilePath +
                             "' does not exist");
  }

  const bool hasScalarOverride = options.scalarRange.has_value();

  SceneGeometry geometry =
      loadPlotFileGeometry(options.plotfilePath,
                           options.variableName,
                           options.minLevel,
                           options.maxLevel,
                           options.logScaleInput,
                           /*normalizeToDataRange=*/!hasScalarOverride);
  if (!geometry.hasProcessedScalarRange) {
    throw std::runtime_error(
        "Internal error: processed scalar range unavailable for color mapping.");
  }
  const float processedMin = geometry.processedScalarRange.first;
  const float processedMax = geometry.processedScalarRange.second;
  const float processedSpan = processedMax - processedMin;
  if (!(processedSpan > 0.0f) || !std::isfinite(processedSpan)) {
    throw std::runtime_error(
        "Failed to establish a finite scalar range for color mapping.");
  }

  const auto toProcessed = [&](float physicalValue) -> float {
    if (!std::isfinite(physicalValue)) {
      throw std::invalid_argument(
          "color_map scalar values must be finite.");
    }
    if (options.logScaleInput) {
      if (!(physicalValue > 0.0f)) {
        throw std::invalid_argument(
            "color_map scalar values must be positive when log scaling is "
            "enabled.");
      }
      return std::log(physicalValue);
    }
    return physicalValue;
  };

  float normalizationMin = processedMin;
  float normalizationMax = processedMax;
  float processedMinOverride = processedMin;
  float processedMaxOverride = processedMax;
  if (hasScalarOverride) {
    processedMinOverride = toProcessed(options.scalarRange->first);
    processedMaxOverride = toProcessed(options.scalarRange->second);
    if (!(processedMinOverride < processedMaxOverride)) {
      throw std::invalid_argument(
          "scalar_range must contain two values with min < max.");
    }
    normalizationMin = processedMinOverride;
    normalizationMax = processedMaxOverride;
  }

  const float normalizationSpan = normalizationMax - normalizationMin;
  if (!(normalizationSpan > 0.0f) || !std::isfinite(normalizationSpan)) {
    throw std::runtime_error(
        "Failed to establish a finite scalar range for color mapping.");
  }

  const auto toNormalized = [&](float processedValue) -> float {
    return (processedValue - normalizationMin) / normalizationSpan;
  };

  const auto clampNormalized = [](float value) -> float {
    if (!std::isfinite(value)) {
      throw std::invalid_argument(
          "color_map produced a non-finite normalized scalar value.");
    }
    return std::clamp(value, 0.0f, 1.0f);
  };

  if (hasScalarOverride) {
    const float inverseSpan = 1.0f / normalizationSpan;
    for (auto& box : geometry.localBoxes) {
      for (float& value : box.cellValues) {
        const float normalized = (value - normalizationMin) * inverseSpan;
        value = std::clamp(normalized, 0.0f, 1.0f);
      }
    }
    geometry.scalarRange = {0.0f, 1.0f};
    geometry.hasScalarRange = true;
  }

  if (options.scalarRange) {
    float normalizedMin = clampNormalized(toNormalized(processedMinOverride));
    float normalizedMax = clampNormalized(toNormalized(processedMaxOverride));
    if (!(normalizedMin < normalizedMax)) {
      throw std::invalid_argument(
          "scalar_range must have min < max after applying log scaling.");
    }
    geometry.scalarRange = {normalizedMin, normalizedMax};
    geometry.hasScalarRange = true;
  }

  std::optional<ColorMap> normalizedColorMap;
  if (options.colorMap) {
    ColorMap converted;
    converted.reserve(options.colorMap->size());
    for (const auto& controlPoint : *options.colorMap) {
      ColorMapControlPoint normalizedPoint = controlPoint;
      const float processedValue = toProcessed(controlPoint.value);
      normalizedPoint.value =
          clampNormalized(toNormalized(processedValue));
      converted.push_back(normalizedPoint);
    }
    normalizedColorMap = std::move(converted);
  }
  const std::optional<ColorMap> emptyColorMap;
  const std::optional<ColorMap>* colorMapPtr =
      normalizedColorMap ? &normalizedColorMap : &emptyColorMap;

  if (options.camera) {
    Vec3 normalizedUp(options.camera->up);
    normalizedUp = safeNormalize(normalizedUp);
    options.camera->up = normalizedUp;
    return renderScene(options.outputFilename,
                       options.parameters,
                       geometry,
                       *options.camera,
                       *colorMapPtr);
  }

  return renderScene(options.outputFilename,
                     options.parameters,
                     geometry,
                     *colorMapPtr);
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

  return run(options);
}
