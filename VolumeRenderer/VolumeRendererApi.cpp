// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#include <VolumeRenderer/VolumeRendererApi.hpp>

#include <AMReX_MultiFabUtil.H>
#include <Common/CameraUtils.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace amrVolumeRenderer {
namespace api {
namespace {

using Vec3 = amrex::RealVect;

Vec3 componentMin(const Vec3& a, const Vec3& b) {
  return Vec3(std::min(a[0], b[0]), std::min(a[1], b[1]),
              std::min(a[2], b[2]));
}

Vec3 componentMax(const Vec3& a, const Vec3& b) {
  return Vec3(std::max(a[0], b[0]), std::max(a[1], b[1]),
              std::max(a[2], b[2]));
}

void validateLevelData(const AmrData& data) {
  if (data.levelData.empty()) {
    throw std::invalid_argument("levelData must include at least one level");
  }
  if (data.levelGeometry.size() != data.levelData.size()) {
    throw std::invalid_argument(
        "levelGeometry must match the size of levelData");
  }
}

void validateComponent(const amrex::MultiFab& mf, int component) {
  if (component < 0 || component >= mf.nComp()) {
    throw std::invalid_argument("component index is out of bounds");
  }
}

VolumeRenderer::SceneGeometry loadMultiFabGeometry(const AmrData& data,
                                                   int requestedMinLevel,
                                                   int requestedMaxLevel,
                                                   int component,
                                                   bool logScaleInput,
                                                   bool normalizeToDataRange) {
  if (AMREX_SPACEDIM != 3) {
    throw std::runtime_error(
        "The volume renderer currently expects 3D data (AMREX_SPACEDIM=3)."
    );
  }

  validateLevelData(data);

  const int finestLevel = static_cast<int>(data.levelData.size()) - 1;
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
    throw std::runtime_error("minLevel must not exceed maxLevel");
  }

  if (maxLevel > 0 &&
      data.refinementRatios.size() < static_cast<std::size_t>(maxLevel)) {
    throw std::invalid_argument(
        "refinementRatios must provide ratios for each level transition");
  }

  amrex::Vector<amrex::MultiFab const*> levelPtrs;
  levelPtrs.reserve(static_cast<std::size_t>(maxLevel) + 1);
  for (int level = 0; level <= maxLevel; ++level) {
    if (data.levelData[level] == nullptr) {
      throw std::invalid_argument("levelData contains a null MultiFab pointer");
    }
    validateComponent(*data.levelData[level], component);
    levelPtrs.push_back(data.levelData[level]);
  }

  amrex::Vector<amrex::IntVect> refinementRatios;
  refinementRatios.reserve((maxLevel > 0) ? static_cast<std::size_t>(maxLevel)
                                          : 0);
  for (int level = 0; level < maxLevel; ++level) {
    refinementRatios.push_back(data.refinementRatios[level]);
  }

  amrex::Vector<amrex::MultiFab> convexified =
      amrex::convexify(levelPtrs, refinementRatios);

  VolumeRenderer::SceneGeometry scene;
  scene.localBoxes.reserve(64);

  Vec3 localMinOriginal(std::numeric_limits<float>::max());
  Vec3 localMaxOriginal(-std::numeric_limits<float>::max());
  bool hasLocalBoxes = false;

  float localScalarMin = std::numeric_limits<float>::infinity();
  float localScalarMax = -std::numeric_limits<float>::infinity();
  bool hasLocalScalars = false;
  float localPositiveMin = std::numeric_limits<float>::infinity();
  bool hasLocalPositive = false;

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

    const amrex::Geometry& geom = data.levelGeometry[level];
    const auto probLo = geom.ProbLoArray();
    const auto cellSize = geom.CellSizeArray();

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

      VolumeRenderer::AmrBox amrBox;
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

      const auto dataView = mf.const_array(mfi);

      for (int k = 0; k < nz; ++k) {
        const int globalK = lo[2] + k;
        for (int j = 0; j < ny; ++j) {
          const int globalJ = lo[1] + j;
          for (int i = 0; i < nx; ++i) {
            const int globalI = lo[0] + i;
            float value = static_cast<float>(
                dataView(globalI, globalJ, globalK, component));
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
  for (int componentIndex = 0; componentIndex < 3; ++componentIndex) {
    const float axisLength = std::fabs(globalExtentOriginal[componentIndex]);
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
        "Failed to locate any volumetric data within the AMReX inputs.");
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
        "Failed to compute a valid scalar range from the AMReX inputs.");
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

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    const int includedLevels = maxLevel - minLevel + 1;
    std::cout << "Loaded AMReX data component " << component << " across "
              << includedLevels << " level(s)";
    if (minLevel > 0 || maxLevel < finestLevel) {
      std::cout << " (levels " << minLevel << "-" << maxLevel << ")";
    }
    if (normalizeToDataRange) {
      std::cout << "; normalized scalar range [0, 1]";
    } else {
      std::cout << "; scalar range [" << globalScalarMin << ", "
                << globalScalarMax << "]";
    }
    if (logScaleInput) {
      std::cout << " (log scaled)";
    }
    std::cout << std::endl;
  }

  return scene;
}

void validateScalarRange(const std::optional<std::pair<float, float>>& range) {
  if (!range) {
    return;
  }
  const float rangeMin = range->first;
  const float rangeMax = range->second;
  if (!std::isfinite(rangeMin) || !std::isfinite(rangeMax) ||
      !(rangeMin < rangeMax)) {
    throw std::invalid_argument(
        "scalar_range must contain two finite values with min < max");
  }
}

void validateColorMap(const std::optional<VolumeRenderer::ColorMap>& colorMap) {
  if (!colorMap) {
    return;
  }
  if (colorMap->size() < 2) {
    throw std::invalid_argument(
        "color map must provide at least two control points");
  }
  float previousValue = -std::numeric_limits<float>::infinity();
  for (const auto& point : *colorMap) {
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
      if (!std::isfinite(component) || component < 0.0f || component > 1.0f) {
        throw std::invalid_argument(std::string("color map ") + name +
                                    " components must be finite and within [0, 1]");
      }
    };

    validateComponent(point.red, "red");
    validateComponent(point.green, "green");
    validateComponent(point.blue, "blue");
    validateComponent(point.alpha, "alpha");
  }
}

void validateCamera(const std::optional<VolumeRenderer::CameraParameters>& camera) {
  if (!camera) {
    return;
  }
  const auto isFiniteVec3 = [](const Vec3& vector) -> bool {
    return std::isfinite(vector[0]) && std::isfinite(vector[1]) &&
           std::isfinite(vector[2]);
  };

  Vec3 eye(camera->eye);
  Vec3 lookAt(camera->lookAt);
  Vec3 up(camera->up);
  if (!isFiniteVec3(eye) || !isFiniteVec3(lookAt) || !isFiniteVec3(up)) {
    throw std::invalid_argument("camera vectors must have finite components");
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

  if (!std::isfinite(camera->fovYDegrees) || !(camera->fovYDegrees > 0.0f) ||
      !(camera->fovYDegrees < 180.0f)) {
    throw std::invalid_argument("camera fov must be in (0, 180) degrees");
  }
  if (!std::isfinite(camera->nearPlane) || !(camera->nearPlane > 0.0f)) {
    throw std::invalid_argument("camera near plane must be > 0");
  }
  if (!std::isfinite(camera->farPlane) ||
      !(camera->farPlane > camera->nearPlane)) {
    throw std::invalid_argument(
        "camera far plane must exceed the near plane");
  }
}

void validateUpVector(const std::optional<amrex::RealVect>& upVector) {
  if (!upVector) {
    return;
  }
  const float length = static_cast<float>(upVector->vectorLength());
  if (!(length > 0.0f) || !std::isfinite(length)) {
    throw std::invalid_argument("up_vector must be non-zero and finite");
  }
}

}  // namespace

int Render(const AmrData& data, const RenderOptions& options) {
  if (options.outputFilename.empty()) {
    throw std::invalid_argument("output filename must not be empty");
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

  validateUpVector(options.upVector);
  validateScalarRange(options.scalarRange);
  validateColorMap(options.colorMap);
  validateCamera(options.camera);

  VolumeRenderer renderer;

  VolumeRenderer::RenderParameters parameters;
  parameters.width = options.width;
  parameters.height = options.height;
  parameters.boxTransparency = options.boxTransparency;
  parameters.antialiasing = options.antialiasing;
  parameters.useVisibilityGraph = options.visibilityGraph;
  parameters.writeVisibilityGraph = options.writeVisibilityGraph;
  if (options.upVector) {
    parameters.cameraUp = *options.upVector;
    parameters.useCustomUp = true;
  }

  const bool hasScalarOverride = options.scalarRange.has_value();

  VolumeRenderer::SceneGeometry geometry =
      loadMultiFabGeometry(data,
                           options.minLevel,
                           options.maxLevel,
                           options.component,
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
      throw std::invalid_argument("color_map scalar values must be finite.");
    }
    if (options.logScaleInput) {
      if (!(physicalValue > 0.0f)) {
        throw std::invalid_argument(
            "color_map scalar values must be positive when log scaling is enabled.");
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

  std::optional<VolumeRenderer::ColorMap> normalizedColorMap;
  if (options.colorMap) {
    VolumeRenderer::ColorMap converted;
    converted.reserve(options.colorMap->size());
    for (const auto& controlPoint : *options.colorMap) {
      VolumeRenderer::ColorMapControlPoint normalizedPoint = controlPoint;
      const float processedValue = toProcessed(controlPoint.value);
      normalizedPoint.value = clampNormalized(toNormalized(processedValue));
      converted.push_back(normalizedPoint);
    }
    normalizedColorMap = std::move(converted);
  }
  const std::optional<VolumeRenderer::ColorMap> emptyColorMap;
  const std::optional<VolumeRenderer::ColorMap>* colorMapPtr =
      normalizedColorMap ? &normalizedColorMap : &emptyColorMap;

  if (options.camera) {
    VolumeRenderer::CameraParameters camera = *options.camera;
    camera.up = camera::safeNormalize(camera.up);
    return renderer.renderScene(options.outputFilename,
                                parameters,
                                geometry,
                                camera,
                                *colorMapPtr);
  }

  return renderer.renderScene(options.outputFilename,
                              parameters,
                              geometry,
                              *colorMapPtr);
}

VolumeRenderer::ScalarHistogram ComputeHistogram(
    const AmrData& data, const HistogramOptions& options) {
  if (options.binCount <= 0) {
    throw std::invalid_argument("binCount must be positive");
  }

  VolumeRenderer::SceneGeometry geometry =
      loadMultiFabGeometry(data,
                           options.minLevel,
                           options.maxLevel,
                           options.component,
                           options.logScaleInput,
                           /*normalizeToDataRange=*/true);

  VolumeRenderer::ScalarHistogram histogram;
  histogram.binCounts.resize(static_cast<std::size_t>(options.binCount), 0);

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
      static_cast<std::size_t>(options.binCount), 0);
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
            static_cast<int>(normalized * static_cast<float>(options.binCount));
        if (index >= options.binCount) {
          index = options.binCount - 1;
        } else if (index < 0) {
          index = 0;
        }
        localCounts[static_cast<std::size_t>(index)] += 1;
        localSamples += 1;
      }
    }
  }

  std::vector<std::uint64_t> globalCounts(
      static_cast<std::size_t>(options.binCount), 0);
  MPI_Allreduce(localCounts.data(),
                globalCounts.data(),
                options.binCount,
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

}  // namespace api
}  // namespace amrVolumeRenderer
