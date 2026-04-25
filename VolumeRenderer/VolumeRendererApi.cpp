// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#include <VolumeRenderer/VolumeRendererApi.hpp>
#include <VolumeRenderer/SceneBuilder.hpp>

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

  auto ownedLevels = std::make_shared<amrex::Vector<amrex::MultiFab>>(
      amrex::convexify(levelPtrs, refinementRatios));

  amrex::Vector<amrVolumeRenderer::detail::LevelGridGeometry> levelGeometry;
  levelGeometry.reserve(data.levelGeometry.size());
  for (const amrex::Geometry& geom : data.levelGeometry) {
    amrVolumeRenderer::detail::LevelGridGeometry grid;
    const auto probLo = geom.ProbLoArray();
    const auto cellSize = geom.CellSizeArray();
    for (int dim = 0; dim < AMREX_SPACEDIM; ++dim) {
      grid.probLo[dim] = probLo[dim];
      grid.cellSize[dim] = cellSize[dim];
    }
    levelGeometry.push_back(grid);
  }

  VolumeRenderer::SceneGeometry scene =
      amrVolumeRenderer::detail::BuildSceneGeometry(
          ownedLevels,
          levelGeometry,
          amrVolumeRenderer::detail::SceneBuildOptions{
              minLevel,
              maxLevel,
              component,
              logScaleInput,
              normalizeToDataRange,
              "Failed to locate any volumetric data within the AMReX inputs.",
              "Failed to compute a valid scalar range from the AMReX inputs."});

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
      std::cout << "; scalar range [" << scene.processedScalarRange.first
                << ", " << scene.processedScalarRange.second << "]";
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
    amrVolumeRenderer::detail::SetSceneNormalizationRange(
        geometry, normalizationMin, normalizationMax);
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
  return amrVolumeRenderer::detail::ComputeSceneHistogram(
      geometry, options.binCount);
}

}  // namespace api
}  // namespace amrVolumeRenderer
