// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#include <VolumeRenderer/SceneBuilder.hpp>

#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_Math.H>
#include <AMReX_ParReduce.H>
#include <AMReX_ParallelDescriptor.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace amrVolumeRenderer {
namespace detail {
namespace {

using Vec3 = amrex::RealVect;

Vec3 componentMin(const Vec3& a, const Vec3& b) {
  return Vec3(std::min(a[0], b[0]),
              std::min(a[1], b[1]),
              std::min(a[2], b[2]));
}

Vec3 componentMax(const Vec3& a, const Vec3& b) {
  return Vec3(std::max(a[0], b[0]),
              std::max(a[1], b[1]),
              std::max(a[2], b[2]));
}

MPI_Datatype mpiRealType() {
  return amrex::ParallelDescriptor::Mpi_typemap<amrex::Real>::type();
}

struct LocalScalarStats {
  amrex::Real minValue = std::numeric_limits<amrex::Real>::infinity();
  amrex::Real maxValue = -std::numeric_limits<amrex::Real>::infinity();
  amrex::Real minPositive = std::numeric_limits<amrex::Real>::infinity();
  amrex::Long finiteCount = 0;
};

LocalScalarStats reduceLocalScalarStats(const amrex::MultiFab& mf,
                                        int component) {
  LocalScalarStats stats;
  if (mf.size() == 0) {
    return stats;
  }

  const auto& arrays = mf.const_arrays();
  const amrex::Real inf = std::numeric_limits<amrex::Real>::infinity();
  const auto reduced = amrex::ParReduce(
      amrex::TypeList<amrex::ReduceOpMin,
                      amrex::ReduceOpMax,
                      amrex::ReduceOpMin,
                      amrex::ReduceOpSum>{},
      amrex::TypeList<amrex::Real,
                      amrex::Real,
                      amrex::Real,
                      amrex::Long>{},
      mf,
      amrex::IntVect(0),
      [=] AMREX_GPU_DEVICE(
          int boxNo,
          int i,
          int j,
          int k) noexcept -> amrex::GpuTuple<amrex::Real,
                                             amrex::Real,
                                             amrex::Real,
                                             amrex::Long> {
        const amrex::Real raw = arrays[boxNo](i, j, k, component);
        if (!amrex::Math::isfinite(raw)) {
          return {inf, -inf, inf, 0};
        }
        const amrex::Real positive = (raw > amrex::Real(0.0)) ? raw : inf;
        return {raw, raw, positive, 1};
      });

  stats.minValue = amrex::get<0>(reduced);
  stats.maxValue = amrex::get<1>(reduced);
  stats.minPositive = amrex::get<2>(reduced);
  stats.finiteCount = amrex::get<3>(reduced);
  return stats;
}

void mergeScalarStats(LocalScalarStats& dst, const LocalScalarStats& src) {
  dst.minValue = std::min(dst.minValue, src.minValue);
  dst.maxValue = std::max(dst.maxValue, src.maxValue);
  dst.minPositive = std::min(dst.minPositive, src.minPositive);
  dst.finiteCount += src.finiteCount;
}

std::pair<float, float> makeScalarRange(amrex::Real minValue,
                                        amrex::Real maxValue) {
  if (minValue == maxValue) {
    maxValue = minValue + amrex::Real(1.0);
  }
  return {static_cast<float>(minValue), static_cast<float>(maxValue)};
}

}  // namespace

VolumeRenderer::SceneGeometry BuildSceneGeometry(
    std::shared_ptr<amrex::Vector<amrex::MultiFab>> ownedLevels,
    const amrex::Vector<LevelGridGeometry>& levelGeometry,
    const SceneBuildOptions& options) {
  if (!ownedLevels) {
    throw std::invalid_argument("ownedLevels must not be null");
  }
  if (ownedLevels->size() != levelGeometry.size()) {
    throw std::invalid_argument(
        "ownedLevels and levelGeometry must have matching sizes");
  }

  VolumeRenderer::SceneGeometry scene;
  scene.ownedLevels = std::move(ownedLevels);
  scene.localBoxes.reserve(64);

  Vec3 localMinOriginal(std::numeric_limits<amrex::Real>::max());
  Vec3 localMaxOriginal(-std::numeric_limits<amrex::Real>::max());
  bool hasLocalBoxes = false;
  LocalScalarStats localScalarStats;

  for (int level = options.minLevel; level <= options.maxLevel; ++level) {
    if (level < 0 || level >= static_cast<int>(scene.ownedLevels->size())) {
      continue;
    }

    amrex::MultiFab& mf = (*scene.ownedLevels)[level];
    if (mf.size() == 0) {
      continue;
    }

    mergeScalarStats(localScalarStats,
                     reduceLocalScalarStats(mf, options.component));

    const auto& geom = levelGeometry[static_cast<std::size_t>(level)];
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
      amrBox.validBox = box;
      amrBox.values = mf.const_array(mfi);
      amrBox.component = options.component;

      Vec3 minCorner(0.0);
      Vec3 maxCorner(0.0);
      minCorner[0] = geom.probLo[0] +
                     static_cast<amrex::Real>(lo[0]) * geom.cellSize[0];
      minCorner[1] = geom.probLo[1] +
                     static_cast<amrex::Real>(lo[1]) * geom.cellSize[1];
      minCorner[2] = geom.probLo[2] +
                     static_cast<amrex::Real>(lo[2]) * geom.cellSize[2];
      maxCorner[0] = geom.probLo[0] +
                     static_cast<amrex::Real>(hi[0] + 1) * geom.cellSize[0];
      maxCorner[1] = geom.probLo[1] +
                     static_cast<amrex::Real>(hi[1] + 1) * geom.cellSize[1];
      maxCorner[2] = geom.probLo[2] +
                     static_cast<amrex::Real>(hi[2] + 1) * geom.cellSize[2];

      amrBox.minCorner = minCorner;
      amrBox.maxCorner = maxCorner;

      localMinOriginal = componentMin(localMinOriginal, minCorner);
      localMaxOriginal = componentMax(localMaxOriginal, maxCorner);
      hasLocalBoxes = true;
      scene.localBoxes.push_back(amrBox);
    }
  }

  if (!hasLocalBoxes) {
    localMinOriginal = Vec3(std::numeric_limits<amrex::Real>::max());
    localMaxOriginal = Vec3(-std::numeric_limits<amrex::Real>::max());
  }

  std::array<amrex::Real, 3> localMinOriginalArray = {
      localMinOriginal[0], localMinOriginal[1], localMinOriginal[2]};
  std::array<amrex::Real, 3> localMaxOriginalArray = {
      localMaxOriginal[0], localMaxOriginal[1], localMaxOriginal[2]};
  std::array<amrex::Real, 3> globalMinOriginalArray = {
      std::numeric_limits<amrex::Real>::max(),
      std::numeric_limits<amrex::Real>::max(),
      std::numeric_limits<amrex::Real>::max()};
  std::array<amrex::Real, 3> globalMaxOriginalArray = {
      -std::numeric_limits<amrex::Real>::max(),
      -std::numeric_limits<amrex::Real>::max(),
      -std::numeric_limits<amrex::Real>::max()};

  MPI_Allreduce(localMinOriginalArray.data(),
                globalMinOriginalArray.data(),
                3,
                mpiRealType(),
                MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(localMaxOriginalArray.data(),
                globalMaxOriginalArray.data(),
                3,
                mpiRealType(),
                MPI_MAX,
                MPI_COMM_WORLD);

  Vec3 globalMinOriginal(globalMinOriginalArray[0],
                         globalMinOriginalArray[1],
                         globalMinOriginalArray[2]);
  Vec3 globalMaxOriginal(globalMaxOriginalArray[0],
                         globalMaxOriginalArray[1],
                         globalMaxOriginalArray[2]);

  amrex::Real minExtent = std::numeric_limits<amrex::Real>::max();
  const Vec3 globalExtentOriginal = globalMaxOriginal - globalMinOriginal;
  for (int component = 0; component < 3; ++component) {
    const amrex::Real axisLength =
        std::abs(globalExtentOriginal[component]);
    if (axisLength > amrex::Real(0.0) && std::isfinite(axisLength)) {
      minExtent = std::min(minExtent, axisLength);
    }
  }

  amrex::Real globalScale = amrex::Real(1.0);
  if (minExtent > amrex::Real(0.0) && std::isfinite(minExtent)) {
    globalScale = amrex::Real(1.0) / minExtent;
  }
  if (!std::isfinite(globalScale) || !(globalScale > amrex::Real(0.0))) {
    globalScale = amrex::Real(1.0);
  }

  if (globalScale != amrex::Real(1.0)) {
    for (auto& box : scene.localBoxes) {
      box.minCorner *= globalScale;
      box.maxCorner *= globalScale;
    }
    globalMinOriginal *= globalScale;
    globalMaxOriginal *= globalScale;
  }

  Vec3 localMin(std::numeric_limits<amrex::Real>::max());
  Vec3 localMax(-std::numeric_limits<amrex::Real>::max());
  for (const auto& box : scene.localBoxes) {
    localMin = componentMin(localMin, box.minCorner);
    localMax = componentMax(localMax, box.maxCorner);
  }

  if (!hasLocalBoxes) {
    localMin = Vec3(std::numeric_limits<amrex::Real>::max());
    localMax = Vec3(-std::numeric_limits<amrex::Real>::max());
  }

  std::array<amrex::Real, 3> localMinArray = {localMin[0],
                                               localMin[1],
                                               localMin[2]};
  std::array<amrex::Real, 3> localMaxArray = {localMax[0],
                                               localMax[1],
                                               localMax[2]};
  std::array<amrex::Real, 3> globalMinArray = {
      std::numeric_limits<amrex::Real>::max(),
      std::numeric_limits<amrex::Real>::max(),
      std::numeric_limits<amrex::Real>::max()};
  std::array<amrex::Real, 3> globalMaxArray = {
      -std::numeric_limits<amrex::Real>::max(),
      -std::numeric_limits<amrex::Real>::max(),
      -std::numeric_limits<amrex::Real>::max()};

  MPI_Allreduce(localMinArray.data(),
                globalMinArray.data(),
                3,
                mpiRealType(),
                MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(localMaxArray.data(),
                globalMaxArray.data(),
                3,
                mpiRealType(),
                MPI_MAX,
                MPI_COMM_WORLD);

  Vec3 globalMin(globalMinArray[0], globalMinArray[1], globalMinArray[2]);
  Vec3 globalMax(globalMaxArray[0], globalMaxArray[1], globalMaxArray[2]);
  if ((globalMin[0] > globalMax[0]) || (globalMin[1] > globalMax[1]) ||
      (globalMin[2] > globalMax[2])) {
    throw std::runtime_error(options.noDataError);
  }

  const Vec3 extent = globalMax - globalMin;
  const amrex::Real maxExtent =
      std::max(extent[0], std::max(extent[1], extent[2]));
  const amrex::Real paddingAmount =
      (maxExtent > amrex::Real(0.0)) ? maxExtent * amrex::Real(0.05)
                                     : amrex::Real(1.0);
  const Vec3 padding(paddingAmount);

  scene.explicitBounds.minCorner = globalMin - padding;
  scene.explicitBounds.maxCorner = globalMax + padding;
  scene.hasExplicitBounds = true;

  const amrex::Real localOriginalMinSend =
      (localScalarStats.finiteCount > 0)
          ? localScalarStats.minValue
          : std::numeric_limits<amrex::Real>::infinity();
  const amrex::Real localOriginalMaxSend =
      (localScalarStats.finiteCount > 0)
          ? localScalarStats.maxValue
          : -std::numeric_limits<amrex::Real>::infinity();
  amrex::Real globalOriginalScalarMin = localOriginalMinSend;
  amrex::Real globalOriginalScalarMax = localOriginalMaxSend;
  amrex::Long globalOriginalFiniteCount = 0;

  MPI_Allreduce(&localOriginalMinSend,
                &globalOriginalScalarMin,
                1,
                mpiRealType(),
                MPI_MIN,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localOriginalMaxSend,
                &globalOriginalScalarMax,
                1,
                mpiRealType(),
                MPI_MAX,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localScalarStats.finiteCount,
                &globalOriginalFiniteCount,
                1,
                amrex::ParallelDescriptor::Mpi_typemap<amrex::Long>::type(),
                MPI_SUM,
                MPI_COMM_WORLD);

  if (globalOriginalFiniteCount > 0 &&
      std::isfinite(globalOriginalScalarMin) &&
      std::isfinite(globalOriginalScalarMax)) {
    scene.originalScalarRange =
        makeScalarRange(globalOriginalScalarMin, globalOriginalScalarMax);
    scene.hasOriginalScalarRange = true;
  }

  amrex::Real processedMin = globalOriginalScalarMin;
  amrex::Real processedMax = globalOriginalScalarMax;

  scene.scalarTransform.logScaleInput = options.logScaleInput;
  scene.scalarTransform.normalizeToUnitRange = false;
  scene.scalarTransform.positiveFloor = amrex::Real(0.0);

  if (options.logScaleInput) {
    const amrex::Real localPositiveMinSend =
        (localScalarStats.minPositive > amrex::Real(0.0) &&
         std::isfinite(localScalarStats.minPositive))
            ? localScalarStats.minPositive
            : std::numeric_limits<amrex::Real>::infinity();
    amrex::Real globalPositiveMin = localPositiveMinSend;
    MPI_Allreduce(&localPositiveMinSend,
                  &globalPositiveMin,
                  1,
                  mpiRealType(),
                  MPI_MIN,
                  MPI_COMM_WORLD);

    const amrex::Long localPositiveCount =
        (localPositiveMinSend < std::numeric_limits<amrex::Real>::infinity())
            ? 1
            : 0;
    amrex::Long globalPositiveCount = 0;
    MPI_Allreduce(&localPositiveCount,
                  &globalPositiveCount,
                  1,
                  amrex::ParallelDescriptor::Mpi_typemap<amrex::Long>::type(),
                  MPI_SUM,
                  MPI_COMM_WORLD);

    if (globalPositiveCount <= 0 || !std::isfinite(globalPositiveMin) ||
        !(globalPositiveMin > amrex::Real(0.0))) {
      throw std::runtime_error(
          "Log scaling requested but no positive scalar values were found.");
    }

    scene.scalarTransform.positiveFloor = globalPositiveMin;
    processedMin = std::log(globalPositiveMin);
    processedMax =
        std::log(std::max(globalOriginalScalarMax, globalPositiveMin));
  }

  if (!std::isfinite(processedMin) || !std::isfinite(processedMax)) {
    throw std::runtime_error(options.invalidScalarError);
  }
  if (processedMin == processedMax) {
    processedMax = processedMin + amrex::Real(1.0);
  }

  scene.processedScalarRange = makeScalarRange(processedMin, processedMax);
  scene.hasProcessedScalarRange = true;
  scene.scalarTransform.processedMin = processedMin;
  scene.scalarTransform.processedMax = processedMax;
  scene.scalarTransform.inverseProcessedSpan =
      amrex::Real(1.0) / (processedMax - processedMin);
  scene.scalarTransform.normalizationMin = processedMin;
  scene.scalarTransform.normalizationMax = processedMax;
  scene.scalarTransform.inverseNormalizationSpan =
      scene.scalarTransform.inverseProcessedSpan;

  scene.scalarRange = scene.processedScalarRange;
  scene.hasScalarRange = true;
  if (options.normalizeToDataRange) {
    SetSceneNormalizationRange(scene, processedMin, processedMax);
  }

  scene.localBoxes.shrink_to_fit();
  return scene;
}

void SetSceneNormalizationRange(VolumeRenderer::SceneGeometry& geometry,
                                amrex::Real normalizationMin,
                                amrex::Real normalizationMax) {
  const amrex::Real span = normalizationMax - normalizationMin;
  if (!(span > amrex::Real(0.0)) || !std::isfinite(span)) {
    throw std::runtime_error(
        "Failed to establish a finite scalar range for color mapping.");
  }

  geometry.scalarTransform.normalizeToUnitRange = true;
  geometry.scalarTransform.normalizationMin = normalizationMin;
  geometry.scalarTransform.normalizationMax = normalizationMax;
  geometry.scalarTransform.inverseNormalizationSpan =
      amrex::Real(1.0) / span;
  geometry.scalarRange = {0.0f, 1.0f};
  geometry.hasScalarRange = true;
}

VolumeRenderer::ScalarHistogram ComputeSceneHistogram(
    const VolumeRenderer::SceneGeometry& geometry,
    int binCount) {
  if (binCount <= 0) {
    throw std::invalid_argument("binCount must be positive");
  }

  VolumeRenderer::ScalarHistogram histogram;
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
  if (!(rangeWidth > 0.0f) || !std::isfinite(rangeWidth)) {
    return histogram;
  }

  amrex::Gpu::DeviceVector<unsigned long long> deviceCounts(
      static_cast<std::size_t>(binCount), 0ULL);
  unsigned long long* countsPtr = deviceCounts.data();
  const float inverseWidth = 1.0f / rangeWidth;
  const auto transform = geometry.scalarTransform;

  for (const auto& box : geometry.localBoxes) {
    const int nx = box.validBox.length(0);
    const int ny = box.validBox.length(1);
    const int nz = box.validBox.length(2);
    if (nx <= 0 || ny <= 0 || nz <= 0) {
      continue;
    }

    const amrex::Long planeSize =
        static_cast<amrex::Long>(nx) * static_cast<amrex::Long>(ny);
    const amrex::Long totalCells = planeSize * static_cast<amrex::Long>(nz);
    const amrex::IntVect lo = box.validBox.smallEnd();
    const auto values = box.values;
    const int component = box.component;

    amrex::ParallelFor(
        totalCells,
        [=] AMREX_GPU_DEVICE(amrex::Long linearIndex) noexcept {
          const int localK = static_cast<int>(linearIndex / planeSize);
          const amrex::Long remainder =
              linearIndex - static_cast<amrex::Long>(localK) * planeSize;
          const int localJ = static_cast<int>(
              remainder / static_cast<amrex::Long>(nx));
          const int localI = static_cast<int>(
              remainder - static_cast<amrex::Long>(localJ) *
                              static_cast<amrex::Long>(nx));

          const int i = lo[0] + localI;
          const int j = lo[1] + localJ;
          const int k = lo[2] + localK;

          float value = applyScalarTransform(values(i, j, k, component), transform);
          if (value < rangeMin) {
            value = rangeMin;
          } else if (value > rangeMax) {
            value = rangeMax;
          }

          float normalized = (value - rangeMin) * inverseWidth;
          if (normalized < 0.0f) {
            normalized = 0.0f;
          } else if (normalized > 1.0f) {
            normalized = 1.0f;
          }

          int index = static_cast<int>(normalized * static_cast<float>(binCount));
          if (index >= binCount) {
            index = binCount - 1;
          } else if (index < 0) {
            index = 0;
          }
          amrex::Gpu::Atomic::Add(&countsPtr[index], 1ULL);
        });
  }

  amrex::Gpu::streamSynchronize();

  amrex::Gpu::HostVector<unsigned long long> hostCounts(deviceCounts.size());
  amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                        deviceCounts.begin(),
                        deviceCounts.end(),
                        hostCounts.begin());
  amrex::Gpu::streamSynchronize();

  std::vector<std::uint64_t> localCounts(static_cast<std::size_t>(binCount), 0);
  std::uint64_t localSamples = 0;
  for (int index = 0; index < binCount; ++index) {
    const std::uint64_t count =
        static_cast<std::uint64_t>(hostCounts[static_cast<std::size_t>(index)]);
    localCounts[static_cast<std::size_t>(index)] = count;
    localSamples += count;
  }

  std::vector<std::uint64_t> globalCounts(
      static_cast<std::size_t>(binCount), 0);
  std::uint64_t globalSamples = 0;

  MPI_Allreduce(localCounts.data(),
                globalCounts.data(),
                binCount,
                MPI_UINT64_T,
                MPI_SUM,
                MPI_COMM_WORLD);
  MPI_Allreduce(&localSamples,
                &globalSamples,
                1,
                MPI_UINT64_T,
                MPI_SUM,
                MPI_COMM_WORLD);

  histogram.binCounts = std::move(globalCounts);
  histogram.sampleCount = globalSamples;
  if (!histogram.hasProcessedRange || globalSamples == 0) {
    histogram.binCounts.assign(histogram.binCounts.size(), 0);
  }

  return histogram;
}

}  // namespace detail
}  // namespace amrVolumeRenderer
