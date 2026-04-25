// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#ifndef AMRVOLUMERENDERER_VOLUME_TYPES_HPP
#define AMRVOLUMERENDERER_VOLUME_TYPES_HPP

#include <AMReX_Array4.H>
#include <AMReX_Box.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_Math.H>
#include <AMReX_RealVect.H>

#include <cmath>
#include <vector>

namespace amrVolumeRenderer {
namespace volume {

struct ScalarTransform {
  bool logScaleInput = false;
  bool normalizeToUnitRange = false;
  amrex::Real positiveFloor = amrex::Real(0.0);
  amrex::Real processedMin = amrex::Real(0.0);
  amrex::Real processedMax = amrex::Real(1.0);
  amrex::Real inverseProcessedSpan = amrex::Real(1.0);
  amrex::Real normalizationMin = amrex::Real(0.0);
  amrex::Real normalizationMax = amrex::Real(1.0);
  amrex::Real inverseNormalizationSpan = amrex::Real(1.0);
};

AMREX_GPU_HOST_DEVICE inline amrex::Real sanitizeScalarSample(
    amrex::Real raw) noexcept {
  return amrex::Math::isfinite(raw) ? raw : amrex::Real(0.0);
}

AMREX_GPU_HOST_DEVICE inline amrex::Real toProcessedScalar(
    amrex::Real raw,
    const ScalarTransform& transform) noexcept {
  amrex::Real processed = sanitizeScalarSample(raw);
  if (transform.logScaleInput) {
    if (!(processed > amrex::Real(0.0))) {
      processed = transform.positiveFloor;
    } else if (processed < transform.positiveFloor) {
      processed = transform.positiveFloor;
    }
    processed = std::log(processed);
  }
  return processed;
}

AMREX_GPU_HOST_DEVICE inline float applyScalarTransform(
    amrex::Real raw,
    const ScalarTransform& transform) noexcept {
  amrex::Real value = toProcessedScalar(raw, transform);
  if (transform.normalizeToUnitRange) {
    value =
        (value - transform.normalizationMin) * transform.inverseNormalizationSpan;
    if (value < amrex::Real(0.0)) {
      value = amrex::Real(0.0);
    } else if (value > amrex::Real(1.0)) {
      value = amrex::Real(1.0);
    }
  }
  return static_cast<float>(value);
}

struct AmrBox {
  amrex::RealVect minCorner;
  amrex::RealVect maxCorner;
  amrex::IntVect cellDimensions;
  amrex::Box validBox;
  amrex::Array4<amrex::Real const> values;
  int component = 0;
};

struct VolumeBounds {
  amrex::RealVect minCorner;
  amrex::RealVect maxCorner;
};

struct CameraParameters {
  amrex::RealVect eye;
  amrex::RealVect lookAt;
  amrex::RealVect up;
  float fovYDegrees;
  float nearPlane;
  float farPlane;
};

struct ColorMapControlPoint {
  float value;
  float red;
  float green;
  float blue;
  float alpha;
};

using ColorMap = std::vector<ColorMapControlPoint>;

}  // namespace volume
}  // namespace amrVolumeRenderer

#endif  // AMRVOLUMERENDERER_VOLUME_TYPES_HPP
