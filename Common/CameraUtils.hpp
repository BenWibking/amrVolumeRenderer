// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#ifndef AMRVOLUMERENDERER_CAMERA_UTILS_HPP
#define AMRVOLUMERENDERER_CAMERA_UTILS_HPP

#include <AMReX_RealVect.H>
#include <AMReX_SmallMatrix.H>

#include <cmath>

namespace amrVolumeRenderer {
namespace camera {

inline amrex::RealVect safeNormalize(const amrex::RealVect& input) {
  const amrex::Real length = input.vectorLength();
  if (length > 0.0 && std::isfinite(static_cast<double>(length))) {
    return input / length;
  }
  return amrex::RealVect(0.0, 0.0, -1.0);
}

inline amrex::SmallMatrix<float, 4, 4> makeViewMatrix(
    const amrex::RealVect& eye,
    const amrex::RealVect& lookAt,
    const amrex::RealVect& up) {
  const amrex::RealVect forward = safeNormalize(lookAt - eye);
  amrex::RealVect right = forward.crossProduct(up);
  const amrex::Real rightLength = right.vectorLength();
  if (rightLength > 0.0 && std::isfinite(static_cast<double>(rightLength))) {
    right /= rightLength;
  } else {
    right = amrex::RealVect(1.0, 0.0, 0.0);
  }
  const amrex::RealVect upOrtho = right.crossProduct(forward);

  amrex::SmallMatrix<float, 4, 4> view =
      amrex::SmallMatrix<float, 4, 4>::Identity();
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

}  // namespace camera
}  // namespace amrVolumeRenderer

#endif  // AMRVOLUMERENDERER_CAMERA_UTILS_HPP
