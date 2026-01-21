// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#ifndef AMRVOLUMERENDERER_VOLUME_TYPES_HPP
#define AMRVOLUMERENDERER_VOLUME_TYPES_HPP

#include <AMReX_IntVect.H>
#include <AMReX_RealVect.H>

#include <vector>

namespace amrVolumeRenderer {
namespace volume {

struct AmrBox {
  amrex::RealVect minCorner;
  amrex::RealVect maxCorner;
  amrex::IntVect cellDimensions;
  std::vector<float> cellValues;
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
