// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#ifndef AMRVOLUMERENDERER_VOLUME_TYPES_HPP
#define AMRVOLUMERENDERER_VOLUME_TYPES_HPP

#include <amrVolumeRendererConfig.h>

#ifdef AMRVOLUMERENDERER_ENABLE_VISKORES

#include <viskores/Types.h>

#include <vector>

namespace amrVolumeRenderer {
namespace volume {

struct AmrBox {
  viskores::Vec3f_32 minCorner;
  viskores::Vec3f_32 maxCorner;
  viskores::Id3 cellDimensions;
  std::vector<float> cellValues;
};

struct VolumeBounds {
  viskores::Vec3f_32 minCorner;
  viskores::Vec3f_32 maxCorner;
};

struct CameraParameters {
  viskores::Vec3f_32 eye;
  viskores::Vec3f_32 lookAt;
  viskores::Vec3f_32 up;
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

#endif  // AMRVOLUMERENDERER_ENABLE_VISKORES

#endif  // AMRVOLUMERENDERER_VOLUME_TYPES_HPP
