// miniGraphics is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Copyright (c) 2017
// National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
// the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
// certain rights in this software.

#ifndef MINIGRAPHICS_VOLUME_TYPES_HPP
#define MINIGRAPHICS_VOLUME_TYPES_HPP

#include <miniGraphicsConfig.h>

#ifdef MINIGRAPHICS_ENABLE_VISKORES

#include <viskores/Types.h>

#include <vector>

namespace minigraphics {
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
}  // namespace minigraphics

#endif  // MINIGRAPHICS_ENABLE_VISKORES

#endif  // MINIGRAPHICS_VOLUME_TYPES_HPP
