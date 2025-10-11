// miniGraphics is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Copyright (c) 2017
// National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
// the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
// certain rights in this software.

#ifndef MINIGRAPHICS_VOLUME_PAINTER_VISKORES_HPP
#define MINIGRAPHICS_VOLUME_PAINTER_VISKORES_HPP

#include <miniGraphicsConfig.h>

#if defined(MINIGRAPHICS_ENABLE_VISKORES)

#include <Common/ImageFull.hpp>
#include <Common/VolumeTypes.hpp>

#include <viskores/cont/ColorTable.h>
#include <viskores/cont/DataSet.h>
#include <viskores/rendering/Canvas.h>
#include <viskores/rendering/View3D.h>

#include <vector>

/// \brief Renders volumetric `VolumeBox` data to an `ImageFull` using Viskores.
class VolumePainterViskores {
 public:
  VolumePainterViskores();
  ~VolumePainterViskores();

  void paint(const std::vector<minigraphics::volume::VolumeBox>& boxes,
             const minigraphics::volume::VolumeBounds& bounds,
             int samplesPerAxis,
             int rank,
             int numProcs,
             float boxTransparency,
             ImageFull& image,
             const minigraphics::volume::CameraParameters& camera,
             const viskores::Vec3f_32* colorOverride = nullptr);

 private:
  viskores::cont::DataSet boxesToDataSet(
      const std::vector<minigraphics::volume::VolumeBox>& boxes,
      const minigraphics::volume::VolumeBounds& bounds,
      int samplesPerAxis) const;

  viskores::cont::ColorTable buildColorTable(int numProcs,
                                             float alphaScale) const;

  void setupCamera(viskores::rendering::View3D& view,
                   const minigraphics::volume::CameraParameters& camera);

  void canvasToImage(const viskores::rendering::Canvas& canvas,
                     ImageFull& image,
                     const viskores::Vec3f_32* colorOverride) const;
};

#endif  // MINIGRAPHICS_ENABLE_VISKORES

#endif  // MINIGRAPHICS_VOLUME_PAINTER_VISKORES_HPP
