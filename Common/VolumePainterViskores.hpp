// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#ifndef AMRVOLUMERENDERER_VOLUME_PAINTER_VISKORES_HPP
#define AMRVOLUMERENDERER_VOLUME_PAINTER_VISKORES_HPP

#include <amrVolumeRendererConfig.h>

#if defined(AMRVOLUMERENDERER_ENABLE_VISKORES)

#include <Common/ImageFull.hpp>
#include <Common/VolumeTypes.hpp>

#include <viskores/cont/ColorTable.h>
#include <viskores/cont/DataSet.h>
#include <viskores/rendering/Canvas.h>
#include <viskores/rendering/View3D.h>

#include <utility>
#include <vector>

/// \brief Renders volumetric AMR box data to an `ImageFull` using Viskores.
class VolumePainterViskores {
 public:
  VolumePainterViskores();
  ~VolumePainterViskores();

  void paint(const amrVolumeRenderer::volume::AmrBox& box,
             const amrVolumeRenderer::volume::VolumeBounds& bounds,
             const std::pair<float, float>& scalarRange,
             int rank,
             int numProcs,
             float boxTransparency,
             int antialiasing,
             float referenceSampleDistance,
             ImageFull& image,
             const amrVolumeRenderer::volume::CameraParameters& camera,
             const amrVolumeRenderer::volume::ColorMap* colorMap);

 private:
  viskores::cont::DataSet boxToDataSet(
      const amrVolumeRenderer::volume::AmrBox& box,
      const std::pair<float, float>& scalarRange) const;

  viskores::cont::ColorTable buildColorTable(
      float alphaScale,
      float normalizationFactor,
      const std::pair<float, float>& scalarRange,
      const amrVolumeRenderer::volume::ColorMap* colorMap) const;

  void setupCamera(viskores::rendering::View3D& view,
                   const amrVolumeRenderer::volume::CameraParameters& camera);

  void canvasToImage(const viskores::rendering::Canvas& canvas,
                     ImageFull& image) const;
};

#endif  // AMRVOLUMERENDERER_ENABLE_VISKORES

#endif  // AMRVOLUMERENDERER_VOLUME_PAINTER_VISKORES_HPP
