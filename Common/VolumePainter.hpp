// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#ifndef AMRVOLUMERENDERER_VOLUME_PAINTER_HPP
#define AMRVOLUMERENDERER_VOLUME_PAINTER_HPP

#include <Common/ImageFull.hpp>
#include <Common/VolumeTypes.hpp>

#include <utility>
#include <vector>

/// \brief Renders volumetric AMR box data to an `ImageFull` using the renderer backend.
class VolumePainter {
 public:
  VolumePainter();
  ~VolumePainter();

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
};

#endif  // AMRVOLUMERENDERER_VOLUME_PAINTER_HPP
