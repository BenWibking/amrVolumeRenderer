// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#ifndef AMRVOLUMERENDERER_VOLUME_RENDERER_API_HPP
#define AMRVOLUMERENDERER_VOLUME_RENDERER_API_HPP

#include <AMReX_Geometry.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_RealVect.H>

#include <VolumeRenderer/VolumeRenderer.hpp>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace amrVolumeRenderer {
namespace api {

struct AmrData {
  amrex::Vector<amrex::MultiFab const*> levelData;
  amrex::Vector<amrex::Geometry> levelGeometry;
  amrex::Vector<amrex::IntVect> refinementRatios;
};

struct RenderOptions {
  int width = 512;
  int height = 512;
  float boxTransparency = 0.0f;
  int antialiasing = 1;
  bool visibilityGraph = true;
  bool writeVisibilityGraph = false;
  int minLevel = 0;
  int maxLevel = -1;
  bool logScaleInput = false;
  int component = 0;
  std::string outputFilename = "volume-renderer.ppm";
  std::optional<amrex::RealVect> upVector = std::nullopt;
  std::optional<std::pair<float, float>> scalarRange = std::nullopt;
  std::optional<VolumeRenderer::CameraParameters> camera = std::nullopt;
  std::optional<VolumeRenderer::ColorMap> colorMap = std::nullopt;
};

struct HistogramOptions {
  int minLevel = 0;
  int maxLevel = -1;
  bool logScaleInput = false;
  int binCount = 256;
  int component = 0;
};

int Render(const AmrData& data, const RenderOptions& options);

VolumeRenderer::ScalarHistogram ComputeHistogram(const AmrData& data,
                                                 const HistogramOptions& options);

}  // namespace api
}  // namespace amrVolumeRenderer

#endif  // AMRVOLUMERENDERER_VOLUME_RENDERER_API_HPP
