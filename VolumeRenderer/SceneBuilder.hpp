// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#ifndef AMRVOLUMERENDERER_SCENE_BUILDER_HPP
#define AMRVOLUMERENDERER_SCENE_BUILDER_HPP

#include <AMReX_Array.H>

#include <VolumeRenderer/VolumeRenderer.hpp>

#include <string>

namespace amrVolumeRenderer {
namespace detail {

struct LevelGridGeometry {
  amrex::Array<amrex::Real, AMREX_SPACEDIM> probLo;
  amrex::Array<amrex::Real, AMREX_SPACEDIM> cellSize;
};

struct SceneBuildOptions {
  int minLevel = 0;
  int maxLevel = -1;
  int component = 0;
  bool logScaleInput = false;
  bool normalizeToDataRange = true;
  std::string noDataError;
  std::string invalidScalarError;
};

VolumeRenderer::SceneGeometry BuildSceneGeometry(
    std::shared_ptr<amrex::Vector<amrex::MultiFab>> ownedLevels,
    const amrex::Vector<LevelGridGeometry>& levelGeometry,
    const SceneBuildOptions& options);

void SetSceneNormalizationRange(VolumeRenderer::SceneGeometry& geometry,
                                amrex::Real normalizationMin,
                                amrex::Real normalizationMax);

VolumeRenderer::ScalarHistogram ComputeSceneHistogram(
    const VolumeRenderer::SceneGeometry& geometry,
    int binCount);

}  // namespace detail
}  // namespace amrVolumeRenderer

#endif  // AMRVOLUMERENDERER_SCENE_BUILDER_HPP
