#ifndef VISKORES_VOLUME_EXAMPLE_HPP
#define VISKORES_VOLUME_EXAMPLE_HPP

#include <miniGraphicsConfig.h>

#include <Common/Compositor.hpp>
#include <Common/ImageFull.hpp>
#include <Common/VolumeTypes.hpp>

#include <viskores/Types.h>

#include <mpi.h>

#include <vector>

#ifndef MINIGRAPHICS_ENABLE_VISKORES
#error "ViskoresVolumeExample requires MINIGRAPHICS_ENABLE_VISKORES"
#endif

/// \brief Miniapp that renders distributed volumetric data using Viskores.
class ViskoresVolumeExample {
 public:
  ViskoresVolumeExample();

  /// \brief Execute the miniapp entry point.
  int run(int argc, char** argv);

  using VolumeBox = minigraphics::volume::VolumeBox;
  using VolumeBounds = minigraphics::volume::VolumeBounds;
  using CameraParameters = minigraphics::volume::CameraParameters;

 private:
  void initialize() const;
  std::vector<VolumeBox> createRankSpecificBoxes(VolumeBounds& globalBounds) const;
  void paint(const std::vector<VolumeBox>& boxes,
             const VolumeBounds& bounds,
             int samplesPerAxis,
             float boxTransparency,
             ImageFull& image,
             const CameraParameters& camera,
             const viskores::Vec3f_32* colorOverride = nullptr);
  Compositor* getCompositor();
  MPI_Group buildVisibilityOrderedGroup(const CameraParameters& camera,
                                        float aspect,
                                        MPI_Group baseGroup,
                                        bool useVisibilityGraph,
                                        const std::vector<VolumeBox>& localBoxes) const;

  int rank;
  int numProcs;

};

#endif  // VISKORES_VOLUME_EXAMPLE_HPP
