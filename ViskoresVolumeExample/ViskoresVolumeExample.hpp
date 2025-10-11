#ifndef VISKORES_VOLUME_EXAMPLE_HPP
#define VISKORES_VOLUME_EXAMPLE_HPP

#include <miniGraphicsConfig.h>

#include <Common/Compositor.hpp>
#include <Common/ImageFull.hpp>

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

  struct VolumeBox {
    viskores::Vec3f_32 minCorner;
    viskores::Vec3f_32 maxCorner;
    float scalarValue;
    viskores::Vec3f_32 color;
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
                                        MPI_Group baseGroup) const;

  int rank;
  int numProcs;

  mutable viskores::Vec3f_32 localCentroid;
  mutable viskores::Vec3f_32 localBoundsMin;
  mutable viskores::Vec3f_32 localBoundsMax;
  mutable bool hasLocalData;
};

#endif  // VISKORES_VOLUME_EXAMPLE_HPP
