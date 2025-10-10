#ifndef VISKORES_VOLUME_EXAMPLE_HPP
#define VISKORES_VOLUME_EXAMPLE_HPP

#include <miniGraphicsConfig.h>

#include <Common/Compositor.hpp>
#include <Common/ImageFull.hpp>

#include <glm/vec3.hpp>

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
    glm::vec3 minCorner;
    glm::vec3 maxCorner;
    float scalarValue;
  };

  struct VolumeBounds {
    glm::vec3 minCorner;
    glm::vec3 maxCorner;
  };

  struct CameraParameters {
    glm::vec3 eye;
    glm::vec3 lookAt;
    glm::vec3 up;
    float fovYDegrees;
    float nearPlane;
    float farPlane;
  };

 private:
  void initialize() const;
  std::vector<VolumeBox> createRankSpecificBoxes(VolumeBounds& globalBounds) const;
  double paint(const std::vector<VolumeBox>& boxes,
               const VolumeBounds& bounds,
               int samplesPerAxis,
               ImageFull& image,
               const CameraParameters& camera);
  Compositor* getCompositor();
  MPI_Group buildVisibilityOrderedGroup(const CameraParameters& camera,
                                        float aspect,
                                        MPI_Group baseGroup) const;

  int rank;
  int numProcs;

  mutable glm::vec3 localCentroid;
  mutable bool hasLocalData;
};

#endif  // VISKORES_VOLUME_EXAMPLE_HPP
