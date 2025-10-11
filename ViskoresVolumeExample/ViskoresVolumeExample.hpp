#ifndef VISKORES_VOLUME_EXAMPLE_HPP
#define VISKORES_VOLUME_EXAMPLE_HPP

#include <miniGraphicsConfig.h>

#include <Common/Compositor.hpp>
#include <Common/ImageFull.hpp>
#include <Common/VolumeTypes.hpp>

#include <viskores/Types.h>

#include <mpi.h>

#include <string>
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

  struct RenderParameters {
    int width = 512;
    int height = 512;
    int trials = 1;
    int samplesPerAxis = 64;
    float boxTransparency = 0.0f;
    bool useVisibilityGraph = true;
    unsigned int cameraSeed = 91021u;
  };

  struct SceneGeometry {
    std::vector<VolumeBox> localBoxes;
    VolumeBounds explicitBounds;
    bool hasExplicitBounds = false;
  };

  /// \brief Render the provided scene geometry using the configured compositor.
  ///
  /// \param outputFilenameBase Base name used when saving rendered images.
  /// \param parameters Rendering parameters such as resolution and sampling.
  /// \param geometry Scene geometry containing per-rank volume boxes and
  ///                 optional explicit volume bounds.
  /// \return 0 on success, non-zero on failure.
  int renderScene(const std::string& outputFilenameBase,
                  const RenderParameters& parameters,
                  const SceneGeometry& geometry);

 private:
  void initialize() const;
  SceneGeometry createRankSpecificGeometry() const;
  VolumeBounds computeGlobalBounds(const std::vector<VolumeBox>& boxes,
                                   bool hasExplicitBounds,
                                   const VolumeBounds& explicitBounds) const;
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
