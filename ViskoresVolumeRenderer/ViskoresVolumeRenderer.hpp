#ifndef VISKORES_VOLUME_RENDERER_HPP
#define VISKORES_VOLUME_RENDERER_HPP

#include <amrVolumeRendererConfig.h>
#include <mpi.h>
#include <viskores/Types.h>

#include <Common/Compositor.hpp>
#include <Common/ImageFull.hpp>
#include <Common/VolumeTypes.hpp>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#ifndef AMRVOLUMERENDERER_ENABLE_VISKORES
#error "ViskoresVolumeRenderer requires AMRVOLUMERENDERER_ENABLE_VISKORES"
#endif

/// \brief Miniapp that renders distributed volumetric data using Viskores.
class ViskoresVolumeRenderer {
 public:
  ViskoresVolumeRenderer();

  /// \brief Execute the miniapp entry point.
  int run(int argc, char** argv);

  using AmrBox = amrVolumeRenderer::volume::AmrBox;
  using VolumeBounds = amrVolumeRenderer::volume::VolumeBounds;
  using CameraParameters = amrVolumeRenderer::volume::CameraParameters;
  using ColorMap = amrVolumeRenderer::volume::ColorMap;
  using ColorMapControlPoint = amrVolumeRenderer::volume::ColorMapControlPoint;

  struct RenderParameters {
    int width = 512;
    int height = 512;
    float boxTransparency = 0.0f;
    int antialiasing = 1;
    bool useVisibilityGraph = true;
    bool writeVisibilityGraph = false;
    unsigned int cameraSeed = 91021u;
    viskores::Vec3f_32 cameraUp = viskores::Vec3f_32(0.0f, 1.0f, 0.0f);
    bool useCustomUp = false;
    bool printCamera = false;
  };

  struct SceneGeometry {
    std::vector<AmrBox> localBoxes;
    VolumeBounds explicitBounds;
    bool hasExplicitBounds = false;
    std::pair<float, float> scalarRange = {0.0f, 1.0f};
    bool hasScalarRange = false;
    std::pair<float, float> processedScalarRange = {0.0f, 1.0f};
    bool hasProcessedScalarRange = false;
    std::pair<float, float> originalScalarRange = {0.0f, 1.0f};
    bool hasOriginalScalarRange = false;
  };

  struct RunOptions {
    RenderParameters parameters;
    std::string outputFilename = "viskores-volume.ppm";
    std::string plotfilePath;
    std::string variableName;
    int minLevel = 0;
    int maxLevel = -1;
    bool logScaleInput = false;
    bool exitEarly = false;
    std::optional<CameraParameters> camera;
    std::optional<std::pair<float, float>> scalarRange;
    std::optional<ColorMap> colorMap;
  };

  struct ScalarHistogram {
    std::vector<std::uint64_t> binCounts;
    std::pair<float, float> normalizedRange = {0.0f, 1.0f};
    std::pair<float, float> processedRange = {0.0f, 1.0f};
    bool hasProcessedRange = false;
    std::pair<float, float> originalRange = {0.0f, 1.0f};
    bool hasOriginalRange = false;
    std::uint64_t sampleCount = 0;
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
                  const SceneGeometry& geometry,
                  const std::optional<ColorMap>& colorMap = std::nullopt);

  /// \brief Render the provided scene geometry with an explicit camera.
  ///
  /// \param outputFilenameBase Base name used when saving rendered images.
  /// \param parameters Rendering parameters such as resolution and sampling.
  /// \param geometry Scene geometry containing per-rank volume boxes and
  ///                 optional explicit volume bounds.
  /// \param camera Camera parameters describing the view transform.
  /// \return 0 on success, non-zero on failure.
  int renderScene(const std::string& outputFilenameBase,
                  const RenderParameters& parameters,
                  const SceneGeometry& geometry,
                  const CameraParameters& camera,
                  const std::optional<ColorMap>& colorMap = std::nullopt);

  /// \brief Execute the miniapp using pre-parsed run options.
  int run(const RunOptions& options);

  ScalarHistogram computeScalarHistogram(const std::string& plotfilePath,
                                         const std::string& variableName,
                                         int requestedMinLevel,
                                         int requestedMaxLevel,
                                         bool logScaleInput,
                                         int binCount) const;

 private:
  void validateRenderParameters(const RenderParameters& parameters) const;
  void initialize() const;
  SceneGeometry loadPlotFileGeometry(const std::string& plotfilePath,
                                     const std::string& variableName,
                                     int requestedMinLevel,
                                     int requestedMaxLevel,
                                     bool logScaleInput) const;
  VolumeBounds computeGlobalBounds(const std::vector<AmrBox>& boxes,
                                   bool hasExplicitBounds,
                                   const VolumeBounds& explicitBounds) const;
  VolumeBounds computeTightBounds(const std::vector<AmrBox>& boxes,
                                  const VolumeBounds& fallback) const;
  std::pair<float, float> computeGlobalScalarRange(
      const std::vector<AmrBox>& boxes) const;
  void paint(const AmrBox& box,
             const VolumeBounds& bounds,
             const std::pair<float, float>& scalarRange,
             float boxTransparency,
             int antialiasing,
             float referenceSampleDistance,
             ImageFull& image,
             const CameraParameters& camera,
             const ColorMap* colorMap);
  Compositor* getCompositor();
  MPI_Group buildVisibilityOrderedGroup(
      const CameraParameters& camera,
      float aspect,
      MPI_Group baseGroup,
      bool useVisibilityGraph,
      bool writeVisibilityGraph,
      const std::vector<AmrBox>& localBoxes) const;
  int renderSingleTrial(const std::string& outputFilename,
                        const RenderParameters& parameters,
                        const SceneGeometry& geometry,
                        const VolumeBounds& bounds,
                        const std::pair<float, float>& scalarRange,
                        Compositor* compositor,
                        MPI_Group baseGroup,
                        const CameraParameters& camera,
                        const ColorMap* colorMap);

  int rank;
  int numProcs;
};

#endif  // VISKORES_VOLUME_RENDERER_HPP
