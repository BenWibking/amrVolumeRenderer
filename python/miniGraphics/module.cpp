#include <AMReX.H>
#include <mpi.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <viskores/VectorAnalysis.h>

#include <ViskoresVolumeRenderer/ViskoresVolumeRenderer.hpp>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

class RuntimeScope {
 public:
  RuntimeScope() {
    int mpiInitialized = 0;
    MPI_Initialized(&mpiInitialized);
    if (!mpiInitialized) {
      int argc = 0;
      char** argv = nullptr;
      MPI_Init(&argc, &argv);
      ownsMpi_ = true;
    }

    amrexInitialized_ = amrex::Initialized();
    if (!amrexInitialized_) {
      int argc = 0;
      char** argv = nullptr;
      amrex::Initialize(argc, argv, false, MPI_COMM_WORLD);
    }
  }

  ~RuntimeScope() {
    if (!amrexInitialized_) {
      amrex::Finalize();
    }
    if (ownsMpi_) {
      MPI_Finalize();
    }
  }

  RuntimeScope(const RuntimeScope&) = delete;
  RuntimeScope& operator=(const RuntimeScope&) = delete;

 private:
  bool ownsMpi_ = false;
  bool amrexInitialized_ = false;
};

std::string normalizeOutput(const std::optional<std::string>& requested,
                            const std::string& fallback) {
  if (requested && requested->empty()) {
    throw std::invalid_argument("output filename must not be empty");
  }
  return requested ? *requested : fallback;
}

}  // namespace

int render_volume(const std::string& plotfilePath,
                  int width = 512,
                  int height = 512,
                  int trials = 1,
                  float boxTransparency = 0.0f,
                  int antialiasing = 1,
                  bool visibilityGraph = true,
                  bool writeVisibilityGraph = false,
                  std::optional<std::string> variableName = std::nullopt,
                  int minLevel = 0,
                  int maxLevel = -1,
                  bool logScaleInput = false,
                  std::optional<std::array<float, 3>> upVector = std::nullopt,
                  std::optional<std::string> outputFilename = std::nullopt) {
  RuntimeScope runtime;

  ViskoresVolumeRenderer::RunOptions options;
  options.plotfilePath = plotfilePath;
  options.parameters.width = width;
  options.parameters.height = height;
  options.parameters.trials = trials;
  options.parameters.boxTransparency = boxTransparency;
  options.parameters.antialiasing = antialiasing;
  options.parameters.useVisibilityGraph = visibilityGraph;
  options.parameters.writeVisibilityGraph = writeVisibilityGraph;
  options.logScaleInput = logScaleInput;
  options.minLevel = minLevel;
  options.maxLevel = maxLevel;
  options.outputFilename =
      normalizeOutput(outputFilename, options.outputFilename);

  if (variableName) {
    options.variableName = *variableName;
  } else {
    options.variableName.clear();
  }

  if (upVector) {
    const auto& vector = *upVector;
    viskores::Vec3f_32 up(vector[0], vector[1], vector[2]);
    const float length = viskores::Magnitude(up);
    if (!(length > 0.0f) || !std::isfinite(length)) {
      throw std::invalid_argument(
          "up_vector must contain finite, non-zero components");
    }
    options.parameters.cameraUp = up / length;
    options.parameters.useCustomUp = true;
  } else {
    options.parameters.useCustomUp = false;
  }

  ViskoresVolumeRenderer renderer;
  int exitCode = 0;
  {
    nb::gil_scoped_release release;
    exitCode = renderer.run(options);
  }

  if (exitCode != 0) {
    throw std::runtime_error("ViskoresVolumeRenderer returned exit code " +
                             std::to_string(exitCode));
  }
  return exitCode;
}

NB_MODULE(miniGraphics_ext, module) {
  module.doc() =
      "Python bindings for the miniGraphics Viskores volume renderer.";

  module.def(
      "render",
      &render_volume,
      nb::arg("plotfile"),
      nb::arg("width") = 512,
      nb::arg("height") = 512,
      nb::arg("trials") = 1,
      nb::arg("box_transparency") = 0.0f,
      nb::arg("antialiasing") = 1,
      nb::arg("visibility_graph") = true,
      nb::arg("write_visibility_graph") = false,
      nb::arg("variable") = nb::none(),
      nb::arg("min_level") = 0,
      nb::arg("max_level") = -1,
      nb::arg("log_scale") = false,
      nb::arg("up_vector") = nb::none(),
      nb::arg("output") = nb::none(),
      "Render a plotfile using the DirectSend compositor.\n\n"
      "Parameters mirror the command line flags of the ViskoresVolumeRenderer "
      "executable.");
}
