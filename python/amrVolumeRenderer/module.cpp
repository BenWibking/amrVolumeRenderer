#include <AMReX.H>
#include <AMReX_RealVect.H>
#include <mpi.h>
#include <nanobind/nanobind.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <VolumeRenderer/VolumeRenderer.hpp>
#include <array>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace nb = nanobind;

namespace {

struct RuntimeState {
  bool mpiOwned = false;
  bool mpiInitialized = false;
  bool amrexOwned = false;
  bool amrexInitialized = false;
  int manualRefCount = 0;
};

RuntimeState& runtimeState() {
  static RuntimeState state;
  return state;
}

void ensure_runtime_initialized() {
  auto& state = runtimeState();

  int mpiFinalized = 0;
  MPI_Finalized(&mpiFinalized);
  if (mpiFinalized) {
    throw std::runtime_error(
        "MPI has already been finalized and cannot be re-initialized.");
  }

  int mpiInitFlag = 0;
  MPI_Initialized(&mpiInitFlag);
  if (!mpiInitFlag) {
    int argc = 0;
    char** argv = nullptr;
    MPI_Init(&argc, &argv);
    state.mpiOwned = true;
    state.mpiInitialized = true;
  } else {
    state.mpiInitialized = true;
  }

  if (!amrex::Initialized()) {
    int argc = 0;
    char** argv = nullptr;
    amrex::Initialize(argc, argv, false, MPI_COMM_WORLD);
    state.amrexOwned = true;
    state.amrexInitialized = true;
  } else {
    state.amrexInitialized = true;
  }
}

void finalize_runtime_if_owned() {
  auto& state = runtimeState();

  if (state.amrexOwned && state.amrexInitialized) {
    amrex::Finalize();
    state.amrexInitialized = false;
    state.amrexOwned = false;
  }

  if (state.mpiOwned && state.mpiInitialized) {
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
      MPI_Finalize();
      state.mpiInitialized = false;
    }
    state.mpiOwned = false;
  }
}

class RuntimeScope {
 public:
  RuntimeScope() { ensure_runtime_initialized(); }

  ~RuntimeScope() {
    auto& state = runtimeState();
    if (state.manualRefCount == 0) {
      finalize_runtime_if_owned();
    }
  }

  RuntimeScope(const RuntimeScope&) = delete;
  RuntimeScope& operator=(const RuntimeScope&) = delete;
};

void initialize_runtime() {
  auto& state = runtimeState();
  ensure_runtime_initialized();
  state.manualRefCount++;
}

void finalize_runtime() {
  auto& state = runtimeState();
  if (state.manualRefCount == 0) {
    throw std::runtime_error(
        "amrVolumeRenderer.finalize_runtime requires a matching initialize_runtime call");
  }
  state.manualRefCount--;
  if (state.manualRefCount == 0) {
    finalize_runtime_if_owned();
  }
}

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
                  float boxTransparency = 0.0f,
                  int antialiasing = 1,
                  bool visibilityGraph = true,
                  bool writeVisibilityGraph = false,
                  std::optional<std::string> variableName = std::nullopt,
                  int minLevel = 0,
                  int maxLevel = -1,
                  bool logScaleInput = false,
                  std::optional<std::array<float, 3>> upVector = std::nullopt,
                  std::optional<std::string> outputFilename = std::nullopt,
                  std::optional<std::array<float, 2>> scalarRange = std::nullopt,
                  std::optional<std::array<float, 3>> cameraEye = std::nullopt,
                  std::optional<std::array<float, 3>> cameraLookAt = std::nullopt,
                  std::optional<std::array<float, 3>> cameraUp = std::nullopt,
                  std::optional<float> cameraFovYDegrees = std::nullopt,
                  std::optional<float> cameraNearPlane = std::nullopt,
                  std::optional<float> cameraFarPlane = std::nullopt,
                  std::optional<std::vector<std::array<float, 5>>> colorMap =
                      std::nullopt) {
  RuntimeScope runtime;

  VolumeRenderer::RunOptions options;
  options.plotfilePath = plotfilePath;
  options.parameters.width = width;
  options.parameters.height = height;
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
    amrex::RealVect up(vector[0], vector[1], vector[2]);
    const float length = static_cast<float>(up.vectorLength());
    if (!(length > 0.0f) || !std::isfinite(length)) {
      throw std::invalid_argument(
          "up_vector must contain finite, non-zero components");
    }
    options.parameters.cameraUp = up / length;
    options.parameters.useCustomUp = true;
  } else {
    options.parameters.useCustomUp = false;
  }

  if (scalarRange) {
    const float rangeMin = (*scalarRange)[0];
    const float rangeMax = (*scalarRange)[1];
    if (!std::isfinite(rangeMin) || !std::isfinite(rangeMax) ||
        !(rangeMin < rangeMax)) {
      throw std::invalid_argument(
          "scalar_range must contain two finite values with min < max");
    }
    options.scalarRange = std::make_pair(rangeMin, rangeMax);
  }

  const bool anyCameraParameter =
      cameraEye.has_value() || cameraLookAt.has_value() ||
      cameraUp.has_value() || cameraFovYDegrees.has_value() ||
      cameraNearPlane.has_value() || cameraFarPlane.has_value();

  if (anyCameraParameter) {
    if (!cameraEye || !cameraLookAt) {
      throw std::invalid_argument(
          "camera_eye and camera_look_at must be provided when specifying a "
          "camera");
    }

    const auto toVec3 = [](const std::array<float, 3>& values) {
      return amrex::RealVect(values[0], values[1], values[2]);
    };

    amrex::RealVect eye = toVec3(*cameraEye);
    amrex::RealVect lookAt = toVec3(*cameraLookAt);
    const std::array<float, 3> upArray =
        cameraUp.value_or(std::array<float, 3>{0.0f, 1.0f, 0.0f});
    amrex::RealVect up = toVec3(upArray);
    const float upLength = static_cast<float>(up.vectorLength());
    if (!(upLength > 0.0f) || !std::isfinite(upLength)) {
      throw std::invalid_argument(
          "camera_up must contain finite, non-zero components");
    }
    up = up / upLength;

    const float fovY = cameraFovYDegrees.value_or(45.0f);
    const float nearPlane = cameraNearPlane.value_or(0.1f);
    const float farPlane = cameraFarPlane.value_or(1000.0f);
    options.camera = VolumeRenderer::CameraParameters{
        eye, lookAt, up, fovY, nearPlane, farPlane};
  }

  if (colorMap) {
    VolumeRenderer::ColorMap controlPoints;
    controlPoints.reserve(colorMap->size());
    for (const auto& entry : *colorMap) {
      VolumeRenderer::ColorMapControlPoint point;
      point.value = entry[0];
      point.red = entry[1];
      point.green = entry[2];
      point.blue = entry[3];
      point.alpha = entry[4];
      controlPoints.push_back(point);
    }
    options.colorMap = std::move(controlPoints);
  }

  VolumeRenderer renderer;
  int exitCode = 0;
  {
    nb::gil_scoped_release release;
    exitCode = renderer.run(options);
  }

  if (exitCode != 0) {
    throw std::runtime_error("VolumeRenderer returned exit code " +
                             std::to_string(exitCode));
  }
  return exitCode;
}

NB_MODULE(amrVolumeRenderer_ext, module) {
  module.doc() =
      "Python bindings for the amrVolumeRenderer AMReX volume renderer.";

  module.def(
      "initialize_runtime",
      &initialize_runtime,
      "Initialize MPI and AMReX ahead of multiple render() invocations.");
  module.def("finalize_runtime",
             &finalize_runtime,
             "Finalize MPI and AMReX after initialize_runtime().");
  module.def(
      "render",
      &render_volume,
      nb::arg("plotfile"),
      nb::arg("width") = 512,
      nb::arg("height") = 512,
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
      nb::arg("scalar_range") = nb::none(),
      nb::arg("camera_eye") = nb::none(),
      nb::arg("camera_look_at") = nb::none(),
      nb::arg("camera_up") = nb::none(),
      nb::arg("camera_fov_y") = nb::none(),
      nb::arg("camera_near") = nb::none(),
      nb::arg("camera_far") = nb::none(),
      nb::arg("color_map") = nb::none(),
      "Render a plotfile using the DirectSend compositor.\n\n"
      "Parameters mirror the command line flags of the volume_renderer "
      "executable. Additional keyword arguments allow specifying "
      "scalar_range=(min, max), an explicit camera via camera_* options, and "
      "color_map=[(value, r, g, b, a), ...] with physical scalar values.");
  module.def(
      "compute_histogram",
      [](const std::string& plotfilePath,
         nb::object variableName,
         int minLevel,
         int maxLevel,
         bool logScaleInput,
         int binCount) {
        RuntimeScope runtime;

        std::optional<std::string> variable;
        if (!variableName.is_none()) {
          variable = nb::cast<std::string>(variableName);
        }

        VolumeRenderer renderer;
        VolumeRenderer::ScalarHistogram histogram =
            renderer.computeScalarHistogram(plotfilePath,
                                            variable.value_or(std::string()),
                                            minLevel,
                                            maxLevel,
                                            logScaleInput,
                                            binCount);

        nb::dict result;
        result["counts"] = histogram.binCounts;
        result["normalized_range"] =
            std::array<float, 2>{histogram.normalizedRange.first,
                                 histogram.normalizedRange.second};
        if (histogram.hasProcessedRange) {
          result["processed_range"] =
              std::array<float, 2>{histogram.processedRange.first,
                                   histogram.processedRange.second};
        } else {
          result["processed_range"] = nb::none();
        }
        if (histogram.hasOriginalRange) {
          result["original_range"] =
              std::array<float, 2>{histogram.originalRange.first,
                                   histogram.originalRange.second};
        } else {
          result["original_range"] = nb::none();
        }
        result["samples"] = histogram.sampleCount;
        return result;
      },
      nb::arg("plotfile"),
      nb::arg("variable") = nb::none(),
      nb::arg("min_level") = 0,
      nb::arg("max_level") = -1,
      nb::arg("log_scale") = false,
      nb::arg("bins") = 256,
      "Compute a histogram of normalized scalar values used during rendering.");
}
