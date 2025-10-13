# Viskores Volume Renderer

This miniapp demonstrates volumetric rendering with Viskores' `MapperVolume` combined with miniGraphics' lattice compositing pipeline.

## Overview

- Each MPI rank owns a collection of non-overlapping AMR boxes (uniform bricks with their own cell spacing).
- Every box is passed directly to Viskores as an individual uniform dataset built from the provided cell-centered values.
- Viskores' volume mapper shades each brick while miniGraphics' DirectSend compositor blends the per-rank renders after visibility ordering.

## Building

The target builds only when Viskores is discoverable by CMake.

```bash
cmake -S . -B build -DMINIGRAPHICS_ENABLE_VISKORES=ON -DViskores_DIR=/path/to/viskores
cmake --build build --target ViskoresVolumeRenderer -j
```

## Running

```bash
mpirun -np 4 build/bin/ViskoresVolumeRenderer \
  --width 512 --height 512 --trials 3 --antialiasing 4
```

### Command Line Options

- `--width` / `--height`: Framebuffer size in pixels (default: 512×512).
- `--trials`: Number of render iterations for timing (default: 1).
- `--box-transparency`: Per-box transparency factor in `[0, 1]` (default: 0).
- `--antialiasing`: Supersampling factor (must be a positive perfect square: 1, 4, 9, ...).
- `--visibility-graph` / `--no-visibility-graph`: Toggle visibility-graph ordering (enabled by default).
- `--output`: Destination filename for the composited image (default: `viskores-volume-trial.ppm`).

Images are written on rank 0. When more than one trial is requested, each filename receives a `-trial-<N>` suffix inserted before the extension.

## Library Usage

The rendering pipeline can be invoked programmatically. Provide a scene description (per-rank boxes plus optional explicit bounds), configure rendering parameters, and call `renderScene`:

```cpp
#include "ViskoresVolumeRenderer.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  ViskoresVolumeRenderer example;

  ViskoresVolumeRenderer::SceneGeometry geometry;
  ViskoresVolumeRenderer::AmrBox box;
  box.minCorner = {-0.5f, -0.5f, -0.5f};
  box.maxCorner = {0.5f, 0.5f, 0.5f};
  box.cellDimensions = viskores::Id3(32, 32, 32);
  const float cellValue = 1.0f;  // pick a distinct scalar per AMR box
  box.cellValues.assign(32 * 32 * 32, cellValue);
  geometry.localBoxes.push_back(box);
  // Optionally set geometry.explicitBounds and geometry.hasExplicitBounds.

  ViskoresVolumeRenderer::RenderParameters params;
  params.width = 800;
  params.height = 600;
  params.antialiasing = 4;  // 2x2 supersampling

  ViskoresVolumeRenderer::CameraParameters camera;
  camera.eye = {0.0f, 0.5f, 3.0f};
  camera.lookAt = {0.0f, 0.0f, 0.0f};
  camera.up = {0.0f, 1.0f, 0.0f};
  camera.fovYDegrees = 45.0f;
  camera.nearPlane = 0.1f;
  camera.farPlane = 20.0f;

  example.renderScene("custom-output.ppm", params, geometry, camera);

  MPI_Finalize();
}
```

Omitting the camera argument keeps the previous behaviour, allowing the example
to generate an orbiting view for each trial automatically.

## Implementation Notes

- Global bounds are computed with MPI reductions to keep camera framing consistent across ranks.
- Every AMR brick is converted to a dedicated uniform dataset so its native cell spacing drives the ray-marching step size, even when boxes differ in resolution.
- Brick constant values flow through a jet color map with a shared scalar range, so assigning a distinct scalar per box yields distinct colors automatically.
- An `antialiasing` supersampling factor (1, 4, 9, …) controls the ray-march sample spacing, allowing higher-quality renders at the cost of more work.
- The camera animates around the volume for each trial to illustrate how the composited result changes with view direction.
- Camera state (eye, aim point, up vector, FOV, and clipping range) is computed explicitly per trial and passed straight to Viskores, avoiding any OpenGL matrix conversions.
