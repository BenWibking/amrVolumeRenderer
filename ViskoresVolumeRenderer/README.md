# Viskores Volume Renderer

This miniapp demonstrates volumetric rendering with Viskores' `MapperVolume` combined with amrVolumeRenderer's lattice compositing pipeline.

## Overview

- Each MPI rank owns a collection of non-overlapping AMR boxes (uniform bricks with their own cell spacing).
- Every box is passed directly to Viskores as an individual uniform dataset built from the provided cell-centered values.
- Viskores' volume mapper shades each brick while amrVolumeRenderer's DirectSend compositor blends the per-rank renders after visibility ordering.

## Building

The target builds only when Viskores is discoverable by CMake.

```bash
cmake -S . -B build -DAMRVOLUMERENDERER_ENABLE_VISKORES=ON -DViskores_DIR=/path/to/viskores
cmake --build build --target ViskoresVolumeRenderer -j
```

## Running

```bash
mpirun -np 4 build/bin/ViskoresVolumeRenderer \
  --width 512 --height 512 --antialiasing 4
```

### Command Line Options

- `--width` / `--height`: Framebuffer size in pixels (default: 512×512).
- `--box-transparency`: Per-box transparency factor in `[0, 1]` (default: 0).
- `--antialiasing`: Supersampling factor (must be a positive perfect square: 1, 4, 9, ...).
- `--visibility-graph` / `--no-visibility-graph`: Toggle visibility-graph ordering (enabled by default).
- `--output`: Destination filename for the composited image (supports `.ppm` and `.png`, default: `viskores-volume.ppm`).

Images are written on rank 0.
PNG outputs are saved as 8-bit RGB with the alpha channel discarded.

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

  example.renderScene("custom-output.png", params, geometry, camera);

  MPI_Finalize();
}
```

Omitting the camera argument keeps the previous behaviour, allowing the example
to generate an orbiting view automatically using the configured camera seed.

## Implementation Notes

- Global bounds are computed with MPI reductions to keep camera framing consistent across ranks.
- Every AMR brick is converted to a dedicated uniform dataset so its native cell spacing drives the ray-marching step size, even when boxes differ in resolution.
- Brick constant values flow through a jet color map with a shared scalar range, so assigning a distinct scalar per box yields distinct colors automatically.
- An `antialiasing` supersampling factor (1, 4, 9, …) controls screen-space supersampling; the ray-march step size now follows the native AMR spacing and brightness stays consistent across levels.
- Opacity samples are normalized by the ray step so AMR refinement does not change the apparent density of a feature.
- The camera animates around the volume using a randomized orbit (seeded by `cameraSeed`) to illustrate how the composited result changes with view direction.
- Camera state (eye, aim point, up vector, FOV, and clipping range) is computed explicitly for each render pass and passed straight to Viskores, avoiding any OpenGL matrix conversions.
