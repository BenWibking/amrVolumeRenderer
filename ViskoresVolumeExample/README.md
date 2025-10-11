# Viskores Volume Example

This miniapp demonstrates volumetric rendering with Viskores' `MapperVolume` combined with miniGraphics' lattice compositing pipeline.

## Overview

- Each MPI rank owns a subset of axis-aligned boxes defined on a global lattice.
- The boxes are converted into a shared structured volume where voxels inside a rank's boxes receive a per-rank scalar.
- Viskores' volume mapper shades the scalar field while miniGraphics' DirectSend compositor merges the per-rank renders.

## Building

The target builds only when Viskores is discoverable by CMake.

```bash
cmake -S . -B build -DMINIGRAPHICS_ENABLE_VISKORES=ON -DViskores_DIR=/path/to/viskores
cmake --build build --target ViskoresVolumeExample -j
```

## Running

```bash
mpirun -np 4 build/bin/ViskoresVolumeExample \
  --width 512 --height 512 --samples 96 --trials 3
```

### Command Line Options

- `--width` / `--height`: Framebuffer size in pixels (default: 512×512).
- `--trials`: Number of render iterations for timing (default: 1).
- `--samples`: Resolution of the structured volume along each axis (default: 64).
- `--box-transparency`: Per-box transparency factor in `[0, 1]` (default: 0).
- `--visibility-graph` / `--no-visibility-graph`: Toggle visibility-graph ordering (enabled by default).
- `--output`: Destination filename for the composited image (default: `viskores-volume-trial.ppm`).

Images are written on rank 0. When more than one trial is requested, each filename receives a `-trial-<N>` suffix inserted before the extension.

## Library Usage

The rendering pipeline can be invoked programmatically. Provide a scene description (per-rank boxes plus optional explicit bounds), configure rendering parameters, and call `renderScene`:

```cpp
#include "ViskoresVolumeExample.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  ViskoresVolumeExample example;

  ViskoresVolumeExample::SceneGeometry geometry;
  geometry.localBoxes = /* rank-local boxes */;
  // Optionally set geometry.explicitBounds and geometry.hasExplicitBounds.

  ViskoresVolumeExample::RenderParameters params;
  params.width = 800;
  params.height = 600;

  example.renderScene("custom-output.ppm", params, geometry);

  MPI_Finalize();
}
```

## Implementation Notes

- Global bounds are computed with MPI reductions to keep camera framing consistent across ranks.
- The scalar value assigned to a voxel encodes the owning rank; a shared gather step ensures every rank shades the full set of boxes while colors remain tied to the original owners.
- The camera animates around the volume for each trial to illustrate how the composited result changes with view direction.
- Camera state (eye, aim point, up vector, FOV, and clipping range) is computed explicitly per trial and passed straight to Viskores, avoiding any OpenGL matrix conversions.
