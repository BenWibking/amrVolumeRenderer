# Splat renderer (prototype)

`SplatRenderer/` draws the Gaussian splat files written by `gaussian_export`.
It contains a CUDA backend meant for interactive use and a multithreaded CPU
backend that serves as its reference. The library has no AMReX, MPI or
graphics-API dependency, so a display application can link it directly. The
planned target is a native plugin for the getReal3D/Unity stack on the
VizLab wall, which would draw the CUDA output into a Direct3D 11 texture
through CUDA–D3D interop.

## Building

```bash
# standalone, with CUDA (CUDA 12+; tested with 13.4 on an RTX PRO 6000 Blackwell)
cmake -S SplatRenderer -B build-splat -DSPLAT_RENDERER_ENABLE_CUDA=ON
cmake --build build-splat -j
ctest --test-dir build-splat
```

Without `-DSPLAT_RENDERER_ENABLE_CUDA=ON` only the CPU backend is built; the
CUDA class becomes a stub that throws. The directory is also built as part of
amrVolumeRenderer.

## `splat_render`

```bash
build-splat/bin/splat_render disk.splat --eye 0 -25 18 --target 0 0 0 --up 0 0 1 \
    --width 1920 --height 1080 --output disk.ppm
# wall-style off-axis stereo for one screen
splat_render disk.splat --eye 0.1 0 3 \
    --screen -2 -1 0  2 -1 0  -2 1 0 --stereo 0.065 --width 9600 --height 1890
# benchmark without data
splat_render --synthetic-disk 30000000 --eye 0 0 40 --fov 9.6 \
    --width 9600 --height 1890 --stereo 0.2 --density-range -26 -21 --extinction 100 --frames 20
```

`--compare` renders the same frame with both backends and reports the
difference. Run `--help` for the camera, LOD and transfer-function options.

## Pipeline

1. **Preprocess** (one thread per record):
   * Select the record with the stateless LOD rule from the export.
   * Drop records whose phase is masked out or whose density is below the
     transfer-function range.
   * Look up the record's extinction (from log density) and colour (from log
     temperature).
   * Project it with an off-axis pinhole camera defined by the eye and the
     screen rectangle (EWA splatting, with a 0.3 px² low-pass dilation).
   * Count the 16×16 tiles it overlaps, after cutting its extent where the
     opacity would fall below 1/255.
2. **Sort**:
   * Prefix-sum the tile counts and emit one key per overlapped tile, with the
     tile index in the high 32 bits and the depth in the low 32.
   * Radix-sort the keys (CUB), then find each tile's range in the sorted list.
3. **Rasterize**: one CUDA block per tile blends that tile's splats front to
   back, stopping once every pixel in the tile is opaque.

**Optical model.** The optical depth is
`tau = kappa(rho) · V · |det J| · N(x)`, where:
* `V` is the volume the record occupies;
* `J` is the projection Jacobian, so `|det J|` converts world area to pixels;
* `N` is the projected 2D Gaussian, normalised to unit integral in pixels.

The opacity is `alpha = 1 − exp(−tau)`. Summed over the screen, the optical
depth equals `kappa · V · |det J|` whatever the splat's shape or LOD level,
and a test checks this. The colour comes from the temperature colormap and
is weighted by opacity. The transfer function is two 256-entry tables
(extinction vs log ρ, colour vs log T) plus a phase mask, so it can be edited
every frame.

Stored covariances are moment matched, so splats are widened by
`sigmaScale` (default 1.8) when drawn.

**Stereo for row-interleaved walls.** `HalfHeightEyeCamera` renders each eye
at half height and samples exactly the rows that eye will display.
`InterleaveRows` then merges the two eyes into one image.

## Performance

RTX PRO 6000 Blackwell Max-Q, CUDA 13.4. One GPU's share of the wall:
9600×1890 pixels, interleaved stereo (two eyes at 9600×945), synthetic disk,
best of 20 frames:

| Records on GPU | Splats drawn (both eyes) | Frame | fps |
|---|---|---|---|
| 10 M | 3.3 M | 5.1 ms | 197 |
| 30 M | 9.9 M | 15.9 ms | 63 |
| 30 M, edge-on | 12.5 M | 21.1 ms | 47 |
| 100 M | 33 M | 54.4 ms | 18 |

Cost is about 0.5 ms per million records, split roughly 35% preprocess, 20%
sort and 45% rasterization. At 60 Hz one GPU handles about 30 M records on
the device, of which about 10 M are drawn across both eyes.

## Known limitations and next steps

* **Faint sub-pixel splats are dropped** when their peak opacity is below
  1/255, even though many of them together would be visible. The LOD cut
  avoids this by keeping drawn splats about 1.5 px across; drawing leaves
  only (`--no-lod`) at low resolution loses them.
* **Preprocessing reads every record once per eye.** Sharing one LOD and cull
  pass between the eyes, and skipping whole spatial chunks of records that
  are out of view or at the wrong LOD level, would remove most of the
  preprocess cost.
* **Records are 64 bytes.** Quantising positions within chunks, storing the
  covariance as half floats, and storing scalars as 16 bits would bring them
  to about 24 bytes, halving memory and preprocess bandwidth.
* **Interop is not written yet.** `RenderToDevice` returns a device pointer to
  float4 pixels; the Unity plugin still needs the D3D11 interop on top of it.
* **Splats are sorted by center depth,** so overlapping splats can pop when
  their order changes, as in standard 3D Gaussian splatting.
