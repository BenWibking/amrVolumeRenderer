# Gaussian splat export

`gaussian_export` converts an AMReX plotfile (density and temperature) into a
level-of-detail (LOD) hierarchy of 3D Gaussians. The output is one binary file
that a GPU splat renderer can load as a flat array. The renderer chooses a
per-view LOD cut from each record's octree key alone, so it needs no tree
pointers.

The tool is written for large static-refinement runs such as the Quokka
`DiskGalaxy` problem: about 200 billion cells and 20 TB. It reads only the two
requested variables, streams one box at a time, and exchanges only small
summaries over MPI.

## Usage

```bash
mpirun -np 1024 build/bin/gaussian_export \
  --plotfile plt0100000 --output disk.splat \
  --tol-log-density 0.3 --tol-log-temperature 0.3
```

Without `--output` the tool only prints statistics. Use that to tune the
tolerances on a subregion (`--region xlo ylo zlo xhi yhi zhi`, in output units
relative to the domain center) before running on the full domain. Run
`gaussian_export --help` for all options. `python/examples/inspect_splats.py`
reads a file, prints a summary, checks the LOD rule for a viewpoint, and
renders a quick projection.

## Algorithm

* **One octree in finest-level index space.** A node at depth `d` covers `2^d`
  finest cells per axis and has integer key `(finest index) >> d`. A cell on
  AMR level `l` enters as a leaf at depth `d_l = log2(refinement ratio from l
  to the finest level)`. Cells covered by a finer level are skipped, because
  the finer level represents them. Refinement ratios must be powers of two and
  cells must be cubic.
* **Temperature phases.** Cells are binned by `log10 T` (`--phase-edges`,
  default `4.5,5.5,6.5`). Each phase forms its own forest, so hot and cold gas
  are never averaged into the same splat.
* **Aggregation.** Each node stores its mass (density × volume), its
  mass-weighted centroid and covariance, its occupied volume, and the
  mass-weighted mean `log10 T`. Moments are combined with the parallel-axis
  rule, so mass is conserved exactly up to round-off. A leaf cell of side `a`
  has variance `a²/12` per axis, which is moment matched: a full uniform cube
  of side `a` at any depth gets the same value.
* **Collapse.** A node is *homogeneous* when its phase fills it completely and
  the ranges of `log10 ρ` and `log10 T` inside it are within the tolerances.
  The descendants of a homogeneous node are never written.
  `--min-depth N` additionally drops every record finer than depth `N`.
* **Parallel decomposition.**
  * The tree is cut into regions at `--partition-depth` (default 5, i.e. 32³
    finest cells). With `blocking_factor = 32`, each region lies inside one
    box, so the rank that reads the box reduces the region without
    communication.
  * Regions that straddle boxes or AMR levels send their leaf cells to one
    owner rank first.
  * Above the partition depth, nodes are merged one depth at a time, with
    siblings routed to a hashed owner rank (`MPI_Alltoallv`).
  * The result does not depend on the partition depth or the rank count. The
    tests check this.

## File format (version 1)

The file starts with a 4096-byte header containing JSON padded with spaces.
After it come `recordCount` little-endian records of 64 bytes each:

| Field | Type | Meaning |
|---|---|---|
| `position` | f32×3 | mass-weighted centroid (output length units, origin at `centerCm`) |
| `covariance` | f32×6 | xx, xy, xz, yy, yz, zz (units²), moment matched |
| `volume` | f32 | occupied volume (units³) |
| `logDensity` | f32 | log10 of volume-averaged density (plotfile units) |
| `logTemperature` | f32 | mass-weighted mean log10 T |
| `key` | i32×3 | octree coordinates at `depth` |
| `depth` | u8 | octree depth (0 = finest cells, `topDepth` = root) |
| `phase` | u8 | temperature phase |
| `flags` | u8 | 1 = leaf (no descendants stored), 2 = homogeneous, 4 = root |
| `reserved` | u8 | 0 |

The header also records the geometry (`domainLo`, `finestCellSize`,
`topDepth`, `lengthUnit`, `centerCm`) and every export parameter. Within each
rank's contiguous block, records are in Morton order of their octree cell. The
default output unit is kpc, so assets should be scaled to the scene in Unity
(getReal3D expects metre-scale assets).

## Runtime contract for a renderer

**Level-of-detail cut.** The node with key `k` and depth `d` is a cube of side
`s = finestCellSize · 2^d` with lower corner `domainLo + k·s`. Let `R` be the
radius of its bounding sphere and `c` its center, and define
`m(node) = R / (|eye − c| − R)`, with `m = ∞` if the eye is inside the sphere.
Draw a record if and only if

```
(leaf or m(self) <= t)  and  (root or m(parent) > t),   parent = (k >> 1, d + 1)
```

Child bounding spheres nest inside their parent's, so `m` never increases down
the tree, and every root-to-leaf path has exactly one drawn record. The drawn
records therefore tile the data with no gaps or overlaps. To choose `t` for a
target splat size `p` in pixels, use `t ≈ p / f`, where `f` is the focal
length in pixels.

On a multi-process wall, every process must evaluate the cut from the same
point (the tracked head position), not from its own projection. Otherwise
stripes disagree where they meet. Only frustum culling should differ between
processes.

**Splat footprint.** The stored covariance is moment matched. Drawing it
unchanged leaves a visible lattice pattern, so widen each Gaussian by about
`1.8×` in σ (multiply the covariance by about 3.2) when drawing.

**Opacity with an editable transfer function.** Let the transfer function
give an extinction coefficient `κ(ρ, T)` (per unit length) and a colour
`C(ρ, T)`. For a record with occupied volume `V` and projected 2D Gaussian
`N₂(x)`, normalised to unit integral over the screen-plane footprint in
world units:

```
α(x) = 1 − exp(−κ(ρ, T) · V · N₂(x))
```

For `κ ∝ ρ` this conserves the integrated optical depth across LOD levels,
in the optically thin limit.

## Tuning notes

From the low-resolution example `DiskGalaxy` plotfile (3.2 M cells, 6 levels):

| Settings | Cells per record |
|---|---|
| tolerances 0.1 dex | 7.6 |
| tolerances 0.3 dex | 12.5 |
| tolerances 0.5 dex | 23 |
| tolerances 0.3 dex, `--min-depth 1` | 36 |

Cells on phase boundaries never collapse, because no node of either phase is
completely filled. Compression on a production snapshot will differ, so
measure it on a representative subregion first.
