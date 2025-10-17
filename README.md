# amrVolumeRenderer

amrVolumeRenderer is a parallel volume renderer for
block-structured AMR data written with AMReX. The system runs efficiently on
distributed-memory platforms using MPI, consumes native AMReX plotfiles, and
exposes its full rendering pipeline through a scriptable Python API for
offline batch jobs, interactive exploration, and automated regression testing.

## Highlights

- Scalable DirectSend compositing path tuned for AMR workloads and large node counts.
- Zero-copy ingestion of AMReX plotfiles, including multi-level refinement control.
- Hybrid C++/Python driver model: drive renders from MPI-enabled Python scripts or native binaries.
- Deterministic outputs with shared image formats for reproducible workflows.
- Optional regression suite to validate compositing and sampling logic on each change.

## Quick Start

Install Viskores first; the renderer relies on its libraries being available when
you configure the project. With Spack:

```sh
spack install viskores
```

Activate the package in your environment (for example, `spack load viskores`)
before running CMake or provide the install prefix via `Viskores_DIR`.

1. Clone the repository:

   ```sh
   git clone https://github.com/<your-org>/amrVolumeRenderer.git
   cd amrVolumeRenderer
   ```

   CMake will fetch AMReX, nanobind, and other dependencies during the configure step—no submodules required.

2. Configure and build out-of-source:

   ```sh
   cmake -S . -B build -DAMRVOLUMERENDERER_ENABLE_TESTING=ON
   cmake --build build --target all -j
   ```

   Requirements: CMake 3.3+, a C++11 (or newer) compiler, Viskores, and an MPI
   implementation such as OpenMPI or MPICH.

3. Launch the reference DirectSend compositor (replace `plt0010` with your AMReX plotfile):

   ```sh
   mpirun -np 4 build/bin/DirectSendBase --input=plt0010 --width=512 --height=512
   ```

   Pass `--help` to list all rendering and AMR selection options.

## Python API

amrVolumeRenderer builds a `nanobind`-based extension named `amrVolumeRenderer_ext`
and ships a thin Python package under `python/`. After configuring with CMake,
install the package into your environment:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e python
```

Render directly from Python (use `mpirun` for multi-rank jobs):

```python
from amrVolumeRenderer import render

render(
    input_path="plt0010",
    output_path="volume.ppm",
    width=1024,
    height=1024,
    min_level=0,
    max_level=-1,
    antialiasing=4,
    box_transparency=0.15,
    visibility_graph=True,
)
```

Keyword arguments mirror the CLI flags exposed by `DirectSendBase` and
`ViskoresVolumeRenderer`. The module initializes and finalizes MPI/AMReX on
demand when run inside a Python interpreter.

## Testing

Keep tests enabled during configuration (`-DAMRVOLUMERENDERER_ENABLE_TESTING=ON`)
and execute the regression suite regularly:

```sh
ctest --test-dir build -V
```

The suite exercises the compositor across representative MPI matrices while
keeping image sizes small for fast turnaround.

## Repository Layout

- `DirectSend/` MPI compositor implementations and supporting utilities.
- `Common/` Rendering primitives, image buffers, and reusable utilities (with tests under `Common/Testing/`).
- `ViskoresVolumeRenderer/` CLI driver that exercises the production pipeline.
- `python/` Scriptable bindings built atop the C++ core.
- External dependencies (AMReX, nanobind) are populated under `build/_deps/` when configuring via CMake.
- `CMake/` Build-time helpers and shared CMake modules.

## Acknowledgements

The DirectSend compositor implementation and benchmarking harnesses draw on prior
work contributed to the miniGraphics repository (https://github.com/sandialabs/miniGraphics).
We thank Kenneth Moreland for making this implementation available.

## License

amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
See `LICENSE.txt` for details.

Copyright (C) 2025 Ben Wibking.

Copyright (c) 2017
National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.
