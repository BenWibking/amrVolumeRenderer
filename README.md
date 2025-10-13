# miniGraphics

miniGraphics demonstrates sort-last parallel rendering in an MPI environment.
The miniapp focuses on the DirectSend compositor while sharing rendering
utilities across a small family of miniapps. DirectSend implements a
straightforward image compositing path where each rank sends its rendered
image directly to a designated compositor; variants exercise different
communication overlap strategies.

## Build

Prerequisites:

  * CMake 3.3+
  * A C++11 compliant compiler
  * MPI implementation and launcher (e.g., OpenMPI, MPICH)

Configure and build in an out-of-source tree:

```sh
cmake -S . -B build -DMINIGRAPHICS_ENABLE_TESTING=ON
cmake --build build --target all -j
```

Re-run the `cmake --build` command after making source changes. Generated
artifacts stay under `build/`, which can be removed safely when starting fresh.

## Run

Launch the reference DirectSend compositor locally with four MPI ranks:

```sh
mpirun -np 4 build/bin/DirectSendBase --width=256 --height=256
```

The executable accepts additional options for image dimensions and scene
selection; inspect `--help` for details.

## Test

Enable testing during configuration (see Build) and run the regression suite:

```sh
ctest --test-dir build -V
```

The suite drives the MPI matrix defined in `CTestTestfile.cmake`; expect small
image sizes to minimize runtime.

## Repository Layout

  * `DirectSend/` DirectSend compositor implementations and shared utilities.
    - `DirectSend/Base/` Primary compositing path used by the sample app.
  * `Common/` Shared rendering primitives, image buffers, and utilities.
  * `ViskoresVolumeRenderer/` Example miniapp that drives the compositor with Viskores.
  * `Reference/` Sample configurations and reference imagery.
  * `ThirdParty/` Vendored headers (GLM, IceT, optionparser).
  * `CMake/` Auxiliary build scripts and macros.
  * `Utilites/` Miscellaneous developer scripts.

## License

miniGraphics is distributed under the OSI-approved BSD 3-clause License.
See `LICENSE.txt` for details.

Copyright (c) 2017
National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

SDR# 2261
