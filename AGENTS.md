# Repository Guidelines

## Project Structure & Module Organization
The amrVolumeRenderer suite lives under `DirectSend/` (MPI compositor core), `Common/` (rendering primitives, image types, utilities), `ViskoresVolumeRenderer/` (volume driver miniapp), and the `AMReX/` submodule required for AMR inputs. CMake helpers live in `CMake/`, while generated binaries and intermediates stay inside `build/`. Tests reside in `Common/Testing/` alongside the components they cover. Keep new assets or sample frames in their own subtree to avoid polluting `build/`.

## Build, Test, and Development Commands
Configure once with `cmake -S . -B build -DAMRVOLUMERENDERER_ENABLE_TESTING=ON`. Build using `cmake --build build --target all -j`. Run the sample compositor via `mpirun -np 4 build/bin/volume_renderer --plotfile=plt0010 --width=256 --height=256`. Execute the regression suite with `ctest --test-dir build -V`. Use `cmake --build build --target install` only when you need staged artifacts; clean by deleting `build/`.

## Coding Style & Naming Conventions
C++ sources follow the repository `.clang-format`, which extends the Google style (two-space indents, wrapped argument lists). Run `clang-format -i DirectSend/Base/*.cpp` before committing. Favor CamelCase for classes and function-style verbs starting with uppercase (e.g., `PostReceives`). Constants use `kName` or `ALL_CAPS`; stick to existing patterns nearby. Keep headers self-contained and prefer `#include <Common/...>` style paths for shared components.

## Testing Guidelines
Unit tests target `Common` images and compositing; add new suites under the relevant module’s `Testing/` directory with filenames ending in `Test.cpp`. After introducing MPI behavior, add lightweight image fixtures to keep `ctest` runtimes manageable. Always enable tests during configuration and run `ctest --test-dir build` before opening a PR; document any required MPI launcher flags in the test README if they differ.

## Commit & Pull Request Guidelines
Commit history favors short, imperative summaries (`add computeTightBounds`, `remove debug logs`). Group logical changes together and include detail in the body when touching MPI synchronization or data layouts. Pull requests should cite related issues, describe the rendering path affected, and attach screenshots or checksum diffs for visual changes. Highlight testing performed (`ctest`, custom MPI run) so reviewers can reproduce.
