# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build System

**Standard Build Commands:**
```bash
# Full project build
mkdir miniGraphics-build
cd miniGraphics-build
cmake ../miniGraphics
make -j

# Individual miniapp build
mkdir build-algorithm
cd build-algorithm
cmake ../miniGraphics/AlgorithmName/Variant
make -j
```

**CMake Options:**
- `MINIGRAPHICS_ENABLE_TESTING=ON/OFF` (default: ON) - Build test suite
- `MINIGRAPHICS_ENABLE_ICET=ON/OFF` (default: ON) - Include IceT miniapp
- `MINIGRAPHICS_ENABLE_OPENGL=ON/OFF` (default: OFF) - OpenGL rendering support

**Testing:**
```bash
# Run all tests
ctest

# Tests are automatically generated for all combinations of:
# - Color buffers: --color-ubyte, --color-float
# - Depth buffers: --depth-float, --depth-none  
# - Image compression: --enable-image-compress, --disable-image-compress
```

## Architecture

**Core Concept:** miniGraphics demonstrates parallel sort-last rendering algorithms. Each miniapp implements a different image compositing strategy for combining multiple rendered images into a single result.

**Project Structure:**
- **Common/** Shared utilities (image classes, compositor base, YAML helpers, mesh handling)
- **DirectSend/Base/** Direct-send compositor used by the sample application
- **ViskoresVolumeExample/** Miniapp that drives the compositor using Viskores for volume rendering
- **ThirdParty/** External dependencies (GLM, optionparser, IceT)
- **Reference/** Sample imagery and configurations
- **Utilites/** Small helper scripts

**Key Dependencies:**
- MPI (required) - parallel operations
- C++11 compiler (required)
- OpenGL/GLEW/GLFW (optional) - hardware rendering

**Execution Flow:** Miniapp → Viskores volume renderer → DirectSend compositor
- The miniapp constructs volume geometry and camera parameters
- Viskores renders per-rank images
- The compositor combines images from multiple processes

**Adding New Algorithms:** Follow existing pattern - create directory with CMakeLists.txt, implement compositor class inheriting from base, use `miniGraphics_executable()` macro for consistent build configuration.

## Development Notes

**Build Artifacts:**
- Executables: `${CMAKE_BINARY_DIR}/bin/`
- Libraries: `${CMAKE_BINARY_DIR}/lib/`

**Code Style:** Each miniapp follows consistent pattern with algorithm-specific compositor implementation and shared infrastructure for image handling, timing, and output generation.
