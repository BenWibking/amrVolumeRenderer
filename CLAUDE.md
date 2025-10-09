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
- **Algorithm Directories:** Each top-level directory (BinarySwap/, DirectSend/, 2-3-Swap/, RadixK/, IceT/) contains variants of a specific compositing algorithm
- **Common/:** Shared utilities (Image classes, Compositor base, Mesh handling, file I/O)
- **Paint/:** Rendering backends (PainterViskores for distributed rendering)
- **ThirdParty/:** External dependencies (GLM, optionparser, IceT)

**Key Dependencies:**
- MPI (required) - parallel operations
- C++11 compiler (required)
- OpenGL/GLEW/GLFW (optional) - hardware rendering

**Execution Flow:** MainLoop → Compositor → Painter
- Painter renders local geometry
- Compositor combines images from multiple processes using algorithm-specific strategy
- MainLoop coordinates overall execution and output

**Adding New Algorithms:** Follow existing pattern - create directory with CMakeLists.txt, implement compositor class inheriting from base, use `miniGraphics_executable()` macro for consistent build configuration.

## Development Notes

**Build Artifacts:**
- Executables: `${CMAKE_BINARY_DIR}/bin/`
- Libraries: `${CMAKE_BINARY_DIR}/lib/`

**Code Style:** Each miniapp follows consistent pattern with algorithm-specific compositor implementation and shared infrastructure for image handling, timing, and output generation.
