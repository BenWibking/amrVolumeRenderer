# Viskores Integration Example

This example demonstrates the integration of Viskores rendering capabilities with miniGraphics' parallel compositing framework.

## Overview

The ViskoresExample shows how to use the new `PainterViskores` class to perform rank-based distributed rendering where each MPI rank renders its own data using Viskores, and the results are then composited using miniGraphics' DirectSend algorithm.

## Features

- **Rank-based rendering**: Each MPI rank creates and renders its own mesh data using Viskores
- **Automatic fallback**: If Viskores is not available, falls back to the simple painter
- **Distributed visualization**: Demonstrates how Viskores can be used in a distributed parallel rendering context
- **Color-coded ranks**: Each rank renders with a unique color to visualize the distribution

## Building

The example will only be built if Viskores is found by CMake. To build with Viskores support:

```bash
mkdir build
cd build

# Configure with Viskores path
cmake -DViskores_DIR=/path/to/viskores/lib/cmake/viskores ../miniGraphics

# Build
make -j

# Run with MPI
mpirun -np 4 ./bin/ViskoresExample --width=800 --height=600
```

## Usage

The example accepts standard miniGraphics command line arguments:

- `--width=W`: Set image width (default: 1024)
- `--height=H`: Set image height (default: 768)
- `--trials=N`: Number of rendering trials for timing
- `--yaml-output=file`: Output timing data to YAML file

## Implementation Details

### Data Distribution
Each rank creates a simple box mesh positioned in a grid layout based on the rank number. The boxes are colored uniquely per rank using a hue-based color scheme.

### Rendering Pipeline
1. Each rank creates its local mesh data
2. The `PainterViskores` converts the miniGraphics mesh to a Viskores DataSet
3. Viskores renders the data locally on each rank
4. The DirectSend compositor combines all rank images into the final result

### Viskores Integration
The integration demonstrates:
- Converting miniGraphics mesh format to Viskores DataSet
- Setting up Viskores Scene, Actor, and View components
- Rendering with Viskores and converting back to miniGraphics image format
- Proper MPI rank information handling for distributed visualization

## Testing

The example is automatically included in the miniGraphics test suite when Viskores is available. Tests run with different image formats and compositing options.