// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// gaussian_export: convert an AMReX plotfile into a level-of-detail hierarchy
// of Gaussian splats (see README.md in this directory).

#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <mpi.h>

#include <array>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "PlotfileSplatExporter.hpp"

namespace {

using amrVolumeRenderer::splat::ExportOptions;

void PrintUsage() {
  std::cout <<
      R"(Usage: gaussian_export --plotfile PATH [options]

Builds a level-of-detail octree of Gaussian splats from density and
temperature and writes it to a single binary file (JSON header + 64-byte
records). Without --output only statistics are printed.

Options:
  --plotfile PATH            AMReX plotfile to read (required)
  --output FILE              Output splat file
  --density NAME             Density variable (default: gasDensity)
  --temperature NAME         Temperature variable (default: temperature)
  --phase-edges A,B,...      log10(T) edges between temperature phases
                             (default: 4.5,5.5,6.5; empty string = one phase)
  --tol-log-density X        Max log10(density) range of a collapsed node
                             (default: 0.1)
  --tol-log-temperature X    Max log10(T) range of a collapsed node
                             (default: 0.1)
  --min-depth N              Do not emit records finer than octree depth N
                             (depth 0 = finest cells; default: 0)
  --partition-depth N        Depth of regions reduced without communication
                             (default: 5, i.e. 32^3 finest cells)
  --max-level N              Finest AMR level to read (default: all)
  --length-unit-cm X         Output length unit in cm (default: 1 kpc)
  --length-unit-name NAME    Name of the output length unit (default: kpc)
  --center X Y Z             Output origin in plotfile coordinates (cm)
                             (default: domain center)
  --region XLO YLO ZLO XHI YHI ZHI
                             Only export cells inside this box (output units,
                             relative to the center)
  --help                     Show this message
)";
}

std::vector<double> ParseList(const std::string& text) {
  std::vector<double> values;
  std::stringstream stream(text);
  std::string item;
  while (std::getline(stream, item, ',')) {
    if (!item.empty()) {
      values.push_back(std::stod(item));
    }
  }
  return values;
}

/// Returns false if the program should exit without exporting.
bool ParseArguments(int argc,
                    char** argv,
                    ExportOptions& options,
                    std::string& outputPath) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto next = [&](int count = 1) {
      if (i + count >= argc) {
        throw std::invalid_argument(arg + " requires " + std::to_string(count) +
                                    " value(s)");
      }
      return std::string(argv[++i]);
    };
    if (arg == "--help" || arg == "-h") {
      return false;
    } else if (arg == "--plotfile") {
      options.plotfilePath = next();
    } else if (arg == "--output") {
      outputPath = next();
    } else if (arg == "--density") {
      options.densityVariable = next();
    } else if (arg == "--temperature") {
      options.temperatureVariable = next();
    } else if (arg == "--phase-edges") {
      options.phaseEdges = ParseList(next());
    } else if (arg == "--tol-log-density") {
      options.toleranceLogDensity = std::stof(next());
    } else if (arg == "--tol-log-temperature") {
      options.toleranceLogTemperature = std::stof(next());
    } else if (arg == "--min-depth") {
      options.minDepth = std::stoi(next());
    } else if (arg == "--partition-depth") {
      options.partitionDepth = std::stoi(next());
    } else if (arg == "--max-level") {
      options.maxLevel = std::stoi(next());
    } else if (arg == "--length-unit-cm") {
      options.lengthUnitCm = std::stod(next());
    } else if (arg == "--length-unit-name") {
      options.lengthUnitName = next();
    } else if (arg == "--center") {
      std::array<double, 3> center;
      for (double& c : center) {
        c = std::stod(next());
      }
      options.centerCm = center;
    } else if (arg == "--region") {
      std::array<double, 6> region;
      for (double& r : region) {
        r = std::stod(next());
      }
      options.region = region;
    } else {
      throw std::invalid_argument("Unknown option '" + arg + "'");
    }
  }
  if (options.plotfilePath.empty()) {
    throw std::invalid_argument("--plotfile is required");
  }
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  amrex::Initialize(argc, argv, false, MPI_COMM_WORLD);
  const int rank = amrex::ParallelDescriptor::MyProc();

  int exitCode = 0;
  try {
    ExportOptions options;
    std::string outputPath;
    if (!ParseArguments(argc, argv, options, outputPath)) {
      if (rank == 0) {
        PrintUsage();
      }
    } else {
      auto result = amrVolumeRenderer::splat::ExportPlotfileSplats(options);
      if (!outputPath.empty()) {
        amrVolumeRenderer::splat::WriteSplatFile(outputPath, options, result);
      }
      if (rank == 0) {
        amrVolumeRenderer::splat::PrintExportSummary(options, result);
        if (!outputPath.empty()) {
          std::cout << "Wrote " << result.statistics.records << " records to "
                    << outputPath << std::endl;
        }
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "Error on rank " << rank << ": " << error.what() << std::endl;
    if (rank == 0) {
      std::cerr << "Run with --help for usage." << std::endl;
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  amrex::Finalize();
  MPI_Finalize();
  return exitCode;
}
