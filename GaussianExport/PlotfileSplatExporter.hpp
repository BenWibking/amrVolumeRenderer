// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Distributed conversion of an AMReX plotfile (density + temperature) into a
// level-of-detail hierarchy of Gaussian splats. See GaussianExport/README.md
// for the algorithm and the output format.

#ifndef AMRVOLUMERENDERER_PLOTFILE_SPLAT_EXPORTER_HPP
#define AMRVOLUMERENDERER_PLOTFILE_SPLAT_EXPORTER_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "SplatOctree.hpp"

namespace amrVolumeRenderer {
namespace splat {

/// Size of the JSON header at the start of a splat file.
constexpr std::size_t kSplatFileHeaderBytes = 4096;

struct ExportOptions {
  std::string plotfilePath;
  std::string densityVariable = "gasDensity";
  std::string temperatureVariable = "temperature";
  /// Increasing log10(temperature) bin edges separating the phases.
  std::vector<double> phaseEdges = {4.5, 5.5, 6.5};
  float toleranceLogDensity = 0.1f;
  float toleranceLogTemperature = 0.1f;
  /// Records shallower than this octree depth are never emitted.
  int minDepth = 0;
  /// Octree depth of the regions that are reduced without communication.
  int partitionDepth = 5;
  /// Finest AMR level to read (-1 = finest in the plotfile).
  int maxLevel = -1;
  /// Output length unit in centimetres (default: kiloparsec).
  double lengthUnitCm = 3.0856775814913673e21;
  std::string lengthUnitName = "kpc";
  /// Origin of the output coordinates in plotfile (cm) coordinates. Defaults
  /// to the center of the domain.
  std::optional<std::array<double, 3>> centerCm;
  /// Optional subregion (output units, relative to the center):
  /// {xlo, ylo, zlo, xhi, yhi, zhi}. Only cells whose centers lie inside are
  /// exported.
  std::optional<std::array<double, 6>> region;
};

struct ExportStatistics {
  std::uint64_t cellsRead = 0;
  std::uint64_t cellsExported = 0;
  std::uint64_t cellsSkippedInvalid = 0;
  std::uint64_t mixedLeavesExchanged = 0;
  std::uint64_t records = 0;
  std::uint64_t leafRecords = 0;
  std::uint64_t homogeneousRecords = 0;
  std::vector<std::uint64_t> recordsPerDepth;
  std::vector<std::uint64_t> recordsPerPhase;
  /// Mass and volume entering as cells and leaving as roots. They agree to
  /// round-off if the hierarchy conserves mass.
  double cellWeight = 0.0;
  double rootWeight = 0.0;
  double cellVolume = 0.0;
  double rootVolume = 0.0;
  /// Wall-clock seconds (maximum over ranks).
  double readSeconds = 0.0;
  double localReduceSeconds = 0.0;
  double exchangeSeconds = 0.0;
  double upperTreeSeconds = 0.0;
};

struct ExportResult {
  /// Records owned by this rank (unordered).
  std::vector<SplatRecord> localRecords;
  OctreeGeometry geometry;
  OctreeParams params;
  /// Global statistics (identical on every rank).
  ExportStatistics statistics;
  std::array<double, 3> centerCm = {0.0, 0.0, 0.0};
  double time = 0.0;
  int maxLevel = 0;
};

/// Build the splat hierarchy. Collective over AMReX's communicator.
ExportResult ExportPlotfileSplats(const ExportOptions& options);

/// Header text written at the start of the output file.
std::string BuildSplatFileHeader(const ExportOptions& options,
                                 const ExportResult& result);

/// Write all ranks' records to a single file with MPI-IO. Collective. Sorts
/// each rank's records spatially before writing.
void WriteSplatFile(const std::string& path,
                    const ExportOptions& options,
                    ExportResult& result);

/// Print the statistics in human readable form (call on one rank).
void PrintExportSummary(const ExportOptions& options,
                        const ExportResult& result);

}  // namespace splat
}  // namespace amrVolumeRenderer

#endif  // AMRVOLUMERENDERER_PLOTFILE_SPLAT_EXPORTER_HPP
