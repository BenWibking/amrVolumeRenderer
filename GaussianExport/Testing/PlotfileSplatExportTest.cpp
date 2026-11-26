// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// MPI test of the plotfile splat exporter. Writes a small two-level plotfile
// whose fine boxes are deliberately misaligned with the partition regions so
// that the boundary-exchange path is exercised, then checks conservation,
// exact LOD tiling, independence from the partition depth, and the file
// writer.

#include <AMReX.H>
#include <AMReX_Geometry.H>
#include <AMReX_MultiFab.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <mpi.h>

#include <GaussianExport/PlotfileSplatExporter.hpp>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

using namespace amrVolumeRenderer::splat;

#define CHECK(condition)                            \
  do {                                              \
    if (!(condition)) {                             \
      std::ostringstream message_;                  \
      message_ << __FILE__ << ':' << __LINE__       \
               << ": check failed: " << #condition; \
      throw std::runtime_error(message_.str());     \
    }                                               \
  } while (0)

constexpr int kCoarseCells = 32;
// Refined region in coarse indices (inclusive). Not aligned to the partition
// regions, so some regions mix both levels.
constexpr int kFineLo = 9;
constexpr int kFineHi = 22;
constexpr double kPhaseEdge = 5.5;

double DensityAt(double x, double y, double z) {
  const double r2 =
      (x - 0.5) * (x - 0.5) + (y - 0.45) * (y - 0.45) + (z - 0.5) * (z - 0.5);
  return 1e-24 * (1.0 + 50.0 * std::exp(-r2 / 0.01));
}

double TemperatureAt(double x, double /*y*/, double /*z*/) {
  return x > 0.6 ? 1e7 : 1e4;
}

void FillLevel(amrex::MultiFab& mf, double dx) {
  for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) {
    const amrex::Box box = mfi.validbox();
    auto arr = mf.array(mfi);
    // Host loop: the test runs on CPU data.
    for (int k = box.smallEnd(2); k <= box.bigEnd(2); ++k) {
      for (int j = box.smallEnd(1); j <= box.bigEnd(1); ++j) {
        for (int i = box.smallEnd(0); i <= box.bigEnd(0); ++i) {
          const double x = (i + 0.5) * dx;
          const double y = (j + 0.5) * dx;
          const double z = (k + 0.5) * dx;
          arr(i, j, k, 0) = DensityAt(x, y, z);
          arr(i, j, k, 1) = TemperatureAt(x, y, z);
        }
      }
    }
  }
}

void WriteTestPlotfile(const std::string& name) {
  amrex::RealBox realBox({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
  amrex::Array<int, 3> periodic = {0, 0, 0};
  const amrex::Box coarseDomain(amrex::IntVect(0),
                                amrex::IntVect(kCoarseCells - 1));
  const amrex::Box fineDomain = amrex::refine(coarseDomain, 2);
  amrex::Geometry coarseGeom(coarseDomain, realBox, 0, periodic);
  amrex::Geometry fineGeom(fineDomain, realBox, 0, periodic);

  amrex::BoxArray coarseBoxes(coarseDomain);
  coarseBoxes.maxSize(16);
  amrex::BoxArray fineBoxes(amrex::refine(
      amrex::Box(amrex::IntVect(kFineLo), amrex::IntVect(kFineHi)), 2));
  fineBoxes.maxSize(10);

  amrex::MultiFab coarse(coarseBoxes,
                         amrex::DistributionMapping(coarseBoxes),
                         2,
                         0,
                         amrex::MFInfo().SetArena(amrex::The_Cpu_Arena()));
  amrex::MultiFab fine(fineBoxes,
                       amrex::DistributionMapping(fineBoxes),
                       2,
                       0,
                       amrex::MFInfo().SetArena(amrex::The_Cpu_Arena()));
  FillLevel(coarse, 1.0 / kCoarseCells);
  FillLevel(fine, 0.5 / kCoarseCells);

  amrex::WriteMultiLevelPlotfile(name,
                                 2,
                                 {&coarse, &fine},
                                 {"gasDensity", "temperature"},
                                 {coarseGeom, fineGeom},
                                 0.0,
                                 {0, 0},
                                 {amrex::IntVect(2)});
}

struct TestCell {
  int ix, iy, iz, depth, phase;
};

std::vector<TestCell> ExpectedCells() {
  std::vector<TestCell> cells;
  auto phaseAt = [](double x) {
    return std::log10(TemperatureAt(x, 0.0, 0.0)) > kPhaseEdge ? 1 : 0;
  };
  for (int k = 0; k < kCoarseCells; ++k)
    for (int j = 0; j < kCoarseCells; ++j)
      for (int i = 0; i < kCoarseCells; ++i) {
        const bool covered = i >= kFineLo && i <= kFineHi && j >= kFineLo &&
                             j <= kFineHi && k >= kFineLo && k <= kFineHi;
        if (!covered) {
          cells.push_back({i, j, k, 1, phaseAt((i + 0.5) / kCoarseCells)});
        }
      }
  for (int k = 2 * kFineLo; k <= 2 * kFineHi + 1; ++k)
    for (int j = 2 * kFineLo; j <= 2 * kFineHi + 1; ++j)
      for (int i = 2 * kFineLo; i <= 2 * kFineHi + 1; ++i)
        cells.push_back({i, j, k, 0, phaseAt((i + 0.5) / (2 * kCoarseCells))});
  return cells;
}

std::vector<SplatRecord> GatherRecords(const std::vector<SplatRecord>& local) {
  MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
  const int nprocs = amrex::ParallelDescriptor::NProcs();
  int count = static_cast<int>(local.size() * sizeof(SplatRecord));
  std::vector<int> counts(nprocs), displs(nprocs);
  MPI_Gather(&count, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, comm);
  int total = 0;
  for (int r = 0; r < nprocs; ++r) {
    displs[r] = total;
    total += counts[r];
  }
  std::vector<SplatRecord> all(static_cast<std::size_t>(total) /
                               sizeof(SplatRecord));
  MPI_Gatherv(local.data(),
              count,
              MPI_BYTE,
              all.data(),
              counts.data(),
              displs.data(),
              MPI_BYTE,
              0,
              comm);
  return all;
}

using RecordKey = std::tuple<int, int, int, int, int>;

RecordKey KeyOf(const SplatRecord& r) {
  return {r.phase, r.depth, r.ix, r.iy, r.iz};
}

void CheckTiling(const std::vector<SplatRecord>& records,
                 const std::vector<TestCell>& cells,
                 const OctreeGeometry& geometry) {
  const std::vector<std::array<double, 3>> eyes = {
      {0.0, 0.0, 3.0}, {0.1, -0.05, 0.0}, {-2.0, 0.3, 0.1}};
  for (const auto& eye : eyes) {
    for (const double threshold : {1e-3, 0.02, 0.1, 0.5, 10.0}) {
      std::set<RecordKey> selected;
      for (const SplatRecord& r : records) {
        if (IsSelected(r, geometry, eye, threshold)) {
          selected.insert(KeyOf(r));
        }
      }
      for (const TestCell& c : cells) {
        int hits = 0;
        for (int depth = c.depth; depth <= geometry.topDepth; ++depth) {
          const int shift = depth - c.depth;
          hits += static_cast<int>(selected.count(
              {c.phase, depth, c.ix >> shift, c.iy >> shift, c.iz >> shift}));
        }
        if (hits != 1) {
          std::ostringstream message;
          message << "cell (" << c.ix << ',' << c.iy << ',' << c.iz
                  << ") depth " << c.depth << " covered " << hits
                  << " times (threshold " << threshold << ")";
          throw std::runtime_error(message.str());
        }
      }
    }
  }
}

/// Run `checks` on rank 0 only and make the outcome collective, so that a
/// failure throws on every rank instead of leaving the others in a collective.
template <typename Checks>
void CheckOnRankZero(Checks&& checks) {
  int failed = 0;
  std::string message = "check failed on rank 0";
  if (amrex::ParallelDescriptor::MyProc() == 0) {
    try {
      checks();
    } catch (const std::exception& error) {
      failed = 1;
      message = error.what();
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,
                &failed,
                1,
                MPI_INT,
                MPI_MAX,
                amrex::ParallelDescriptor::Communicator());
  if (failed) {
    throw std::runtime_error(message);
  }
}

void RunTest() {
  const int rank = amrex::ParallelDescriptor::MyProc();
  const std::string plotfile = "splat_export_test_plt";
  const std::string splatFile = "splat_export_test.splat";
  if (rank == 0) {
    // Leftovers from an earlier failed run.
    std::filesystem::remove(splatFile);
    std::filesystem::remove_all(plotfile);
    std::filesystem::remove_all(plotfile + ".old");
  }
  amrex::ParallelDescriptor::Barrier();
  WriteTestPlotfile(plotfile);
  amrex::ParallelDescriptor::Barrier();

  ExportOptions options;
  options.plotfilePath = plotfile;
  options.phaseEdges = {kPhaseEdge};
  options.lengthUnitCm = 1.0;
  options.lengthUnitName = "cm";
  options.toleranceLogDensity = 0.05f;
  options.toleranceLogTemperature = 0.05f;
  options.partitionDepth = 2;

  ExportResult result = ExportPlotfileSplats(options);
  const ExportStatistics& stats = result.statistics;
  const auto cells = ExpectedCells();
  CHECK(result.geometry.topDepth == 6);
  CHECK(stats.cellsExported == cells.size());
  CHECK(stats.mixedLeavesExchanged > 0);
  CHECK(std::abs(stats.cellVolume - 1.0) < 1e-12);
  CHECK(std::abs(stats.rootVolume - 1.0) < 1e-12);
  CHECK(std::abs(stats.rootWeight - stats.cellWeight) <=
        1e-12 * stats.cellWeight);
  CHECK(stats.records < stats.cellsExported);

  // The partition depth must not change the hierarchy.
  ExportOptions unpartitioned = options;
  unpartitioned.partitionDepth = 6;
  ExportResult reference = ExportPlotfileSplats(unpartitioned);
  CHECK(reference.statistics.records == stats.records);

  const auto all = GatherRecords(result.localRecords);
  const auto allReference = GatherRecords(reference.localRecords);
  CheckOnRankZero([&] {
    CHECK(all.size() == stats.records);
    std::set<RecordKey> keys;
    for (const SplatRecord& r : all) {
      CHECK(keys.insert(KeyOf(r)).second);
    }
    std::set<RecordKey> referenceKeys;
    for (const SplatRecord& r : allReference) {
      referenceKeys.insert(KeyOf(r));
    }
    CHECK(keys == referenceKeys);
    CheckTiling(all, cells, result.geometry);
  });

  WriteSplatFile(splatFile, options, result);
  amrex::ParallelDescriptor::Barrier();
  CheckOnRankZero([&] {
    const auto size = std::filesystem::file_size(splatFile);
    CHECK(size == kSplatFileHeaderBytes + stats.records * sizeof(SplatRecord));
    std::ifstream in(splatFile, std::ios::binary);
    std::string header(kSplatFileHeaderBytes, '\0');
    in.read(header.data(), static_cast<std::streamsize>(header.size()));
    CHECK(header.find("\"recordCount\": " + std::to_string(stats.records)) !=
          std::string::npos);
    std::vector<SplatRecord> fromFile(stats.records);
    in.read(
        reinterpret_cast<char*>(fromFile.data()),
        static_cast<std::streamsize>(fromFile.size() * sizeof(SplatRecord)));
    std::set<RecordKey> fileKeys;
    for (const SplatRecord& r : fromFile) {
      fileKeys.insert(KeyOf(r));
    }
    CHECK(fileKeys.size() == stats.records);
    std::filesystem::remove(splatFile);
    std::filesystem::remove_all(plotfile);
  });
}

}  // namespace

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  amrex::Initialize(argc, argv, false, MPI_COMM_WORLD);
  int status = 0;
  try {
    RunTest();
  } catch (const std::exception& error) {
    std::cerr << "PlotfileSplatExportTest failed on rank "
              << amrex::ParallelDescriptor::MyProc() << ": " << error.what()
              << std::endl;
    status = 1;
  }
  MPI_Allreduce(MPI_IN_PLACE, &status, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (status == 0 && amrex::ParallelDescriptor::MyProc() == 0) {
    std::cout << "PlotfileSplatExportTest passed" << std::endl;
  }
  amrex::Finalize();
  MPI_Finalize();
  return status;
}
