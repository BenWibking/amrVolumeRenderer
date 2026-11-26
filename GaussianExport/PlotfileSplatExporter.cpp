// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.

#include "PlotfileSplatExporter.hpp"

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_GpuDevice.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_PlotFileUtil.H>
#include <AMReX_VisMF.H>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace amrVolumeRenderer {
namespace splat {

namespace {

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

/// One component of one FAB, copied to host memory.
struct HostComponent {
  amrex::Box box;
  std::vector<amrex::Real> data;

  amrex::Real operator()(int i, int j, int k) const {
    const amrex::IntVect lo = box.smallEnd();
    const amrex::IntVect len = box.length();
    const std::size_t offset = static_cast<std::size_t>(i - lo[0]) +
                               static_cast<std::size_t>(len[0]) *
                                   (static_cast<std::size_t>(j - lo[1]) +
                                    static_cast<std::size_t>(len[1]) *
                                        static_cast<std::size_t>(k - lo[2]));
    return data[offset];
  }
};

HostComponent ReadComponent(amrex::VisMF& vismf, int boxIndex, int component) {
  std::unique_ptr<amrex::FArrayBox> fab(vismf.readFAB(boxIndex, component));
  HostComponent result;
  result.box = fab->box();
  result.data.resize(static_cast<std::size_t>(result.box.numPts()));
  const std::size_t bytes = result.data.size() * sizeof(amrex::Real);
#ifdef AMREX_USE_GPU
  if (fab->arena()->isManaged() || fab->arena()->isDevice()) {
    amrex::Gpu::dtoh_memcpy(result.data.data(), fab->dataPtr(), bytes);
    return result;
  }
#endif
  std::memcpy(result.data.data(), fab->dataPtr(), bytes);
  return result;
}

/// RAII wrapper for an MPI datatype describing one Node.
class NodeDatatype {
 public:
  NodeDatatype() {
    MPI_Type_contiguous(static_cast<int>(sizeof(Node)), MPI_BYTE, &type_);
    MPI_Type_commit(&type_);
  }
  ~NodeDatatype() { MPI_Type_free(&type_); }
  NodeDatatype(const NodeDatatype&) = delete;
  NodeDatatype& operator=(const NodeDatatype&) = delete;
  MPI_Datatype get() const { return type_; }

 private:
  MPI_Datatype type_ = MPI_DATATYPE_NULL;
};

int CheckedInt(std::size_t value, const char* what) {
  if (value > static_cast<std::size_t>(INT_MAX)) {
    throw std::runtime_error(
        std::string("MPI count overflow while exchanging ") + what);
  }
  return static_cast<int>(value);
}

/// All-to-all exchange of nodes. Clears `outgoing`.
std::vector<Node> ExchangeNodes(std::vector<std::vector<Node>>& outgoing,
                                MPI_Comm comm,
                                MPI_Datatype nodeType) {
  const int nprocs = static_cast<int>(outgoing.size());
  std::vector<int> sendCounts(nprocs), sendDispls(nprocs);
  std::size_t totalSend = 0;
  for (int r = 0; r < nprocs; ++r) {
    sendCounts[r] = CheckedInt(outgoing[r].size(), "octree nodes");
    sendDispls[r] = CheckedInt(totalSend, "octree nodes");
    totalSend += outgoing[r].size();
  }
  std::vector<Node> sendBuffer;
  sendBuffer.reserve(totalSend);
  for (auto& bucket : outgoing) {
    sendBuffer.insert(sendBuffer.end(), bucket.begin(), bucket.end());
    std::vector<Node>().swap(bucket);
  }

  std::vector<int> recvCounts(nprocs), recvDispls(nprocs);
  MPI_Alltoall(
      sendCounts.data(), 1, MPI_INT, recvCounts.data(), 1, MPI_INT, comm);
  std::size_t totalRecv = 0;
  for (int r = 0; r < nprocs; ++r) {
    recvDispls[r] = CheckedInt(totalRecv, "octree nodes");
    totalRecv += static_cast<std::size_t>(recvCounts[r]);
  }
  CheckedInt(totalRecv, "octree nodes");
  std::vector<Node> recvBuffer(totalRecv);
  MPI_Alltoallv(sendBuffer.data(),
                sendCounts.data(),
                sendDispls.data(),
                nodeType,
                recvBuffer.data(),
                recvCounts.data(),
                recvDispls.data(),
                nodeType,
                comm);
  return recvBuffer;
}

std::uint64_t Mix64(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

/// Rank that merges the children of the node (ix, iy, iz) at `depth`.
int UpperTreeOwner(
    int depth, std::int32_t ix, std::int32_t iy, std::int32_t iz, int nprocs) {
  const std::uint64_t key = MortonCode(static_cast<std::uint32_t>(ix),
                                       static_cast<std::uint32_t>(iy),
                                       static_cast<std::uint32_t>(iz)) ^
                            (static_cast<std::uint64_t>(depth) << 59);
  return static_cast<int>(Mix64(key) % static_cast<std::uint64_t>(nprocs));
}

int Log2Exact(int value) {
  if (value <= 0 || (value & (value - 1)) != 0) {
    return -1;
  }
  int result = 0;
  while ((1 << result) < value) {
    ++result;
  }
  return result;
}

int CeilLog2(std::int64_t value) {
  int result = 0;
  while ((std::int64_t{1} << result) < value) {
    ++result;
  }
  return result;
}

void ValidateOptions(const ExportOptions& options) {
  if (options.plotfilePath.empty()) {
    throw std::invalid_argument("A plotfile path is required");
  }
  if (!std::is_sorted(options.phaseEdges.begin(), options.phaseEdges.end()) ||
      std::adjacent_find(options.phaseEdges.begin(),
                         options.phaseEdges.end()) !=
          options.phaseEdges.end()) {
    throw std::invalid_argument("Phase edges must be strictly increasing");
  }
  if (static_cast<int>(options.phaseEdges.size()) + 1 > kMaxPhases) {
    throw std::invalid_argument("At most " + std::to_string(kMaxPhases - 1) +
                                " phase edges are supported");
  }
  if (!(options.toleranceLogDensity >= 0.0f) ||
      !(options.toleranceLogTemperature >= 0.0f)) {
    throw std::invalid_argument("Tolerances must be non-negative");
  }
  if (!(options.lengthUnitCm > 0.0) || !std::isfinite(options.lengthUnitCm)) {
    throw std::invalid_argument("The length unit must be positive");
  }
  if (options.minDepth < 0 || options.partitionDepth < 0) {
    throw std::invalid_argument("Depths must be non-negative");
  }
  if (options.region) {
    const auto& r = *options.region;
    for (int axis = 0; axis < 3; ++axis) {
      if (!(r[axis] < r[axis + 3])) {
        throw std::invalid_argument(
            "Region lower corner must be below the "
            "upper corner");
      }
    }
  }
}

int FindComponent(const amrex::Vector<std::string>& names,
                  const std::string& name) {
  const auto it = std::find(names.begin(), names.end(), name);
  if (it == names.end()) {
    std::ostringstream message;
    message << "Variable '" << name << "' not found in plotfile. Available:";
    for (const auto& n : names) {
      message << ' ' << n;
    }
    throw std::runtime_error(message.str());
  }
  return static_cast<int>(it - names.begin());
}

void ReduceSum(std::uint64_t& value, MPI_Comm comm) {
  MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_UINT64_T, MPI_SUM, comm);
}

void ReduceSum(double& value, MPI_Comm comm) {
  MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_SUM, comm);
}

void ReduceMax(double& value, MPI_Comm comm) {
  MPI_Allreduce(MPI_IN_PLACE, &value, 1, MPI_DOUBLE, MPI_MAX, comm);
}

void ReduceSum(std::vector<std::uint64_t>& values, MPI_Comm comm) {
  MPI_Allreduce(MPI_IN_PLACE,
                values.data(),
                CheckedInt(values.size(), "statistics"),
                MPI_UINT64_T,
                MPI_SUM,
                comm);
}

std::string JsonEscape(const std::string& text) {
  std::string out;
  for (const char c : text) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out += ' ';
        } else {
          out += c;
        }
    }
  }
  return out;
}

template <typename Container>
void WriteJsonArray(std::ostream& os, const Container& values) {
  os << '[';
  bool first = true;
  for (const auto& v : values) {
    if (!first) {
      os << ", ";
    }
    os << v;
    first = false;
  }
  os << ']';
}

}  // namespace

ExportResult ExportPlotfileSplats(const ExportOptions& options) {
  ValidateOptions(options);

  MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
  const int rank = amrex::ParallelDescriptor::MyProc();
  const int nprocs = amrex::ParallelDescriptor::NProcs();
  const NodeDatatype nodeType;

  const auto readStart = Clock::now();
  amrex::PlotFileData plotfile(options.plotfilePath);
  if (plotfile.spaceDim() != 3) {
    throw std::runtime_error("Only 3D plotfiles are supported");
  }
  const int densityComp =
      FindComponent(plotfile.varNames(), options.densityVariable);
  const int temperatureComp =
      FindComponent(plotfile.varNames(), options.temperatureVariable);

  const int finestLevel = plotfile.finestLevel();
  const int maxLevel = (options.maxLevel < 0 || options.maxLevel > finestLevel)
                           ? finestLevel
                           : options.maxLevel;
  const int nLevels = maxLevel + 1;

  // Octree depth of the cells on each level.
  std::vector<int> levelDepth(nLevels, 0);
  for (int level = maxLevel - 1; level >= 0; --level) {
    const int ratio = plotfile.refRatio(level);
    const int log2Ratio = Log2Exact(ratio);
    if (log2Ratio < 1) {
      throw std::runtime_error("Refinement ratio " + std::to_string(ratio) +
                               " is not a power of two");
    }
    levelDepth[level] = levelDepth[level + 1] + log2Ratio;
  }
  for (int level = 0; level <= maxLevel; ++level) {
    const amrex::Box domain = plotfile.probDomain(level);
    if (domain.smallEnd() != amrex::IntVect(0)) {
      throw std::runtime_error("Problem domains must start at index 0");
    }
  }
  const amrex::Box finestDomain = plotfile.probDomain(maxLevel);
  const int topDepth =
      std::max(levelDepth[0], CeilLog2(finestDomain.longside()));
  if (topDepth > kMaxTopDepth) {
    throw std::runtime_error("The finest level has " +
                             std::to_string(finestDomain.longside()) +
                             " cells across, more than the supported 2^" +
                             std::to_string(kMaxTopDepth));
  }

  const auto dxFinest = plotfile.cellSize(maxLevel);
  for (int axis = 1; axis < 3; ++axis) {
    if (std::abs(dxFinest[axis] - dxFinest[0]) > 1e-6 * dxFinest[0]) {
      throw std::runtime_error(
          "Cells must be cubic (equal spacing on all axes)");
    }
  }
  const auto probLo = plotfile.probLo();
  const auto probHi = plotfile.probHi();
  std::array<double, 3> center;
  for (int axis = 0; axis < 3; ++axis) {
    center[axis] = options.centerCm ? (*options.centerCm)[axis]
                                    : 0.5 * (static_cast<double>(probLo[axis]) +
                                             static_cast<double>(probHi[axis]));
  }
  const double toOutput = 1.0 / options.lengthUnitCm;

  ExportResult result;
  result.centerCm = center;
  result.time = static_cast<double>(plotfile.time());
  result.maxLevel = maxLevel;
  OctreeGeometry& geometry = result.geometry;
  for (int axis = 0; axis < 3; ++axis) {
    geometry.domainLo[axis] =
        (static_cast<double>(probLo[axis]) - center[axis]) * toOutput;
  }
  geometry.finestCellSize = static_cast<double>(dxFinest[0]) * toOutput;
  geometry.topDepth = topDepth;

  OctreeParams& params = result.params;
  params.toleranceLogDensity = options.toleranceLogDensity;
  params.toleranceLogTemperature = options.toleranceLogTemperature;
  params.minDepth = std::min(options.minDepth, topDepth);
  params.topDepth = topDepth;
  const int partitionDepth = std::min(options.partitionDepth, topDepth);

  std::vector<amrex::BoxArray> boxArrays(nLevels);
  std::vector<amrex::DistributionMapping> distributions(nLevels);
  std::vector<amrex::BoxArray> coveredBy(nLevels);
  std::vector<std::unique_ptr<amrex::VisMF>> vismfs(nLevels);
  for (int level = 0; level <= maxLevel; ++level) {
    boxArrays[level] = plotfile.boxArray(level);
    distributions[level] = plotfile.DistributionMap(level);
    if (level < maxLevel) {
      coveredBy[level] = amrex::coarsen(plotfile.boxArray(level + 1),
                                        plotfile.refRatio(level));
    }
    vismfs[level] =
        std::make_unique<amrex::VisMF>(amrex::MultiFabFileFullPrefix(
            level, options.plotfilePath, "Level_", "Cell"));
  }

  // Owner of a partition region = reader of the finest box that touches it.
  std::unordered_map<std::uint64_t, int> regionOwnerCache;
  auto regionOwner = [&](int cx, int cy, int cz) {
    const std::uint64_t key = MortonCode(static_cast<std::uint32_t>(cx),
                                         static_cast<std::uint32_t>(cy),
                                         static_cast<std::uint32_t>(cz));
    const auto cached = regionOwnerCache.find(key);
    if (cached != regionOwnerCache.end()) {
      return cached->second;
    }
    int owner = 0;
    for (int level = maxLevel; level >= 0; --level) {
      if (levelDepth[level] >= partitionDepth) {
        break;
      }
      const int n = 1 << (partitionDepth - levelDepth[level]);
      const amrex::Box regionBox(
          amrex::IntVect(cx * n, cy * n, cz * n),
          amrex::IntVect(cx * n + n - 1, cy * n + n - 1, cz * n + n - 1));
      const auto isects = boxArrays[level].intersections(regionBox, true, 0);
      if (!isects.empty()) {
        owner = distributions[level][isects.front().first];
        break;
      }
    }
    regionOwnerCache.emplace(key, owner);
    return owner;
  };

  ExportStatistics local;
  std::vector<SplatRecord> emitted;
  std::vector<Node> upperNodes;
  std::vector<std::vector<Node>> mixedOutgoing(nprocs);
  double localReduceSeconds = 0.0;

  const auto& region = options.region;
  auto insideRegion = [&](const std::array<double, 3>& p) {
    if (!region) {
      return true;
    }
    for (int axis = 0; axis < 3; ++axis) {
      if (p[axis] < (*region)[axis] || p[axis] > (*region)[axis + 3]) {
        return false;
      }
    }
    return true;
  };

  for (int level = maxLevel; level >= 0; --level) {
    const int depth = levelDepth[level];
    const double cellSize = std::ldexp(geometry.finestCellSize, depth);
    const amrex::BoxArray& boxes = boxArrays[level];
    for (int boxIndex = 0; boxIndex < static_cast<int>(boxes.size());
         ++boxIndex) {
      if (distributions[level][boxIndex] != rank) {
        continue;
      }
      const amrex::Box box = boxes[boxIndex];
      if (region) {
        bool overlaps = true;
        for (int axis = 0; axis < 3; ++axis) {
          const double lo =
              geometry.domainLo[axis] + box.smallEnd(axis) * cellSize;
          const double hi =
              geometry.domainLo[axis] + (box.bigEnd(axis) + 1) * cellSize;
          overlaps =
              overlaps && hi >= (*region)[axis] && lo <= (*region)[axis + 3];
        }
        if (!overlaps) {
          continue;
        }
      }

      const HostComponent density =
          ReadComponent(*vismfs[level], boxIndex, densityComp);
      const HostComponent temperature =
          ReadComponent(*vismfs[level], boxIndex, temperatureComp);
      local.cellsRead += static_cast<std::uint64_t>(box.numPts());

      // Cells covered by the next finer level are represented by it.
      std::vector<std::uint8_t> covered(static_cast<std::size_t>(box.numPts()),
                                        0);
      const amrex::IntVect boxLo = box.smallEnd();
      const amrex::IntVect boxLen = box.length();
      auto cellOffset = [&](int i, int j, int k) {
        return static_cast<std::size_t>(i - boxLo[0]) +
               static_cast<std::size_t>(boxLen[0]) *
                   (static_cast<std::size_t>(j - boxLo[1]) +
                    static_cast<std::size_t>(boxLen[1]) *
                        static_cast<std::size_t>(k - boxLo[2]));
      };
      if (level < maxLevel) {
        for (const auto& isect : coveredBy[level].intersections(box)) {
          const amrex::Box& b = isect.second;
          for (int k = b.smallEnd(2); k <= b.bigEnd(2); ++k) {
            for (int j = b.smallEnd(1); j <= b.bigEnd(1); ++j) {
              for (int i = b.smallEnd(0); i <= b.bigEnd(0); ++i) {
                covered[cellOffset(i, j, k)] = 1;
              }
            }
          }
        }
      }

      auto appendLeaves = [&](const amrex::Box& cells, std::vector<Node>& out) {
        for (int k = cells.smallEnd(2); k <= cells.bigEnd(2); ++k) {
          for (int j = cells.smallEnd(1); j <= cells.bigEnd(1); ++j) {
            for (int i = cells.smallEnd(0); i <= cells.bigEnd(0); ++i) {
              if (covered[cellOffset(i, j, k)]) {
                continue;
              }
              const std::array<double, 3> p = {
                  geometry.domainLo[0] + (i + 0.5) * cellSize,
                  geometry.domainLo[1] + (j + 0.5) * cellSize,
                  geometry.domainLo[2] + (k + 0.5) * cellSize};
              if (!insideRegion(p)) {
                continue;
              }
              const double rho = static_cast<double>(density(i, j, k));
              const double temp = static_cast<double>(temperature(i, j, k));
              if (!std::isfinite(rho) || !std::isfinite(temp)) {
                ++local.cellsSkippedInvalid;
                continue;
              }
              const double rhoClamped = std::max(rho, 0.0);
              const double logRho =
                  rhoClamped > 0.0 ? std::log10(rhoClamped) : -300.0;
              const double logT = temp > 0.0 ? std::log10(temp) : -300.0;
              const int phase = PhaseOf(logT, options.phaseEdges);
              out.push_back(MakeLeafNode(i,
                                         j,
                                         k,
                                         depth,
                                         phase,
                                         p,
                                         cellSize,
                                         rhoClamped,
                                         logRho,
                                         logT));
              local.cellWeight += out.back().weight;
              local.cellVolume += out.back().volume;
              ++local.cellsExported;
            }
          }
        }
      };

      if (depth >= partitionDepth) {
        // Cells coarser than a partition region join the upper tree directly.
        appendLeaves(box, upperNodes);
        continue;
      }

      const int n = 1 << (partitionDepth - depth);
      const amrex::IntVect regionLo = boxLo / n;
      const amrex::IntVect regionHi = box.bigEnd() / n;
      std::vector<Node> leaves;
      for (int rz = regionLo[2]; rz <= regionHi[2]; ++rz) {
        for (int ry = regionLo[1]; ry <= regionHi[1]; ++ry) {
          for (int rx = regionLo[0]; rx <= regionHi[0]; ++rx) {
            const amrex::Box regionBox(
                amrex::IntVect(rx * n, ry * n, rz * n),
                amrex::IntVect(rx * n + n - 1, ry * n + n - 1, rz * n + n - 1));
            leaves.clear();
            appendLeaves(regionBox & box, leaves);
            if (leaves.empty()) {
              continue;
            }
            const bool complete =
                box.contains(regionBox) &&
                (level == maxLevel || !coveredBy[level].intersects(regionBox));
            if (complete) {
              const auto reduceStart = Clock::now();
              ReduceToDepth(
                  leaves, partitionDepth, params, upperNodes, emitted);
              localReduceSeconds += SecondsSince(reduceStart);
            } else {
              auto& bucket = mixedOutgoing[regionOwner(rx, ry, rz)];
              bucket.insert(bucket.end(), leaves.begin(), leaves.end());
              local.mixedLeavesExchanged += leaves.size();
            }
          }
        }
      }
    }
  }
  local.readSeconds = SecondsSince(readStart) - localReduceSeconds;

  // Regions that straddle boxes or AMR levels are reduced by their owner.
  const auto exchangeStart = Clock::now();
  std::vector<Node> received =
      ExchangeNodes(mixedOutgoing, comm, nodeType.get());
  {
    std::vector<std::pair<std::uint64_t, std::uint32_t>> order;
    order.reserve(received.size());
    for (std::size_t i = 0; i < received.size(); ++i) {
      const Node& node = received[i];
      const int shift = partitionDepth - node.depth;
      order.emplace_back(
          MortonCode(static_cast<std::uint32_t>(node.ix >> shift),
                     static_cast<std::uint32_t>(node.iy >> shift),
                     static_cast<std::uint32_t>(node.iz >> shift)),
          static_cast<std::uint32_t>(i));
    }
    std::sort(order.begin(), order.end());
    std::vector<Node> group;
    std::size_t begin = 0;
    while (begin < order.size()) {
      std::size_t end = begin;
      group.clear();
      while (end < order.size() && order[end].first == order[begin].first) {
        group.push_back(received[order[end].second]);
        ++end;
      }
      ReduceToDepth(group, partitionDepth, params, upperNodes, emitted);
      begin = end;
    }
    std::vector<Node>().swap(received);
  }
  local.exchangeSeconds = SecondsSince(exchangeStart);

  // Upper tree: merge one depth at a time, routing siblings to one rank.
  const auto upperStart = Clock::now();
  for (int depth = partitionDepth; depth < topDepth; ++depth) {
    std::vector<std::vector<Node>> outgoing(nprocs);
    std::vector<Node> keep;
    for (const Node& node : upperNodes) {
      if (node.depth == depth) {
        outgoing
            [UpperTreeOwner(
                 depth + 1, node.ix >> 1, node.iy >> 1, node.iz >> 1, nprocs)]
                .push_back(node);
      } else if (node.depth > depth) {
        keep.push_back(node);
      } else {
        throw std::logic_error("Upper tree node below the current depth");
      }
    }
    upperNodes.swap(keep);
    std::vector<Node> siblings = ExchangeNodes(outgoing, comm, nodeType.get());
    ReduceOneDepth(siblings, depth, params, upperNodes, emitted);
  }
  for (const Node& root : upperNodes) {
    local.rootWeight += root.weight;
    local.rootVolume += root.volume;
  }
  EmitRoots(upperNodes, params, emitted);
  local.upperTreeSeconds = SecondsSince(upperStart);
  local.localReduceSeconds = localReduceSeconds;

  local.recordsPerDepth.assign(static_cast<std::size_t>(topDepth) + 1, 0);
  local.recordsPerPhase.assign(options.phaseEdges.size() + 1, 0);
  for (const SplatRecord& record : emitted) {
    ++local.recordsPerDepth[record.depth];
    ++local.recordsPerPhase[record.phase];
    if (record.flags & kRecordLeaf) {
      ++local.leafRecords;
    }
    if (record.flags & kRecordHomogeneous) {
      ++local.homogeneousRecords;
    }
  }
  local.records = emitted.size();

  ExportStatistics& global = result.statistics;
  global = local;
  ReduceSum(global.cellsRead, comm);
  ReduceSum(global.cellsExported, comm);
  ReduceSum(global.cellsSkippedInvalid, comm);
  ReduceSum(global.mixedLeavesExchanged, comm);
  ReduceSum(global.records, comm);
  ReduceSum(global.leafRecords, comm);
  ReduceSum(global.homogeneousRecords, comm);
  ReduceSum(global.recordsPerDepth, comm);
  ReduceSum(global.recordsPerPhase, comm);
  ReduceSum(global.cellWeight, comm);
  ReduceSum(global.rootWeight, comm);
  ReduceSum(global.cellVolume, comm);
  ReduceSum(global.rootVolume, comm);
  ReduceMax(global.readSeconds, comm);
  ReduceMax(global.localReduceSeconds, comm);
  ReduceMax(global.exchangeSeconds, comm);
  ReduceMax(global.upperTreeSeconds, comm);

  result.localRecords = std::move(emitted);
  return result;
}

std::string BuildSplatFileHeader(const ExportOptions& options,
                                 const ExportResult& result) {
  std::ostringstream os;
  os << std::setprecision(17);
  os << "{\n";
  os << "  \"format\": \"amrVolumeRenderer.splats\",\n";
  os << "  \"version\": 1,\n";
  os << "  \"headerBytes\": " << kSplatFileHeaderBytes << ",\n";
  os << "  \"recordBytes\": " << sizeof(SplatRecord) << ",\n";
  os << "  \"recordCount\": " << result.statistics.records << ",\n";
  os << "  \"recordLayout\": \"position f32x3, covariance f32x6 (xx xy xz yy "
        "yz zz), volume f32, logDensity f32, logTemperature f32, key i32x3, "
        "depth u8, phase u8, flags u8, reserved u8\",\n";
  os << "  \"recordFlags\": {\"leaf\": " << int{kRecordLeaf}
     << ", \"homogeneous\": " << int{kRecordHomogeneous}
     << ", \"root\": " << int{kRecordRoot} << "},\n";
  os << "  \"lengthUnit\": \"" << JsonEscape(options.lengthUnitName) << "\",\n";
  os << "  \"lengthUnitCm\": " << options.lengthUnitCm << ",\n";
  os << "  \"centerCm\": ";
  WriteJsonArray(os, result.centerCm);
  os << ",\n";
  os << "  \"domainLo\": ";
  WriteJsonArray(os, result.geometry.domainLo);
  os << ",\n";
  os << "  \"finestCellSize\": " << result.geometry.finestCellSize << ",\n";
  os << "  \"topDepth\": " << result.geometry.topDepth << ",\n";
  os << "  \"minDepth\": " << result.params.minDepth << ",\n";
  os << "  \"phaseEdgesLog10Temperature\": ";
  WriteJsonArray(os, options.phaseEdges);
  os << ",\n";
  os << "  \"toleranceLog10Density\": " << options.toleranceLogDensity << ",\n";
  os << "  \"toleranceLog10Temperature\": " << options.toleranceLogTemperature
     << ",\n";
  os << "  \"densityVariable\": \"" << JsonEscape(options.densityVariable)
     << "\",\n";
  os << "  \"temperatureVariable\": \""
     << JsonEscape(options.temperatureVariable) << "\",\n";
  os << "  \"plotfile\": \"" << JsonEscape(options.plotfilePath) << "\",\n";
  os << "  \"time\": " << result.time << ",\n";
  os << "  \"maxLevel\": " << result.maxLevel << ",\n";
  os << "  \"lodRule\": \"draw a record iff (leaf or m(self) <= t) and (root "
        "or m(parent) > t), where m(node) = R / (|eye - c| - R) for the "
        "bounding sphere (center c, radius R) of the node's octree cube, and "
        "the parent key is (key >> 1, depth + 1)\"\n";
  os << "}\n";
  return os.str();
}

void WriteSplatFile(const std::string& path,
                    const ExportOptions& options,
                    ExportResult& result) {
  MPI_Comm comm = amrex::ParallelDescriptor::Communicator();
  const int rank = amrex::ParallelDescriptor::MyProc();

  std::sort(result.localRecords.begin(),
            result.localRecords.end(),
            RecordSpatialLess);

  std::string header = BuildSplatFileHeader(options, result);
  if (header.size() >= kSplatFileHeaderBytes) {
    throw std::runtime_error("Splat file header exceeds " +
                             std::to_string(kSplatFileHeaderBytes) + " bytes");
  }
  header.resize(kSplatFileHeaderBytes, ' ');
  header.back() = '\n';

  std::uint64_t localCount = result.localRecords.size();
  std::uint64_t offset = 0;
  MPI_Exscan(&localCount, &offset, 1, MPI_UINT64_T, MPI_SUM, comm);
  if (rank == 0) {
    offset = 0;
  }

  MPI_File file;
  int status = MPI_File_open(comm,
                             path.c_str(),
                             MPI_MODE_CREATE | MPI_MODE_WRONLY,
                             MPI_INFO_NULL,
                             &file);
  if (status != MPI_SUCCESS) {
    throw std::runtime_error("Unable to open '" + path + "' for writing");
  }
  MPI_File_set_size(file, 0);
  if (rank == 0) {
    MPI_File_write_at(file,
                      0,
                      header.data(),
                      static_cast<int>(header.size()),
                      MPI_BYTE,
                      MPI_STATUS_IGNORE);
  }
  constexpr std::uint64_t kChunkRecords = std::uint64_t{1} << 24;  // 1 GiB
  const char* bytes = reinterpret_cast<const char*>(result.localRecords.data());
  for (std::uint64_t written = 0; written < localCount;) {
    const std::uint64_t count = std::min(kChunkRecords, localCount - written);
    const MPI_Offset position = static_cast<MPI_Offset>(
        kSplatFileHeaderBytes + (offset + written) * sizeof(SplatRecord));
    status = MPI_File_write_at(file,
                               position,
                               bytes + written * sizeof(SplatRecord),
                               static_cast<int>(count * sizeof(SplatRecord)),
                               MPI_BYTE,
                               MPI_STATUS_IGNORE);
    if (status != MPI_SUCCESS) {
      throw std::runtime_error("Failed to write splat records to '" + path +
                               "'");
    }
    written += count;
  }
  MPI_File_close(&file);
}

void PrintExportSummary(const ExportOptions& options,
                        const ExportResult& result) {
  const ExportStatistics& s = result.statistics;
  std::ostringstream os;
  os << std::setprecision(4);
  os << "Gaussian splat export of '" << options.plotfilePath << "'\n";
  os << "  levels 0-" << result.maxLevel << ", top depth "
     << result.geometry.topDepth << ", finest cell "
     << result.geometry.finestCellSize << ' ' << options.lengthUnitName << "\n";
  os << "  cells read " << s.cellsRead << ", exported " << s.cellsExported
     << ", invalid " << s.cellsSkippedInvalid << ", exchanged at region "
     << "boundaries " << s.mixedLeavesExchanged << "\n";
  os << "  records " << s.records << " (" << s.leafRecords << " leaves, "
     << s.homogeneousRecords << " homogeneous), "
     << (s.records > 0 ? static_cast<double>(s.cellsExported) /
                             static_cast<double>(s.records)
                       : 0.0)
     << " cells per record, "
     << static_cast<double>(s.records * sizeof(SplatRecord)) / (1 << 20)
     << " MiB\n";
  os << "  records per depth:";
  for (std::size_t d = 0; d < s.recordsPerDepth.size(); ++d) {
    if (s.recordsPerDepth[d] > 0) {
      os << ' ' << d << ':' << s.recordsPerDepth[d];
    }
  }
  os << "\n  records per phase:";
  for (std::size_t p = 0; p < s.recordsPerPhase.size(); ++p) {
    os << ' ' << p << ':' << s.recordsPerPhase[p];
  }
  const double massError =
      s.cellWeight > 0.0 ? std::abs(s.rootWeight - s.cellWeight) / s.cellWeight
                         : 0.0;
  const double volumeError =
      s.cellVolume > 0.0 ? std::abs(s.rootVolume - s.cellVolume) / s.cellVolume
                         : 0.0;
  os << "\n  relative mass error " << massError << ", volume error "
     << volumeError << "\n";
  os << "  time (s): read " << s.readSeconds << ", local reduce "
     << s.localReduceSeconds << ", boundary exchange " << s.exchangeSeconds
     << ", upper tree " << s.upperTreeSeconds << "\n";
  std::cout << os.str() << std::flush;
}

}  // namespace splat
}  // namespace amrVolumeRenderer
