// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.

#include "SplatOctree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace amrVolumeRenderer {
namespace splat {

namespace {

constexpr double kLowestLogDensity = -300.0;

std::uint64_t SplitBy3(std::uint32_t value) {
  std::uint64_t x = value & 0x1fffffu;
  x = (x | (x << 32)) & 0x1f00000000ffffull;
  x = (x | (x << 16)) & 0x1f0000ff0000ffull;
  x = (x | (x << 8)) & 0x100f00f00f00f00full;
  x = (x | (x << 4)) & 0x10c30c30c30c30c3ull;
  x = (x | (x << 2)) & 0x1249249249249249ull;
  return x;
}

std::uint64_t ParentSortKey(const Node& node) {
  const std::uint64_t morton =
      MortonCode(static_cast<std::uint32_t>(node.ix >> 1),
                 static_cast<std::uint32_t>(node.iy >> 1),
                 static_cast<std::uint32_t>(node.iz >> 1));
  return (static_cast<std::uint64_t>(node.phase) << 60) | morton;
}

std::uint64_t FullCellCount(int depth) {
  return std::uint64_t{1} << (3 * depth);
}

}  // namespace

int PhaseOf(double logTemperature, const std::vector<double>& edges) {
  const auto it = std::upper_bound(edges.begin(), edges.end(), logTemperature);
  return static_cast<int>(it - edges.begin());
}

Node MakeLeafNode(std::int32_t ix,
                  std::int32_t iy,
                  std::int32_t iz,
                  int depth,
                  int phase,
                  const std::array<double, 3>& center,
                  double cellSize,
                  double density,
                  double logDensity,
                  double logTemperature) {
  Node node{};
  const double volume = cellSize * cellSize * cellSize;
  const double variance = cellSize * cellSize / 12.0;
  node.mean = center;
  node.covariance = {variance, 0.0, 0.0, variance, 0.0, variance};
  node.weight = density * volume;
  node.volume = volume;
  node.meanLogTemperature = logTemperature;
  node.cellCount = FullCellCount(depth);
  node.minLogDensity = static_cast<float>(logDensity);
  node.maxLogDensity = static_cast<float>(logDensity);
  node.minLogTemperature = static_cast<float>(logTemperature);
  node.maxLogTemperature = static_cast<float>(logTemperature);
  node.ix = ix;
  node.iy = iy;
  node.iz = iz;
  node.depth = static_cast<std::uint8_t>(depth);
  node.phase = static_cast<std::uint8_t>(phase);
  node.flags = kNodeHomogeneous | kNodePending | kNodeLeaf;
  return node;
}

Node CombineNodes(const Node* children,
                  std::size_t count,
                  std::int32_t parentX,
                  std::int32_t parentY,
                  std::int32_t parentZ,
                  int parentDepth,
                  int phase) {
  Node parent{};
  parent.ix = parentX;
  parent.iy = parentY;
  parent.iz = parentZ;
  parent.depth = static_cast<std::uint8_t>(parentDepth);
  parent.phase = static_cast<std::uint8_t>(phase);
  parent.minLogDensity = std::numeric_limits<float>::max();
  parent.maxLogDensity = std::numeric_limits<float>::lowest();
  parent.minLogTemperature = std::numeric_limits<float>::max();
  parent.maxLogTemperature = std::numeric_limits<float>::lowest();

  for (std::size_t i = 0; i < count; ++i) {
    const Node& child = children[i];
    parent.weight += child.weight;
    parent.volume += child.volume;
    parent.cellCount += child.cellCount;
    parent.minLogDensity = std::min(parent.minLogDensity, child.minLogDensity);
    parent.maxLogDensity = std::max(parent.maxLogDensity, child.maxLogDensity);
    parent.minLogTemperature =
        std::min(parent.minLogTemperature, child.minLogTemperature);
    parent.maxLogTemperature =
        std::max(parent.maxLogTemperature, child.maxLogTemperature);
  }

  // Weight by mass; fall back to volume if the mass vanishes.
  const bool useMass = parent.weight > 0.0 && std::isfinite(parent.weight);
  auto weightOf = [useMass](const Node& node) {
    return useMass ? node.weight : node.volume;
  };
  double total = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    total += weightOf(children[i]);
  }
  if (!(total > 0.0)) {
    total = 1.0;
  }

  std::array<double, 3> mean = {0.0, 0.0, 0.0};
  double meanLogTemperature = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    const double w = weightOf(children[i]) / total;
    for (int axis = 0; axis < 3; ++axis) {
      mean[axis] += w * children[i].mean[axis];
    }
    meanLogTemperature += w * children[i].meanLogTemperature;
  }

  std::array<double, 6> covariance = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < count; ++i) {
    const Node& child = children[i];
    const double w = weightOf(child) / total;
    const double dx = child.mean[0] - mean[0];
    const double dy = child.mean[1] - mean[1];
    const double dz = child.mean[2] - mean[2];
    covariance[0] += w * (child.covariance[0] + dx * dx);
    covariance[1] += w * (child.covariance[1] + dx * dy);
    covariance[2] += w * (child.covariance[2] + dx * dz);
    covariance[3] += w * (child.covariance[3] + dy * dy);
    covariance[4] += w * (child.covariance[4] + dy * dz);
    covariance[5] += w * (child.covariance[5] + dz * dz);
  }

  parent.mean = mean;
  parent.covariance = covariance;
  parent.meanLogTemperature = meanLogTemperature;
  return parent;
}

SplatRecord MakeRecord(const Node& node, int topDepth) {
  SplatRecord record{};
  for (int axis = 0; axis < 3; ++axis) {
    record.position[axis] = static_cast<float>(node.mean[axis]);
  }
  for (int i = 0; i < 6; ++i) {
    record.covariance[i] = static_cast<float>(node.covariance[i]);
  }
  record.volume = static_cast<float>(node.volume);
  double logDensity = kLowestLogDensity;
  if (node.weight > 0.0 && node.volume > 0.0) {
    logDensity =
        std::max(kLowestLogDensity, std::log10(node.weight / node.volume));
  }
  record.logDensity = static_cast<float>(logDensity);
  record.logTemperature = static_cast<float>(node.meanLogTemperature);
  record.ix = node.ix;
  record.iy = node.iy;
  record.iz = node.iz;
  record.depth = node.depth;
  record.phase = node.phase;
  std::uint8_t flags = 0;
  if (node.flags & kNodeLeaf) {
    flags |= kRecordLeaf;
  }
  if (node.flags & kNodeHomogeneous) {
    flags |= kRecordHomogeneous;
  }
  if (node.depth >= topDepth) {
    flags |= kRecordRoot;
  }
  record.flags = flags;
  return record;
}

void ReduceOneDepth(std::vector<Node>& nodes,
                    int depth,
                    const OctreeParams& params,
                    std::vector<Node>& parents,
                    std::vector<SplatRecord>& emitted) {
  if (nodes.empty()) {
    return;
  }
  if (depth >= params.topDepth) {
    throw std::logic_error("ReduceOneDepth called at or above the top depth");
  }

  std::vector<std::pair<std::uint64_t, std::uint32_t>> order;
  order.reserve(nodes.size());
  for (std::size_t i = 0; i < nodes.size(); ++i) {
    if (nodes[i].depth != depth) {
      throw std::logic_error("ReduceOneDepth received a node of depth " +
                             std::to_string(nodes[i].depth) + ", expected " +
                             std::to_string(depth));
    }
    order.emplace_back(ParentSortKey(nodes[i]), static_cast<std::uint32_t>(i));
  }
  std::sort(order.begin(), order.end());

  const int parentDepth = depth + 1;
  const std::uint64_t fullCount = FullCellCount(parentDepth);
  const bool belowMinDepth = depth < params.minDepth;
  const bool parentBelowMinDepth = parentDepth < params.minDepth;

  std::array<Node, 8> group;
  std::size_t begin = 0;
  while (begin < order.size()) {
    std::size_t end = begin + 1;
    while (end < order.size() && order[end].first == order[begin].first) {
      ++end;
    }
    const std::size_t count = end - begin;
    if (count > group.size()) {
      throw std::logic_error(
          "More than eight children share an octree parent; the input "
          "contains duplicate nodes");
    }
    bool allHomogeneous = true;
    for (std::size_t i = 0; i < count; ++i) {
      group[i] = nodes[order[begin + i].second];
      allHomogeneous =
          allHomogeneous && (group[i].flags & kNodeHomogeneous) != 0;
    }

    const Node& first = group[0];
    Node parent = CombineNodes(group.data(),
                               count,
                               first.ix >> 1,
                               first.iy >> 1,
                               first.iz >> 1,
                               parentDepth,
                               first.phase);

    const bool homogeneous =
        allHomogeneous && parent.cellCount == fullCount &&
        (parent.maxLogDensity - parent.minLogDensity) <=
            params.toleranceLogDensity &&
        (parent.maxLogTemperature - parent.minLogTemperature) <=
            params.toleranceLogTemperature;
    const bool dropChildren = homogeneous || belowMinDepth;

    if (!dropChildren) {
      for (std::size_t i = 0; i < count; ++i) {
        if (group[i].flags & kNodePending) {
          emitted.push_back(MakeRecord(group[i], params.topDepth));
        }
      }
    }

    std::uint8_t flags = 0;
    if (homogeneous) {
      flags |= kNodeHomogeneous;
    }
    if (dropChildren) {
      flags |= kNodeLeaf;
    }
    // A node whose ancestors might still drop it must wait for them.
    const bool pending = homogeneous || parentBelowMinDepth;
    if (pending) {
      flags |= kNodePending;
    }
    parent.flags = flags;
    if (!pending) {
      emitted.push_back(MakeRecord(parent, params.topDepth));
    }
    parents.push_back(parent);
    begin = end;
  }
}

void ReduceToDepth(std::vector<Node>& nodes,
                   int targetDepth,
                   const OctreeParams& params,
                   std::vector<Node>& roots,
                   std::vector<SplatRecord>& emitted) {
  if (nodes.empty()) {
    return;
  }
  std::vector<std::vector<Node>> buckets(static_cast<std::size_t>(targetDepth) +
                                         1);
  for (const Node& node : nodes) {
    if (node.depth > targetDepth) {
      throw std::logic_error(
          "ReduceToDepth received a node deeper than the "
          "target depth");
    }
    buckets[node.depth].push_back(node);
  }
  for (int depth = 0; depth < targetDepth; ++depth) {
    if (buckets[depth].empty()) {
      continue;
    }
    ReduceOneDepth(buckets[depth], depth, params, buckets[depth + 1], emitted);
    std::vector<Node>().swap(buckets[depth]);
  }
  roots.insert(
      roots.end(), buckets[targetDepth].begin(), buckets[targetDepth].end());
}

void EmitRoots(const std::vector<Node>& roots,
               const OctreeParams& params,
               std::vector<SplatRecord>& emitted) {
  for (const Node& root : roots) {
    if (root.flags & kNodePending) {
      emitted.push_back(MakeRecord(root, params.topDepth));
    }
  }
}

std::uint64_t MortonCode(std::uint32_t x, std::uint32_t y, std::uint32_t z) {
  return SplitBy3(x) | (SplitBy3(y) << 1) | (SplitBy3(z) << 2);
}

bool RecordSpatialLess(const SplatRecord& a, const SplatRecord& b) {
  const std::uint64_t ma =
      MortonCode(static_cast<std::uint32_t>(a.ix) << a.depth,
                 static_cast<std::uint32_t>(a.iy) << a.depth,
                 static_cast<std::uint32_t>(a.iz) << a.depth);
  const std::uint64_t mb =
      MortonCode(static_cast<std::uint32_t>(b.ix) << b.depth,
                 static_cast<std::uint32_t>(b.iy) << b.depth,
                 static_cast<std::uint32_t>(b.iz) << b.depth);
  if (ma != mb) {
    return ma < mb;
  }
  if (a.depth != b.depth) {
    return a.depth > b.depth;
  }
  return a.phase < b.phase;
}

double LodMetric(std::int64_t ix,
                 std::int64_t iy,
                 std::int64_t iz,
                 int depth,
                 const OctreeGeometry& geometry,
                 const std::array<double, 3>& eye) {
  const double side = std::ldexp(geometry.finestCellSize, depth);
  const double cx =
      geometry.domainLo[0] + (static_cast<double>(ix) + 0.5) * side;
  const double cy =
      geometry.domainLo[1] + (static_cast<double>(iy) + 0.5) * side;
  const double cz =
      geometry.domainLo[2] + (static_cast<double>(iz) + 0.5) * side;
  const double radius = 0.5 * std::sqrt(3.0) * side;
  const double dx = eye[0] - cx;
  const double dy = eye[1] - cy;
  const double dz = eye[2] - cz;
  const double distance = std::sqrt(dx * dx + dy * dy + dz * dz) - radius;
  if (!(distance > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  return radius / distance;
}

bool IsSelected(const SplatRecord& record,
                const OctreeGeometry& geometry,
                const std::array<double, 3>& eye,
                double threshold) {
  const bool leaf = (record.flags & kRecordLeaf) != 0;
  if (!leaf &&
      LodMetric(record.ix, record.iy, record.iz, record.depth, geometry, eye) >
          threshold) {
    return false;
  }
  if (record.depth >= geometry.topDepth) {
    return true;
  }
  return LodMetric(record.ix >> 1,
                   record.iy >> 1,
                   record.iz >> 1,
                   record.depth + 1,
                   geometry,
                   eye) > threshold;
}

}  // namespace splat
}  // namespace amrVolumeRenderer
