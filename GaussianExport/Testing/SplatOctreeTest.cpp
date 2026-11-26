// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Serial tests of the splat octree: collapse of uniform data, conservation,
// that the stateless LOD rule tiles the data exactly once, and that
// reducing per region and then merging upward matches a global reduction.

#include <GaussianExport/SplatOctree.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
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

bool Near(double a, double b, double relative) {
  return std::abs(a - b) <=
         relative * std::max({std::abs(a), std::abs(b), 1e-300});
}

struct Cell {
  int ix, iy, iz, depth, phase;
  double density, temperature;
};

/// Leaves for a list of cells on a unit-spaced finest grid at the origin.
std::vector<Node> MakeLeaves(const std::vector<Cell>& cells,
                             const OctreeGeometry& geometry) {
  std::vector<Node> leaves;
  for (const Cell& c : cells) {
    const double size = std::ldexp(geometry.finestCellSize, c.depth);
    const std::array<double, 3> center = {
        geometry.domainLo[0] + (c.ix + 0.5) * size,
        geometry.domainLo[1] + (c.iy + 0.5) * size,
        geometry.domainLo[2] + (c.iz + 0.5) * size};
    leaves.push_back(MakeLeafNode(c.ix,
                                  c.iy,
                                  c.iz,
                                  c.depth,
                                  c.phase,
                                  center,
                                  size,
                                  c.density,
                                  std::log10(c.density),
                                  std::log10(c.temperature)));
  }
  return leaves;
}

std::vector<SplatRecord> BuildAll(std::vector<Node> leaves,
                                  const OctreeParams& params) {
  std::vector<SplatRecord> records;
  std::vector<Node> roots;
  ReduceToDepth(leaves, params.topDepth, params, roots, records);
  EmitRoots(roots, params, records);
  return records;
}

/// Each cell must be covered by exactly one selected record of its phase.
void CheckSelectionTiles(const std::vector<SplatRecord>& records,
                         const std::vector<Cell>& cells,
                         const OctreeGeometry& geometry,
                         const std::array<double, 3>& eye,
                         double threshold) {
  std::vector<const SplatRecord*> selected;
  for (const SplatRecord& r : records) {
    if (IsSelected(r, geometry, eye, threshold)) {
      selected.push_back(&r);
    }
  }
  for (const Cell& c : cells) {
    int hits = 0;
    for (const SplatRecord* r : selected) {
      if (r->phase != c.phase || r->depth < c.depth) {
        continue;
      }
      const int shift = r->depth - c.depth;
      if ((c.ix >> shift) == r->ix && (c.iy >> shift) == r->iy &&
          (c.iz >> shift) == r->iz) {
        ++hits;
      }
    }
    if (hits != 1) {
      std::ostringstream message;
      message << "cell (" << c.ix << ',' << c.iy << ',' << c.iz << ") depth "
              << c.depth << " phase " << c.phase << " covered " << hits
              << " times for threshold " << threshold;
      throw std::runtime_error(message.str());
    }
  }
}

void CheckUniqueKeys(const std::vector<SplatRecord>& records) {
  std::map<std::tuple<int, int, int, int, int>, int> seen;
  for (const SplatRecord& r : records) {
    const int count =
        ++seen[std::make_tuple(r.phase, r.depth, r.ix, r.iy, r.iz)];
    CHECK(count == 1);
  }
}

void CheckManyViews(const std::vector<SplatRecord>& records,
                    const std::vector<Cell>& cells,
                    const OctreeGeometry& geometry) {
  const std::vector<std::array<double, 3>> eyes = {{8.0, 8.0, 40.0},
                                                   {-30.0, 5.0, 2.0},
                                                   {3.0, 3.0, 3.0},
                                                   {100.0, 90.0, 80.0}};
  for (const auto& eye : eyes) {
    for (const double threshold : {1e-4, 0.05, 0.2, 1.0, 1e4}) {
      CheckSelectionTiles(records, cells, geometry, eye, threshold);
    }
  }
}

OctreeGeometry UnitGeometry(int topDepth) {
  OctreeGeometry geometry;
  geometry.domainLo = {0.0, 0.0, 0.0};
  geometry.finestCellSize = 1.0;
  geometry.topDepth = topDepth;
  return geometry;
}

void TestUniformBlockCollapses() {
  const int n = 16;
  const OctreeGeometry geometry = UnitGeometry(4);
  OctreeParams params;
  params.topDepth = 4;
  std::vector<Cell> cells;
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) cells.push_back({i, j, k, 0, 0, 2.0, 1e4});
  const auto records = BuildAll(MakeLeaves(cells, geometry), params);
  CHECK(records.size() == 1);
  const SplatRecord& root = records.front();
  CHECK(root.depth == 4);
  CHECK(root.flags == (kRecordLeaf | kRecordHomogeneous | kRecordRoot));
  CHECK(Near(root.volume, n * n * n, 1e-6));
  CHECK(Near(root.logDensity, std::log10(2.0), 1e-6));
  CHECK(Near(root.logTemperature, 4.0, 1e-6));
  for (int axis = 0; axis < 3; ++axis) {
    CHECK(Near(root.position[axis], 8.0, 1e-6));
  }
  // A uniform cube of side a has variance a^2 / 12 on each axis.
  CHECK(Near(root.covariance[0], n * n / 12.0, 1e-5));
  CHECK(Near(root.covariance[3], n * n / 12.0, 1e-5));
  CHECK(Near(root.covariance[5], n * n / 12.0, 1e-5));
  CHECK(std::abs(root.covariance[1]) < 1e-5);
  CheckManyViews(records, cells, geometry);
}

std::vector<Cell> RandomCells(int n, unsigned seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<Cell> cells;
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        double density = 1.0;
        double temperature = 1e4;
        if (i >= n / 2) {
          // Noisy half with two phases.
          density = std::pow(10.0, 2.0 * uniform(rng));
          temperature = uniform(rng) < 0.3 ? 1e7 : 1e4 * (1.0 + uniform(rng));
        } else if (j >= n / 2) {
          // Smooth gradient: collapses partially.
          density = std::pow(10.0, 0.02 * k);
        }
        const int phase = std::log10(temperature) > 5.0 ? 1 : 0;
        cells.push_back({i, j, k, 0, phase, density, temperature});
      }
    }
  }
  return cells;
}

void TestRandomFieldTilesAndConserves() {
  const int n = 16;
  const OctreeGeometry geometry = UnitGeometry(4);
  OctreeParams params;
  params.topDepth = 4;
  params.toleranceLogDensity = 0.1f;
  params.toleranceLogTemperature = 0.1f;
  const auto cells = RandomCells(n, 12345u);
  auto leaves = MakeLeaves(cells, geometry);
  double cellMass = 0.0;
  for (const Node& leaf : leaves) {
    cellMass += leaf.weight;
  }

  std::vector<SplatRecord> records;
  std::vector<Node> roots;
  ReduceToDepth(leaves, params.topDepth, params, roots, records);
  double rootMass = 0.0;
  for (const Node& root : roots) {
    rootMass += root.weight;
  }
  EmitRoots(roots, params, records);

  CHECK(roots.size() == 2);  // one root per phase
  CHECK(Near(rootMass, cellMass, 1e-12));
  CHECK(records.size() < cells.size() + cells.size() / 2);
  CHECK(records.size() > 2);
  CheckUniqueKeys(records);
  CheckManyViews(records, cells, geometry);

  // Every record must be self-consistent with its phase's edge at 5.
  for (const SplatRecord& r : records) {
    CHECK((r.phase == 1) == (r.logTemperature > 5.0f));
  }
}

void TestMinDepth() {
  const int n = 16;
  const OctreeGeometry geometry = UnitGeometry(4);
  OctreeParams params;
  params.topDepth = 4;
  params.minDepth = 2;
  const auto cells = RandomCells(n, 777u);
  const auto records = BuildAll(MakeLeaves(cells, geometry), params);
  for (const SplatRecord& r : records) {
    CHECK(r.depth >= 2);
    if (r.depth == 2) {
      CHECK(r.flags & kRecordLeaf);
    }
  }
  CheckUniqueKeys(records);
  CheckManyViews(records, cells, geometry);
}

void TestMixedDepthLeaves() {
  // A coarse depth-2 cell grid in the upper half (z >= 8) and finest cells in
  // the lower half, mimicking two AMR levels.
  const OctreeGeometry geometry = UnitGeometry(4);
  OctreeParams params;
  params.topDepth = 4;
  std::vector<Cell> cells;
  std::mt19937 rng(99u);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (int k = 0; k < 8; ++k)
    for (int j = 0; j < 16; ++j)
      for (int i = 0; i < 16; ++i)
        cells.push_back({i, j, k, 0, 0, 1.0 + uniform(rng), 1e4});
  for (int k = 2; k < 4; ++k)
    for (int j = 0; j < 4; ++j)
      for (int i = 0; i < 4; ++i)
        cells.push_back({i, j, k, 2, 0, 1.0 + 10.0 * uniform(rng), 1e4});
  auto leaves = MakeLeaves(cells, geometry);
  double cellVolume = 0.0;
  for (const Node& leaf : leaves) {
    cellVolume += leaf.volume;
  }
  CHECK(Near(cellVolume, 16.0 * 16.0 * 16.0, 1e-12));
  const auto records = BuildAll(leaves, params);
  CheckUniqueKeys(records);
  CheckManyViews(records, cells, geometry);
}

/// Reducing per partition region and then merging depth by depth must give
/// the same records as a single global reduction.
void TestRegionDecompositionMatchesGlobal() {
  const int n = 16;
  const int partitionDepth = 2;
  const OctreeGeometry geometry = UnitGeometry(4);
  OctreeParams params;
  params.topDepth = 4;
  const auto cells = RandomCells(n, 2024u);
  const auto global = BuildAll(MakeLeaves(cells, geometry), params);

  std::map<std::tuple<int, int, int>, std::vector<Node>> regions;
  for (const Node& leaf : MakeLeaves(cells, geometry)) {
    regions[{leaf.ix >> partitionDepth,
             leaf.iy >> partitionDepth,
             leaf.iz >> partitionDepth}]
        .push_back(leaf);
  }
  std::vector<SplatRecord> records;
  std::vector<Node> upper;
  for (auto& entry : regions) {
    ReduceToDepth(entry.second, partitionDepth, params, upper, records);
  }
  for (int depth = partitionDepth; depth < params.topDepth; ++depth) {
    std::vector<Node> current;
    std::vector<Node> keep;
    for (const Node& node : upper) {
      (node.depth == depth ? current : keep).push_back(node);
    }
    // Shuffle to make sure grouping does not depend on input order.
    std::shuffle(current.begin(), current.end(), std::mt19937(depth));
    upper.swap(keep);
    ReduceOneDepth(current, depth, params, upper, records);
  }
  EmitRoots(upper, params, records);

  CHECK(records.size() == global.size());
  auto sorted = [](std::vector<SplatRecord> v) {
    std::sort(
        v.begin(), v.end(), [](const SplatRecord& a, const SplatRecord& b) {
          return std::tie(a.phase, a.depth, a.ix, a.iy, a.iz) <
                 std::tie(b.phase, b.depth, b.ix, b.iy, b.iz);
        });
    return v;
  };
  const auto a = sorted(records);
  const auto b = sorted(global);
  for (std::size_t i = 0; i < a.size(); ++i) {
    CHECK(a[i].phase == b[i].phase && a[i].depth == b[i].depth &&
          a[i].ix == b[i].ix && a[i].iy == b[i].iy && a[i].iz == b[i].iz);
    CHECK(a[i].flags == b[i].flags);
    CHECK(Near(a[i].volume, b[i].volume, 1e-6));
    CHECK(Near(a[i].logDensity, b[i].logDensity, 1e-5));
    for (int axis = 0; axis < 3; ++axis) {
      CHECK(Near(a[i].position[axis], b[i].position[axis], 1e-5));
    }
  }
}

void TestPhaseOf() {
  const std::vector<double> edges = {4.5, 5.5, 6.5};
  CHECK(PhaseOf(4.0, edges) == 0);
  CHECK(PhaseOf(4.5, edges) == 1);
  CHECK(PhaseOf(6.0, edges) == 2);
  CHECK(PhaseOf(9.0, edges) == 3);
  CHECK(PhaseOf(9.0, {}) == 0);
}

}  // namespace

int main() {
  try {
    TestPhaseOf();
    TestUniformBlockCollapses();
    TestRandomFieldTilesAndConserves();
    TestMinDepth();
    TestMixedDepthLeaves();
    TestRegionDecompositionMatchesGlobal();
  } catch (const std::exception& error) {
    std::cerr << "SplatOctreeTest failed: " << error.what() << std::endl;
    return 1;
  }
  std::cout << "SplatOctreeTest passed" << std::endl;
  return 0;
}
