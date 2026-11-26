// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Octree construction for exporting AMR data as a level-of-detail hierarchy
// of Gaussian splats. This header has no AMReX or MPI dependency so that the
// tree logic can be unit tested in isolation.
//
// Conventions
// -----------
// * Every node lives on a global octree defined in the index space of the
//   finest AMR level. A node at depth d covers 2^d finest cells per axis and
//   its integer coordinates are (finest index) >> d. AMR cells from level l
//   enter the tree as leaves at depth d_l = log2(cumulative refinement ratio
//   from level l to the finest level).
// * Nodes are partitioned into independent "phases" (temperature bins). Each
//   phase forms its own forest so that gas at very different temperatures is
//   never averaged into one splat.
// * A node is "homogeneous" when it is completely filled by its phase and the
//   ranges of log density and log temperature inside it are within tolerance.
//   The descendants of a homogeneous node are never emitted.
// * Emitted records carry their octree key, so a renderer can choose a
//   level-of-detail cut per record without pointers (see IsSelected()).

#ifndef AMRVOLUMERENDERER_SPLAT_OCTREE_HPP
#define AMRVOLUMERENDERER_SPLAT_OCTREE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace amrVolumeRenderer {
namespace splat {

/// Deepest supported octree. Node coordinates at depth 0 must fit in 20 bits
/// so that (phase, Morton code) sort keys fit in 64 bits, and the number of
/// finest cells in the root (2^(3*depth)) must fit in a uint64.
constexpr int kMaxTopDepth = 20;

/// Maximum number of temperature phases (the phase index is stored in 4 bits
/// of the sort key).
constexpr int kMaxPhases = 16;

enum NodeFlags : std::uint8_t {
  /// Node is completely filled and uniform within tolerance.
  kNodeHomogeneous = 1u << 0,
  /// Node's emission is deferred because an ancestor may still drop it.
  kNodePending = 1u << 1,
  /// None of this node's descendants are emitted.
  kNodeLeaf = 1u << 2,
};

/// Aggregated statistics of the cells inside one octree node (for a single
/// phase). Trivially copyable so that it can be exchanged over MPI as bytes.
struct Node {
  /// Mass-weighted centroid, in output length units.
  std::array<double, 3> mean;
  /// Mass-weighted covariance (xx, xy, xz, yy, yz, zz), in output units^2.
  std::array<double, 6> covariance;
  /// Sum of density * cell volume (density units * output length units^3).
  double weight;
  /// Occupied volume in output length units^3.
  double volume;
  /// Mass-weighted mean of log10(temperature).
  double meanLogTemperature;
  /// Occupied volume counted in finest cells.
  std::uint64_t cellCount;
  float minLogDensity;
  float maxLogDensity;
  float minLogTemperature;
  float maxLogTemperature;
  std::int32_t ix;
  std::int32_t iy;
  std::int32_t iz;
  std::uint8_t depth;
  std::uint8_t phase;
  std::uint8_t flags;
  std::uint8_t reserved;
};

enum RecordFlags : std::uint8_t {
  /// No descendants of this record are stored in the file.
  kRecordLeaf = 1u << 0,
  /// The node was collapsed because its contents are uniform.
  kRecordHomogeneous = 1u << 1,
  /// The node is at the top depth of the octree (it has no parent).
  kRecordRoot = 1u << 2,
};

/// On-disk splat record (little endian, 64 bytes).
struct SplatRecord {
  /// Mass-weighted centroid in output length units.
  float position[3];
  /// Mass-weighted covariance (xx, xy, xz, yy, yz, zz). Moment matched, so a
  /// uniform cube of side a has variance a^2/12 along each axis.
  float covariance[6];
  /// Occupied volume (output length units^3).
  float volume;
  /// log10 of the volume-averaged density.
  float logDensity;
  /// Mass-weighted mean of log10(temperature).
  float logTemperature;
  /// Octree key: integer coordinates at this record's depth.
  std::int32_t ix;
  std::int32_t iy;
  std::int32_t iz;
  std::uint8_t depth;
  std::uint8_t phase;
  std::uint8_t flags;
  std::uint8_t reserved;
};
static_assert(sizeof(SplatRecord) == 64, "SplatRecord must be 64 bytes");

struct OctreeParams {
  /// Maximum allowed range of log10(density) inside a homogeneous node.
  float toleranceLogDensity = 0.1f;
  /// Maximum allowed range of log10(temperature) inside a homogeneous node.
  float toleranceLogTemperature = 0.1f;
  /// Records shallower than this depth are never emitted (coarsens the data).
  int minDepth = 0;
  /// Depth of the octree root.
  int topDepth = 0;
};

/// Geometry needed to interpret octree keys in output coordinates.
struct OctreeGeometry {
  std::array<double, 3> domainLo = {0.0, 0.0, 0.0};
  double finestCellSize = 1.0;
  int topDepth = 0;
};

/// Index of the temperature phase containing log10(T), given increasing bin
/// edges. Returns a value in [0, edges.size()].
int PhaseOf(double logTemperature, const std::vector<double>& edges);

/// Create a leaf node for a single AMR cell.
///
/// \param ix,iy,iz Cell indices on the cell's own AMR level (= octree
///                 coordinates at depth `depth`).
/// \param center   Cell center in output length units.
/// \param cellSize Cell width in output length units.
/// \param density  Cell density (must be finite and non-negative).
Node MakeLeafNode(std::int32_t ix,
                  std::int32_t iy,
                  std::int32_t iz,
                  int depth,
                  int phase,
                  const std::array<double, 3>& center,
                  double cellSize,
                  double density,
                  double logDensity,
                  double logTemperature);

/// Combine the statistics of sibling nodes into their parent.
Node CombineNodes(const Node* children,
                  std::size_t count,
                  std::int32_t parentX,
                  std::int32_t parentY,
                  std::int32_t parentZ,
                  int parentDepth,
                  int phase);

/// Convert a node to its on-disk record.
SplatRecord MakeRecord(const Node& node, int topDepth);

/// Merge all nodes at `depth` (which must all share that depth) into parents
/// at depth + 1. Parents are appended to `parents`; finished records are
/// appended to `emitted`. `nodes` is reordered.
void ReduceOneDepth(std::vector<Node>& nodes,
                    int depth,
                    const OctreeParams& params,
                    std::vector<Node>& parents,
                    std::vector<SplatRecord>& emitted);

/// Merge nodes of mixed depths (all <= targetDepth) up to targetDepth. Nodes
/// that reach targetDepth are appended to `roots`. Every node must lie inside
/// subtrees that are complete in `nodes`, i.e. no other process may hold
/// cells that share an ancestor below targetDepth.
void ReduceToDepth(std::vector<Node>& nodes,
                   int targetDepth,
                   const OctreeParams& params,
                   std::vector<Node>& roots,
                   std::vector<SplatRecord>& emitted);

/// Emit any still-pending nodes at the top depth of the octree.
void EmitRoots(const std::vector<Node>& roots,
               const OctreeParams& params,
               std::vector<SplatRecord>& emitted);

/// Interleave the low 20 bits of three coordinates into a 60-bit Morton code.
std::uint64_t MortonCode(std::uint32_t x, std::uint32_t y, std::uint32_t z);

/// Spatial ordering of records: by Morton code of the finest-index corner,
/// then coarser (deeper) records first, then by phase.
bool RecordSpatialLess(const SplatRecord& a, const SplatRecord& b);

/// Angular-size LOD metric of the octree cell (ix, iy, iz, depth) seen from
/// `eye`: bounding-sphere radius divided by the distance from the eye to the
/// sphere. Returns +infinity if the eye is inside the sphere. Because child
/// bounding spheres are nested inside their parent's, the metric never
/// increases from a parent to its children.
double LodMetric(std::int64_t ix,
                 std::int64_t iy,
                 std::int64_t iz,
                 int depth,
                 const OctreeGeometry& geometry,
                 const std::array<double, 3>& eye);

/// Stateless level-of-detail selection. A record is drawn iff it is a leaf
/// or small enough (metric <= threshold), and it is a root or its parent is
/// too large (parent metric > threshold). Along any root-to-leaf path exactly
/// one record is selected, so the selection tiles the data with no overlap.
bool IsSelected(const SplatRecord& record,
                const OctreeGeometry& geometry,
                const std::array<double, 3>& eye,
                double threshold);

}  // namespace splat
}  // namespace amrVolumeRenderer

#endif  // AMRVOLUMERENDERER_SPLAT_OCTREE_HPP
