// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Plain-old-data types shared by the CPU and CUDA splat renderers. Everything
// here must be trivially copyable so it can live in CUDA constant memory.

#ifndef SPLAT_RENDERER_SPLAT_TYPES_HPP
#define SPLAT_RENDERER_SPLAT_TYPES_HPP

#include <cstdint>

#if defined(__CUDACC__)
#define SR_HD __host__ __device__
#define SR_INLINE __forceinline__
#else
#define SR_HD
#define SR_INLINE inline
#endif

namespace splatrender {

/// Screen tiles are kTileSize x kTileSize pixels (one CUDA block per tile).
constexpr int kTileSize = 16;
/// Entries in each transfer-function lookup table.
constexpr int kTransferTableSize = 256;

struct Vec3 {
  float x, y, z;
};

/// On-disk record written by gaussian_export (format version 1, 64 bytes).
struct SplatRecord {
  float position[3];
  float covariance[6];  // xx xy xz yy yz zz, moment matched
  float volume;
  float logDensity;
  float logTemperature;
  std::int32_t key[3];
  std::uint8_t depth;
  std::uint8_t phase;
  std::uint8_t flags;
  std::uint8_t reserved;
};
static_assert(sizeof(SplatRecord) == 64, "SplatRecord must be 64 bytes");

enum RecordFlags : std::uint8_t {
  kRecordLeaf = 1u << 0,
  kRecordHomogeneous = 1u << 1,
  kRecordRoot = 1u << 2,
};

/// Off-axis pinhole camera defined by an eye and a planar screen rectangle,
/// as used for head-tracked walls. Pixel coordinates of a point with camera
/// coordinates (x, y, d) (d = distance in front of the eye along the screen
/// normal) are px = fx * x / d + cx, py = -fy * y / d + cy, with row 0 at
/// the top of the screen.
struct ScreenCamera {
  Vec3 eye;
  Vec3 right;   // unit vector along the screen's horizontal edge
  Vec3 up;      // unit vector along the screen's vertical edge
  Vec3 normal;  // unit screen normal, pointing from the screen to the viewer
  float fx, fy, cx, cy;
  float nearDepth;
  int width;
  int height;
  /// Limits on x/d and y/d used to keep the projection Jacobian well
  /// behaved for splats far outside the view (as in 3D Gaussian splatting).
  float xOverDMin, xOverDMax, yOverDMin, yOverDMax;
};

/// Level-of-detail selection (see GaussianExport/README.md).
struct LodParams {
  Vec3 eye;         // shared head position, identical on all processes
  float threshold;  // bounding radius / distance; <= 0 draws leaves only
  float domainLo[3];
  float finestCellSize;
  int topDepth;
  int enabled;  // 0: draw every leaf record
};

/// Separable, editable transfer function: extinction from log10(density),
/// emission colour from log10(temperature).
struct TransferFunction {
  float logDensityMin, logDensityMax;
  float logTemperatureMin, logTemperatureMax;
  /// Extinction coefficient per unit length, sampled uniformly in
  /// log10(density) over [logDensityMin, logDensityMax]. Clamped at the ends
  /// except that densities below logDensityMin give zero extinction.
  float extinction[kTransferTableSize];
  /// Emitted colour (linear RGB) per unit opacity, sampled uniformly in
  /// log10(temperature).
  float color[kTransferTableSize][3];
  /// Widening of the stored moment-matched Gaussians (about 1.8 blends a
  /// uniform lattice smoothly).
  float sigmaScale;
  /// Bit p set = draw temperature phase p.
  std::uint32_t phaseMask;
};

struct FrameParams {
  ScreenCamera camera;
  LodParams lod;
  TransferFunction transfer;
  float background[3];
};

/// A splat after projection to the screen.
struct ProjectedSplat {
  float px, py;                  // center in pixels
  float conicA, conicB, conicC;  // inverse 2D covariance (pixels^-2)
  float tau0;                    // peak optical depth
  float depth;                   // sort key (distance along the normal)
  float r, g, b;                 // emitted colour
  std::int32_t tileMinX, tileMinY, tileMaxX, tileMaxY;  // max is exclusive
};

}  // namespace splatrender

#endif  // SPLAT_RENDERER_SPLAT_TYPES_HPP
