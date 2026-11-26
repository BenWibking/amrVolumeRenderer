// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Per-splat math shared verbatim by the CPU and CUDA renderers: level-of-
// detail selection, transfer-function lookup, EWA projection, and the
// per-pixel opacity. Keeping one implementation is what lets the CPU backend
// serve as a reference for the CUDA backend.
//
// Optical model: a splat with occupied volume V and extinction coefficient
// kappa contributes optical depth tau(x) = kappa * V * |det J| * N(x), where
// N is the projected 2D Gaussian (unit integral, in pixels) and |det J| is
// the pixels-per-unit-area scale of the projection at the splat. The total
// optical depth summed over pixels is kappa * V * |det J| regardless of the
// splat's shape, its level of detail, or the anti-aliasing dilation, so mass
// is conserved on screen in the optically thin limit.

#ifndef SPLAT_RENDERER_SPLAT_KERNELS_HPP
#define SPLAT_RENDERER_SPLAT_KERNELS_HPP

#include <math.h>

#include "SplatTypes.hpp"

namespace splatrender {

/// Opacities below this are skipped, as in 3D Gaussian splatting.
constexpr float kMinAlpha = 1.0f / 255.0f;
/// Per-splat opacity cap (keeps the blend numerically well behaved).
constexpr float kMaxAlpha = 0.99f;
/// Blending stops once the remaining transmittance drops below this.
constexpr float kMinTransmittance = 1e-4f;
/// Variance (pixels^2) added to every projected splat as a low-pass filter.
constexpr float kDilation = 0.3f;
/// Largest splat extent drawn, in standard deviations.
constexpr float kMaxSigmas = 4.0f;

SR_HD SR_INLINE float Dot(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

SR_HD SR_INLINE float ClampF(float v, float lo, float hi) {
  return fminf(fmaxf(v, lo), hi);
}

/// Bounding-sphere angular size of octree cell (ix, iy, iz) at `depth`, as
/// in the exporter: R / (|eye - c| - R), or +infinity inside the sphere.
SR_HD SR_INLINE float LodMetric(
    int ix, int iy, int iz, int depth, const LodParams& lod) {
  const float side = ldexpf(lod.finestCellSize, depth);
  const float cx = lod.domainLo[0] + (static_cast<float>(ix) + 0.5f) * side;
  const float cy = lod.domainLo[1] + (static_cast<float>(iy) + 0.5f) * side;
  const float cz = lod.domainLo[2] + (static_cast<float>(iz) + 0.5f) * side;
  const float radius = 0.86602540378f * side;
  const float dx = lod.eye.x - cx;
  const float dy = lod.eye.y - cy;
  const float dz = lod.eye.z - cz;
  const float distance = sqrtf(dx * dx + dy * dy + dz * dz) - radius;
  if (!(distance > 0.0f)) {
    return INFINITY;
  }
  return radius / distance;
}

/// Stateless LOD rule: draw iff (leaf or small enough) and (root or parent
/// too large). With LOD disabled, every leaf record is drawn.
SR_HD SR_INLINE bool IsSelected(const SplatRecord& record,
                                const LodParams& lod) {
  const bool leaf = (record.flags & kRecordLeaf) != 0;
  if (!lod.enabled) {
    return leaf;
  }
  if (!leaf &&
      LodMetric(
          record.key[0], record.key[1], record.key[2], record.depth, lod) >
          lod.threshold) {
    return false;
  }
  if (record.depth >= lod.topDepth) {
    return true;
  }
  return LodMetric(record.key[0] >> 1,
                   record.key[1] >> 1,
                   record.key[2] >> 1,
                   record.depth + 1,
                   lod) > lod.threshold;
}

SR_HD SR_INLINE float TableCoordinate(float value, float lo, float hi) {
  const float span = hi - lo;
  const float u = span > 0.0f ? (value - lo) / span : 0.0f;
  return ClampF(u, 0.0f, 1.0f) * static_cast<float>(kTransferTableSize - 1);
}

SR_HD SR_INLINE float LookupExtinction(const TransferFunction& tf,
                                       float logDensity) {
  if (!(logDensity >= tf.logDensityMin)) {
    return 0.0f;
  }
  const float u =
      TableCoordinate(logDensity, tf.logDensityMin, tf.logDensityMax);
  const int i0 = static_cast<int>(u);
  const int i1 = i0 + 1 < kTransferTableSize ? i0 + 1 : i0;
  const float f = u - static_cast<float>(i0);
  return tf.extinction[i0] + f * (tf.extinction[i1] - tf.extinction[i0]);
}

SR_HD SR_INLINE void LookupColor(const TransferFunction& tf,
                                 float logTemperature,
                                 float& r,
                                 float& g,
                                 float& b) {
  const float u = TableCoordinate(
      logTemperature, tf.logTemperatureMin, tf.logTemperatureMax);
  const int i0 = static_cast<int>(u);
  const int i1 = i0 + 1 < kTransferTableSize ? i0 + 1 : i0;
  const float f = u - static_cast<float>(i0);
  r = tf.color[i0][0] + f * (tf.color[i1][0] - tf.color[i0][0]);
  g = tf.color[i0][1] + f * (tf.color[i1][1] - tf.color[i0][1]);
  b = tf.color[i0][2] + f * (tf.color[i1][2] - tf.color[i0][2]);
}

SR_HD SR_INLINE int TilesX(const ScreenCamera& camera) {
  return (camera.width + kTileSize - 1) / kTileSize;
}

SR_HD SR_INLINE int TilesY(const ScreenCamera& camera) {
  return (camera.height + kTileSize - 1) / kTileSize;
}

/// Select, classify, and project one record. Returns false if the record is
/// not drawn this frame.
SR_HD SR_INLINE bool ProjectSplat(const SplatRecord& record,
                                  const FrameParams& frame,
                                  ProjectedSplat& out) {
  const TransferFunction& tf = frame.transfer;
  if (record.phase >= 32 || ((tf.phaseMask >> record.phase) & 1u) == 0u) {
    return false;
  }
  if (!IsSelected(record, frame.lod)) {
    return false;
  }
  const float kappa = LookupExtinction(tf, record.logDensity);
  if (!(kappa > 0.0f) || !(record.volume > 0.0f)) {
    return false;
  }

  const ScreenCamera& cam = frame.camera;
  const Vec3 rel = {record.position[0] - cam.eye.x,
                    record.position[1] - cam.eye.y,
                    record.position[2] - cam.eye.z};
  const float xc = Dot(rel, cam.right);
  const float yc = Dot(rel, cam.up);
  const float d = -Dot(rel, cam.normal);
  if (!(d > cam.nearDepth)) {
    return false;
  }

  // World covariance (widened), then camera covariance S = M C M^T with the
  // rows of M being (right, up, -normal).
  const float s2 = tf.sigmaScale * tf.sigmaScale;
  const float* c = record.covariance;
  const float cxx = c[0] * s2, cxy = c[1] * s2, cxz = c[2] * s2;
  const float cyy = c[3] * s2, cyz = c[4] * s2, czz = c[5] * s2;
  const Vec3 rows[3] = {
      cam.right, cam.up, {-cam.normal.x, -cam.normal.y, -cam.normal.z}};
  float mc[3][3];  // M * C
  for (int i = 0; i < 3; ++i) {
    const Vec3 m = rows[i];
    mc[i][0] = m.x * cxx + m.y * cxy + m.z * cxz;
    mc[i][1] = m.x * cxy + m.y * cyy + m.z * cyz;
    mc[i][2] = m.x * cxz + m.y * cyz + m.z * czz;
  }
  float s[3][3];  // (M C) M^T, symmetric
  for (int i = 0; i < 3; ++i) {
    for (int j = i; j < 3; ++j) {
      const Vec3 m = rows[j];
      s[i][j] = mc[i][0] * m.x + mc[i][1] * m.y + mc[i][2] * m.z;
      s[j][i] = s[i][j];
    }
  }

  // Jacobian of (px, py) with respect to camera coordinates (x, y, d).
  const float xd = ClampF(xc / d, cam.xOverDMin, cam.xOverDMax);
  const float yd = ClampF(yc / d, cam.yOverDMin, cam.yOverDMax);
  const float invD = 1.0f / d;
  const float j00 = cam.fx * invD, j02 = -cam.fx * xd * invD;
  const float j11 = -cam.fy * invD, j12 = cam.fy * yd * invD;
  // Sigma_pixels = J S J^T with J = [[j00, 0, j02], [0, j11, j12]].
  float a =
      j00 * j00 * s[0][0] + 2.0f * j00 * j02 * s[0][2] + j02 * j02 * s[2][2];
  const float b = j00 * j11 * s[0][1] + j00 * j12 * s[0][2] +
                  j02 * j11 * s[2][1] + j02 * j12 * s[2][2];
  float cc =
      j11 * j11 * s[1][1] + 2.0f * j11 * j12 * s[1][2] + j12 * j12 * s[2][2];
  a += kDilation;
  cc += kDilation;
  const float det = a * cc - b * b;
  if (!(det > 0.0f)) {
    return false;
  }

  const float pixelsPerArea = fabsf(cam.fx * cam.fy) * invD * invD;
  const float tau0 =
      kappa * record.volume * pixelsPerArea / (6.28318530718f * sqrtf(det));
  // The splat never reaches the minimum opacity anywhere.
  const float minTau = 0.003929f;  // -ln(1 - 1/255)
  if (!(tau0 > minTau)) {
    return false;
  }

  const float px = cam.fx * xc * invD + cam.cx;
  const float py = -cam.fy * yc * invD + cam.cy;
  const float mid = 0.5f * (a + cc);
  const float lambdaMax = mid + sqrtf(fmaxf(0.1f, mid * mid - det));
  const float extentSigmas2 =
      fminf(2.0f * logf(tau0 / minTau), kMaxSigmas * kMaxSigmas);
  const float radius = ceilf(sqrtf(lambdaMax * extentSigmas2));

  const int tilesX = TilesX(cam);
  const int tilesY = TilesY(cam);
  const float tile = static_cast<float>(kTileSize);
  const int minX = static_cast<int>(
      ClampF(floorf((px - radius) / tile), 0.0f, static_cast<float>(tilesX)));
  const int maxX = static_cast<int>(ClampF(
      floorf((px + radius) / tile) + 1.0f, 0.0f, static_cast<float>(tilesX)));
  const int minY = static_cast<int>(
      ClampF(floorf((py - radius) / tile), 0.0f, static_cast<float>(tilesY)));
  const int maxY = static_cast<int>(ClampF(
      floorf((py + radius) / tile) + 1.0f, 0.0f, static_cast<float>(tilesY)));
  if (minX >= maxX || minY >= maxY) {
    return false;
  }

  out.px = px;
  out.py = py;
  out.conicA = cc / det;
  out.conicB = -b / det;
  out.conicC = a / det;
  out.tau0 = tau0;
  out.depth = d;
  LookupColor(tf, record.logTemperature, out.r, out.g, out.b);
  out.tileMinX = minX;
  out.tileMinY = minY;
  out.tileMaxX = maxX;
  out.tileMaxY = maxY;
  return true;
}

/// Opacity of a projected splat at an offset (dx, dy) from its center.
SR_HD SR_INLINE float SplatAlpha(
    float tau0, float conicA, float conicB, float conicC, float dx, float dy) {
  const float power = fminf(
      -0.5f * (conicA * dx * dx + conicC * dy * dy) - conicB * dx * dy, 0.0f);
  const float tau = tau0 * expf(power);
  return fminf(1.0f - expf(-tau), kMaxAlpha);
}

/// Tile-major sort key: tile index in the high 32 bits, depth bits below
/// (positive floats order like their bit patterns).
SR_HD SR_INLINE unsigned long long TileDepthKey(unsigned int tile,
                                                float depth) {
  unsigned int bits;
#if defined(__CUDA_ARCH__)
  bits = __float_as_uint(depth);
#else
  union {
    float f;
    unsigned int u;
  } pun;
  pun.f = depth;
  bits = pun.u;
#endif
  return (static_cast<unsigned long long>(tile) << 32) | bits;
}

}  // namespace splatrender

#endif  // SPLAT_RENDERER_SPLAT_KERNELS_HPP
