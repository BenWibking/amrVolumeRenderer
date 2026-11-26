// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// CPU tests of the splat renderer: camera geometry, conservation of optical
// depth on screen, occlusion order, phase masking, and (when built inside
// amrVolumeRenderer) agreement of the LOD rule with the exporter's.

#include <SplatRenderer/SplatKernels.hpp>
#include <SplatRenderer/SplatRenderer.hpp>
#include <SplatRenderer/SplatScene.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef SPLAT_RENDERER_TEST_WITH_EXPORTER
#include <GaussianExport/SplatOctree.hpp>
#endif

namespace {

using namespace splatrender;

#define CHECK(condition)                            \
  do {                                              \
    if (!(condition)) {                             \
      std::ostringstream message_;                  \
      message_ << __FILE__ << ':' << __LINE__       \
               << ": check failed: " << #condition; \
      throw std::runtime_error(message_.str());     \
    }                                               \
  } while (0)

bool Near(double a, double b, double tolerance) {
  return std::fabs(a - b) <= tolerance;
}

SplatRecord MakeSplat(Vec3 position,
                      float sigma,
                      float logDensity,
                      float logTemperature,
                      std::uint8_t phase = 0) {
  SplatRecord r{};
  r.position[0] = position.x;
  r.position[1] = position.y;
  r.position[2] = position.z;
  const float variance = sigma * sigma;
  r.covariance[0] = variance;
  r.covariance[3] = variance;
  r.covariance[5] = variance;
  r.volume =
      std::pow(std::sqrt(12.0f) * sigma, 3.0f);  // cube of side sqrt(12) sigma
  r.logDensity = logDensity;
  r.logTemperature = logTemperature;
  r.phase = phase;
  r.flags = kRecordLeaf | kRecordRoot;
  return r;
}

FrameParams BasicFrame(int width, int height, float extinction) {
  TransferSettings settings;
  settings.logDensityMin = -1.0f;
  settings.logDensityMax = 0.0f;
  settings.extinctionAtMax = extinction;
  settings.densityExponent = 0.0f;  // constant extinction above the minimum
  settings.logTemperatureMin = 0.0f;
  settings.logTemperatureMax = 1.0f;
  settings.colormap = "gray";
  settings.sigmaScale = 1.0f;
  FrameParams frame{};
  frame.transfer = MakeTransferFunction(settings);
  frame.camera = MakeLookAtCamera({0.0f, 0.0f, 10.0f},
                                  {0.0f, 0.0f, 0.0f},
                                  {0.0f, 1.0f, 0.0f},
                                  30.0f,
                                  width,
                                  height,
                                  1e-3f);
  frame.lod.enabled = 0;
  return frame;
}

void TestOffAxisCamera() {
  // A screen rectangle in the z = 0 plane seen from an off-center eye.
  const Vec3 ll = {-2.0f, -1.0f, 0.0f}, lr = {2.0f, -1.0f, 0.0f},
             ul = {-2.0f, 1.0f, 0.0f};
  const ScreenCamera cam =
      MakeScreenCamera({1.0f, 0.5f, 3.0f}, ll, lr, ul, 800, 400, 1e-3f);
  auto project = [&](Vec3 p, float& px, float& py) {
    const Vec3 rel = {p.x - cam.eye.x, p.y - cam.eye.y, p.z - cam.eye.z};
    const float d = -Dot(rel, cam.normal);
    px = cam.fx * Dot(rel, cam.right) / d + cam.cx;
    py = -cam.fy * Dot(rel, cam.up) / d + cam.cy;
  };
  float px, py;
  project(ll, px, py);
  CHECK(Near(px, 0.0, 1e-3) && Near(py, 400.0, 1e-3));
  project(lr, px, py);
  CHECK(Near(px, 800.0, 1e-3) && Near(py, 400.0, 1e-3));
  project(ul, px, py);
  CHECK(Near(px, 0.0, 1e-3) && Near(py, 0.0, 1e-3));
  // A point behind the screen projects through the screen along the ray.
  project({0.0f, 0.0f, -3.0f}, px, py);
  // The ray from the eye crosses the screen at (0.5, 0.25).
  CHECK(Near(px, (0.5 + 2.0) / 4.0 * 800.0, 1e-2));
  CHECK(Near(py, (1.0 - 0.25) / 2.0 * 400.0, 1e-2));

  // Half-height eye cameras sample full-resolution rows 2k + parity.
  for (int parity = 0; parity < 2; ++parity) {
    const ScreenCamera half = HalfHeightEyeCamera(cam, parity);
    const Vec3 p = {0.3f, -0.2f, -1.0f};
    const Vec3 rel = {p.x - cam.eye.x, p.y - cam.eye.y, p.z - cam.eye.z};
    const float d = -Dot(rel, cam.normal);
    const float full = -cam.fy * Dot(rel, cam.up) / d + cam.cy;
    const float halfPy = -half.fy * Dot(rel, half.up) / d + half.cy;
    CHECK(Near(halfPy, (full - parity - 0.5f) / 2.0f + 0.5f, 1e-3));
  }
}

void TestOpticalDepthConserved() {
  const int size = 256;
  const float extinction = 2.0f;  // peak optical depth ~0.6
  FrameParams frame = BasicFrame(size, size, extinction);
  const SplatRecord splat = MakeSplat({0.1f, -0.05f, 0.0f}, 0.05f, 0.0f, 0.5f);
  CpuSplatRenderer renderer;
  renderer.SetRecords(&splat, 1);
  Image image;
  const RenderStats stats = renderer.Render(frame, image);
  CHECK(stats.visibleSplats == 1);
  double sumTau = 0.0;
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const double alpha = image.Pixel(x, y)[3];
      sumTau += -std::log(1.0 - alpha);
    }
  }
  const float d = 10.0f;
  const double expected =
      extinction * splat.volume * frame.camera.fx * frame.camera.fy / (d * d);
  // Tails below the minimum opacity are dropped, so allow a few percent.
  CHECK(sumTau > 0.95 * expected && sumTau <= 1.001 * expected);
}

void TestOcclusionOrder() {
  FrameParams frame = BasicFrame(64, 64, 50.0f);
  // Front splat is hot (white), back splat is cold (black on a red background).
  std::vector<SplatRecord> splats = {
      MakeSplat({0.0f, 0.0f, -2.0f}, 0.3f, 0.0f, 0.0f),
      MakeSplat({0.0f, 0.0f, 2.0f}, 0.3f, 0.0f, 1.0f)};
  frame.background[0] = 1.0f;
  CpuSplatRenderer renderer;
  renderer.SetRecords(splats.data(), splats.size());
  Image image;
  renderer.Render(frame, image);
  const float* center = image.Pixel(32, 32);
  CHECK(center[3] > 0.99f);
  CHECK(center[1] > 0.9f);  // the white front splat dominates
  // Seen from behind, the black splat is in front.
  frame.camera = MakeLookAtCamera({0.0f, 0.0f, -10.0f},
                                  {0.0f, 0.0f, 0.0f},
                                  {0.0f, 1.0f, 0.0f},
                                  30.0f,
                                  64,
                                  64,
                                  1e-3f);
  renderer.Render(frame, image);
  CHECK(image.Pixel(32, 32)[1] < 0.1f);
  // Far from the splats the background shows through.
  CHECK(image.Pixel(1, 1)[0] > 0.99f && image.Pixel(1, 1)[3] < 0.01f);
}

void TestPhaseMaskAndDensityCut() {
  FrameParams frame = BasicFrame(64, 64, 10.0f);
  SplatRecord hidden = MakeSplat({0.0f, 0.0f, 0.0f}, 0.3f, 0.0f, 0.5f, 3);
  ProjectedSplat projected;
  CHECK(ProjectSplat(hidden, frame, projected));
  frame.transfer.phaseMask = ~(1u << 3);
  CHECK(!ProjectSplat(hidden, frame, projected));
  frame.transfer.phaseMask = ~0u;
  hidden.logDensity = -2.0f;  // below the transfer function's range
  CHECK(!ProjectSplat(hidden, frame, projected));
}

#ifdef SPLAT_RENDERER_TEST_WITH_EXPORTER
/// The renderer's float LOD rule must agree with the exporter's double one.
void TestLodMatchesExporter() {
  namespace ex = amrVolumeRenderer::splat;
  const int n = 32;
  ex::OctreeParams params;
  params.topDepth = 5;
  params.toleranceLogDensity = 0.2f;
  ex::OctreeGeometry geometry;
  geometry.domainLo = {-16.0, -16.0, -16.0};
  geometry.finestCellSize = 1.0;
  geometry.topDepth = 5;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  std::vector<ex::Node> leaves;
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        const double rho = i < n / 2 ? 1.0 : std::pow(10.0, uniform(rng));
        leaves.push_back(ex::MakeLeafNode(
            i,
            j,
            k,
            0,
            0,
            {-16.0 + i + 0.5, -16.0 + j + 0.5, -16.0 + k + 0.5},
            1.0,
            rho,
            std::log10(rho),
            4.0));
      }
  std::vector<ex::SplatRecord> exported;
  std::vector<ex::Node> roots;
  ex::ReduceToDepth(leaves, params.topDepth, params, roots, exported);
  ex::EmitRoots(roots, params, exported);

  std::vector<SplatRecord> records(exported.size());
  std::memcpy(
      records.data(), exported.data(), exported.size() * sizeof(SplatRecord));
  SplatFileHeader header;
  header.domainLo[0] = header.domainLo[1] = header.domainLo[2] = -16.0;
  header.finestCellSize = 1.0;
  header.topDepth = 5;

  std::size_t checked = 0, mismatched = 0;
  for (const Vec3 eye : {Vec3{0.0f, 0.0f, 60.0f},
                         Vec3{3.0f, -2.0f, 1.0f},
                         Vec3{-40.0f, 10.0f, 5.0f}}) {
    for (const float threshold : {0.01f, 0.05f, 0.2f, 1.0f}) {
      const LodParams lod = MakeLodParams(header, eye, threshold);
      for (std::size_t i = 0; i < records.size(); ++i) {
        const bool mine = IsSelected(records[i], lod);
        const bool theirs = ex::IsSelected(
            exported[i], geometry, {eye.x, eye.y, eye.z}, threshold);
        ++checked;
        mismatched += mine != theirs ? 1 : 0;
      }
    }
  }
  // Float rounding can only flip records sitting exactly on the threshold.
  CHECK(mismatched * 10000 <= checked);
}
#endif

}  // namespace

int main() {
  try {
    TestOffAxisCamera();
    TestOpticalDepthConserved();
    TestOcclusionOrder();
    TestPhaseMaskAndDensityCut();
#ifdef SPLAT_RENDERER_TEST_WITH_EXPORTER
    TestLodMatchesExporter();
#endif
  } catch (const std::exception& error) {
    std::cerr << "SplatRendererTest failed: " << error.what() << std::endl;
    return 1;
  }
  std::cout << "SplatRendererTest passed" << std::endl;
  return 0;
}
