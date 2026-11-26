// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// splat_render: offline renderer and benchmark for splat files written by
// gaussian_export. Renders with the CUDA backend when available.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "SplatRenderer.hpp"
#include "SplatScene.hpp"

namespace {

using namespace splatrender;

void PrintUsage() {
  std::cout << R"(Usage: splat_render FILE.splat [options]

Camera (output length units of the splat file, e.g. kpc):
  --width W --height H        image size (default 1920 1080)
  --eye X Y Z                 eye / head position (default: in front of the data)
  --target X Y Z              look-at point (default: center of the data)
  --up X Y Z                  up vector (default 0 1 0)
  --fov DEG                   vertical field of view (default 45)
  --screen LLx LLy LLz LRx LRy LRz ULx ULy ULz
                              off-axis screen corners (lower-left, lower-right,
                              upper-left); replaces --target/--up/--fov
  --stereo IPD                row-interleaved stereo, eyes at head +/- IPD/2
                              along the screen's horizontal axis
  --left-row-parity P         rows 2k+P show the left eye (default 0)

Level of detail:
  --lod-pixels P              draw records about P pixels across (default 1.5)
  --lod-eye X Y Z             LOD reference point (default: the head)
  --no-lod                    draw every leaf record

Transfer function:
  --density-range LO HI       log10 density range (default: top 4 dex of data)
  --extinction K              extinction per unit length at density HI
                              (default 0.3 / finest cell size)
  --density-exponent G        extinction ~ density^G (default 1)
  --temperature-range LO HI   log10 T range for the colormap (default: data)
  --colormap NAME             inferno | viridis | coolwarm | gray
  --brightness B              emission scale (default 1)
  --sigma-scale S             Gaussian widening (default 1.8)
  --phases LIST               comma-separated phases to draw (default all)
  --background R G B          linear background colour (default 0 0 0)

Output:
  --backend cpu|cuda          (default cuda if available)
  --device N                  CUDA device (default 0)
  --frames N                  render N times and report timings (default 1)
  --exposure E                image exposure (default 1)
  --output FILE.ppm           write the image
  --compare                   render with both backends and report the
                              difference (needs CUDA)

Benchmarking:
  --synthetic-disk N          instead of FILE, generate N random splats shaped
                              like a gas disk (kpc; scale length 3.4, scale
                              height 0.34, cell-sized splats)
)";
}

Vec3 ReadVec3(char** argv, int& i, int argc, const std::string& name) {
  if (i + 3 >= argc) {
    throw std::invalid_argument(name + " needs three values");
  }
  Vec3 v;
  v.x = std::stof(argv[++i]);
  v.y = std::stof(argv[++i]);
  v.z = std::stof(argv[++i]);
  return v;
}

struct Options {
  std::string input;
  int width = 1920;
  int height = 1080;
  bool haveEye = false, haveTarget = false, haveScreen = false;
  Vec3 eye{}, target{}, up = {0.0f, 1.0f, 0.0f};
  float fov = 45.0f;
  Vec3 screen[3]{};
  float stereoIpd = 0.0f;
  int leftRowParity = 0;
  float lodPixels = 1.5f;
  bool lod = true;
  bool haveLodEye = false;
  Vec3 lodEye{};
  bool haveDensityRange = false, haveTemperatureRange = false,
       haveExtinction = false;
  TransferSettings transfer;
  float background[3] = {0.0f, 0.0f, 0.0f};
  std::string backend;
  int device = 0;
  int frames = 1;
  float exposure = 1.0f;
  std::string output;
  bool compare = false;
  std::uint64_t syntheticDisk = 0;
};

Options ParseOptions(int argc, char** argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        throw std::invalid_argument(arg + " needs a value");
      }
      return argv[++i];
    };
    if (arg == "--help" || arg == "-h") {
      PrintUsage();
      std::exit(0);
    } else if (arg == "--width") {
      o.width = std::stoi(next());
    } else if (arg == "--height") {
      o.height = std::stoi(next());
    } else if (arg == "--eye") {
      o.eye = ReadVec3(argv, i, argc, arg);
      o.haveEye = true;
    } else if (arg == "--target") {
      o.target = ReadVec3(argv, i, argc, arg);
      o.haveTarget = true;
    } else if (arg == "--up") {
      o.up = ReadVec3(argv, i, argc, arg);
    } else if (arg == "--fov") {
      o.fov = std::stof(next());
    } else if (arg == "--screen") {
      for (Vec3& corner : o.screen) {
        corner = ReadVec3(argv, i, argc, arg);
      }
      o.haveScreen = true;
    } else if (arg == "--stereo") {
      o.stereoIpd = std::stof(next());
    } else if (arg == "--left-row-parity") {
      o.leftRowParity = std::stoi(next()) & 1;
    } else if (arg == "--lod-pixels") {
      o.lodPixels = std::stof(next());
    } else if (arg == "--lod-eye") {
      o.lodEye = ReadVec3(argv, i, argc, arg);
      o.haveLodEye = true;
    } else if (arg == "--no-lod") {
      o.lod = false;
    } else if (arg == "--density-range") {
      o.transfer.logDensityMin = std::stof(next());
      o.transfer.logDensityMax = std::stof(next());
      o.haveDensityRange = true;
    } else if (arg == "--extinction") {
      o.transfer.extinctionAtMax = std::stof(next());
      o.haveExtinction = true;
    } else if (arg == "--density-exponent") {
      o.transfer.densityExponent = std::stof(next());
    } else if (arg == "--temperature-range") {
      o.transfer.logTemperatureMin = std::stof(next());
      o.transfer.logTemperatureMax = std::stof(next());
      o.haveTemperatureRange = true;
    } else if (arg == "--colormap") {
      o.transfer.colormap = next();
    } else if (arg == "--brightness") {
      o.transfer.brightness = std::stof(next());
    } else if (arg == "--sigma-scale") {
      o.transfer.sigmaScale = std::stof(next());
    } else if (arg == "--phases") {
      o.transfer.phaseMask = 0;
      std::stringstream list(next());
      std::string item;
      while (std::getline(list, item, ',')) {
        if (!item.empty()) {
          o.transfer.phaseMask |= 1u << std::stoi(item);
        }
      }
    } else if (arg == "--background") {
      const Vec3 bg = ReadVec3(argv, i, argc, arg);
      o.background[0] = bg.x;
      o.background[1] = bg.y;
      o.background[2] = bg.z;
    } else if (arg == "--backend") {
      o.backend = next();
    } else if (arg == "--device") {
      o.device = std::stoi(next());
    } else if (arg == "--frames") {
      o.frames = std::max(1, std::stoi(next()));
    } else if (arg == "--exposure") {
      o.exposure = std::stof(next());
    } else if (arg == "--output") {
      o.output = next();
    } else if (arg == "--compare") {
      o.compare = true;
    } else if (arg == "--synthetic-disk") {
      o.syntheticDisk = std::stoull(next());
    } else if (!arg.empty() && arg[0] == '-') {
      throw std::invalid_argument("Unknown option '" + arg + "'");
    } else if (o.input.empty()) {
      o.input = arg;
    } else {
      throw std::invalid_argument("Unexpected argument '" + arg + "'");
    }
  }
  if (o.input.empty() && o.syntheticDisk == 0) {
    throw std::invalid_argument("No splat file given");
  }
  return o;
}

/// Random splats with the shape of an exponential gas disk, for benchmarks.
SplatScene SyntheticDisk(std::uint64_t count) {
  SplatScene scene;
  scene.header.recordCount = count;
  scene.header.finestCellSize = 0.0045;
  scene.header.topDepth = 0;
  scene.header.lengthUnit = "kpc";
  scene.records.resize(count);
  const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
  std::vector<std::thread> workers;
  for (unsigned t = 0; t < threads; ++t) {
    workers.emplace_back([&, t] {
      std::mt19937_64 rng(1234567 + t);
      std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
      std::normal_distribution<float> normal(0.0f, 1.0f);
      std::exponential_distribution<float> radial(1.0f / 3.4f);
      std::exponential_distribution<float> vertical(1.0f / 0.34f);
      for (std::uint64_t i = t; i < count; i += threads) {
        SplatRecord& r = scene.records[i];
        r = SplatRecord{};
        float radius;
        do {
          radius = radial(rng) + radial(rng);  // surface density ~ R exp(-R/h)
        } while (radius > 16.0f);
        const float angle = 6.2831853f * uniform(rng);
        const float height =
            (uniform(rng) < 0.5f ? -1.0f : 1.0f) * vertical(rng);
        r.position[0] = radius * std::cos(angle);
        r.position[1] = radius * std::sin(angle);
        r.position[2] = height;
        // Cell-sized splats: 4.5 pc to ~70 pc, smaller near the midplane.
        const float level =
            std::min(4.0f, std::fabs(height) / 0.4f + 2.0f * uniform(rng));
        const float side = 0.0045f * std::exp2(std::floor(level));
        const float variance = side * side / 12.0f;
        r.covariance[0] = r.covariance[3] = r.covariance[5] = variance;
        r.volume = side * side * side;
        r.logDensity = -22.5f - 0.43f * radius / 3.4f -
                       std::fabs(height) / 0.34f * 0.43f + 0.5f * normal(rng);
        r.logTemperature = uniform(rng) < 0.15f ? 6.0f + 0.5f * normal(rng)
                                                : 4.0f + 0.2f * normal(rng);
        r.phase = r.logTemperature > 5.5f ? 2 : 0;
        r.flags = kRecordLeaf;
      }
    });
  }
  for (auto& w : workers) {
    w.join();
  }
  ComputeSceneRanges(scene);
  return scene;
}

/// Bounds of the leaf records dense enough to be visible.
void VisibleBounds(const SplatScene& scene,
                   float logDensityMin,
                   Vec3& lo,
                   Vec3& hi) {
  lo = {1e30f, 1e30f, 1e30f};
  hi = {-1e30f, -1e30f, -1e30f};
  for (const SplatRecord& r : scene.records) {
    if ((r.flags & kRecordLeaf) && r.logDensity >= logDensityMin) {
      lo = {std::min(lo.x, r.position[0]),
            std::min(lo.y, r.position[1]),
            std::min(lo.z, r.position[2])};
      hi = {std::max(hi.x, r.position[0]),
            std::max(hi.y, r.position[1]),
            std::max(hi.z, r.position[2])};
    }
  }
  if (lo.x > hi.x) {
    lo = scene.boundsLo;
    hi = scene.boundsHi;
  }
}

/// A renderer of either backend behind one interface.
class AnyRenderer {
 public:
  AnyRenderer(const std::string& backend, int device, const SplatScene& scene) {
    if (backend == "cuda") {
      cuda_ = std::make_unique<CudaSplatRenderer>(device);
      cuda_->SetRecords(scene.records.data(), scene.records.size());
      name_ = "cuda (" + cuda_->DeviceName() + ")";
    } else if (backend == "cpu") {
      cpu_.SetRecords(scene.records.data(), scene.records.size());
      name_ = "cpu";
    } else {
      throw std::invalid_argument("Unknown backend '" + backend + "'");
    }
  }
  RenderStats Render(const FrameParams& frame, Image& image) {
    return cuda_ ? cuda_->Render(frame, image) : cpu_.Render(frame, image);
  }
  const std::string& name() const { return name_; }

 private:
  CpuSplatRenderer cpu_;
  std::unique_ptr<CudaSplatRenderer> cuda_;
  std::string name_;
};

/// Frames for one (mono) or two (stereo, half height each) views.
std::vector<FrameParams> BuildFrames(const Options& o,
                                     const SplatScene& scene,
                                     const TransferFunction& tf) {
  Vec3 lo, hi;
  VisibleBounds(scene, tf.logDensityMin, lo, hi);
  const Vec3 center = {
      0.5f * (lo.x + hi.x), 0.5f * (lo.y + hi.y), 0.5f * (lo.z + hi.z)};
  const float extent = std::max({hi.x - lo.x, hi.y - lo.y, hi.z - lo.z, 1e-6f});
  const Vec3 target = o.haveTarget ? o.target : center;
  const Vec3 head =
      o.haveEye ? o.eye : Vec3{center.x, center.y, center.z + 1.5f * extent};
  const float nearDepth = 1e-4f * extent;

  auto cameraFor = [&](const Vec3& eye) {
    if (o.haveScreen) {
      return MakeScreenCamera(eye,
                              o.screen[0],
                              o.screen[1],
                              o.screen[2],
                              o.width,
                              o.height,
                              nearDepth);
    }
    return MakeLookAtCamera(
        eye, target, o.up, o.fov, o.width, o.height, nearDepth);
  };

  const ScreenCamera headCamera = cameraFor(head);
  FrameParams base{};
  base.transfer = tf;
  base.lod = MakeLodParams(scene.header,
                           o.haveLodEye ? o.lodEye : head,
                           LodThresholdForPixels(headCamera, o.lodPixels));
  base.lod.enabled = o.lod ? 1 : 0;
  for (int c = 0; c < 3; ++c) {
    base.background[c] = o.background[c];
  }

  std::vector<FrameParams> frames;
  if (o.stereoIpd <= 0.0f) {
    base.camera = headCamera;
    frames.push_back(base);
    return frames;
  }
  // Eyes are offset along the screen's horizontal axis. Both use the head
  // for LOD so the two eyes draw the same records.
  const Vec3 axis = headCamera.right;
  const float h = 0.5f * o.stereoIpd;
  const Vec3 leftEye = {
      head.x - h * axis.x, head.y - h * axis.y, head.z - h * axis.z};
  const Vec3 rightEye = {
      head.x + h * axis.x, head.y + h * axis.y, head.z + h * axis.z};
  // With --target, keep looking at the same point (toe-in); with --screen
  // the projection is off-axis and correct by construction.
  for (int parity = 0; parity < 2; ++parity) {
    const Vec3& eye = (parity == o.leftRowParity) ? leftEye : rightEye;
    FrameParams frame = base;
    frame.camera = HalfHeightEyeCamera(cameraFor(eye), parity);
    frames.push_back(frame);
  }
  return frames;
}

Image RenderViews(AnyRenderer& renderer,
                  const std::vector<FrameParams>& frames,
                  RenderStats& total) {
  std::vector<Image> images(frames.size());
  total = RenderStats{};
  for (std::size_t v = 0; v < frames.size(); ++v) {
    const RenderStats s = renderer.Render(frames[v], images[v]);
    total.records = s.records;
    total.visibleSplats += s.visibleSplats;
    total.tileInstances += s.tileInstances;
    total.preprocessMs += s.preprocessMs;
    total.sortMs += s.sortMs;
    total.rasterMs += s.rasterMs;
    total.totalMs += s.totalMs;
  }
  return frames.size() == 1 ? images[0] : InterleaveRows(images[0], images[1]);
}

void PrintStats(const std::string& name, const RenderStats& s, int views) {
  std::printf(
      "%s: %llu records, %llu splats drawn, %llu tile instances (%d view%s)\n"
      "  preprocess %.2f ms, sort %.2f ms, raster %.2f ms, total %.2f ms\n",
      name.c_str(),
      static_cast<unsigned long long>(s.records),
      static_cast<unsigned long long>(s.visibleSplats),
      static_cast<unsigned long long>(s.tileInstances),
      views,
      views == 1 ? "" : "s",
      s.preprocessMs,
      s.sortMs,
      s.rasterMs,
      s.totalMs);
}

int Run(int argc, char** argv) {
  Options o = ParseOptions(argc, argv);
  const SplatScene scene = o.syntheticDisk > 0 ? SyntheticDisk(o.syntheticDisk)
                                               : LoadSplatFile(o.input);
  if (o.syntheticDisk > 0) {
    o.lod = false;  // synthetic records carry no octree keys
    o.input = "synthetic disk";
  }
  std::printf("Loaded %zu records from %s (top depth %d, finest cell %g %s)\n",
              scene.records.size(),
              o.input.c_str(),
              scene.header.topDepth,
              scene.header.finestCellSize,
              scene.header.lengthUnit.c_str());

  TransferSettings& t = o.transfer;
  if (!o.haveDensityRange) {
    t.logDensityMax = scene.logDensityRange[1];
    t.logDensityMin =
        std::max(scene.logDensityRange[0], t.logDensityMax - 4.0f);
  }
  if (!o.haveTemperatureRange) {
    t.logTemperatureMin = scene.logTemperatureRange[0];
    t.logTemperatureMax =
        std::max(scene.logTemperatureRange[1], t.logTemperatureMin + 0.1f);
  }
  if (!o.haveExtinction) {
    t.extinctionAtMax = 0.3f / static_cast<float>(scene.header.finestCellSize);
  }
  const TransferFunction tf = MakeTransferFunction(t);
  std::printf(
      "Transfer function: log density [%g, %g], extinction %g at the top, "
      "log T [%g, %g], %s\n",
      t.logDensityMin,
      t.logDensityMax,
      t.extinctionAtMax,
      t.logTemperatureMin,
      t.logTemperatureMax,
      t.colormap.c_str());

  const std::vector<FrameParams> frames = BuildFrames(o, scene, tf);
  const int views = static_cast<int>(frames.size());

  if (o.backend.empty()) {
    o.backend = CudaBackendAvailable() ? "cuda" : "cpu";
  }
  AnyRenderer renderer(o.backend, o.device, scene);
  RenderStats stats;
  Image image;
  double best = 1e30;
  for (int f = 0; f < o.frames; ++f) {
    image = RenderViews(renderer, frames, stats);
    best = std::min(best, stats.totalMs);
    if (f == 0 || f == o.frames - 1) {
      PrintStats(
          renderer.name() + (f == 0 ? " (first frame)" : " (last frame)"),
          stats,
          views);
    }
  }
  if (o.frames > 1) {
    std::printf("Best of %d frames: %.2f ms (%.1f fps)\n",
                o.frames,
                best,
                1000.0 / best);
  }

  if (o.compare) {
    if (!CudaBackendAvailable()) {
      throw std::runtime_error("--compare needs the CUDA backend");
    }
    AnyRenderer other(o.backend == "cuda" ? "cpu" : "cuda", o.device, scene);
    RenderStats otherStats;
    const Image reference = RenderViews(other, frames, otherStats);
    PrintStats(other.name(), otherStats, views);
    double maxDiff = 0.0, sumDiff = 0.0;
    for (std::size_t i = 0; i < image.rgba.size(); ++i) {
      const double d =
          std::fabs(static_cast<double>(image.rgba[i]) - reference.rgba[i]);
      maxDiff = std::max(maxDiff, d);
      sumDiff += d;
    }
    std::printf("CPU vs CUDA: max |difference| %.3g, mean %.3g\n",
                maxDiff,
                sumDiff / std::max<std::size_t>(1, image.rgba.size()));
  }

  if (!o.output.empty()) {
    WritePpm(image, o.output, o.exposure);
    std::printf("Wrote %dx%d image to %s\n",
                image.width,
                image.height,
                o.output.c_str());
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception& error) {
    std::cerr << "splat_render: " << error.what()
              << "\n(run with --help for usage)" << std::endl;
    return 1;
  }
}
