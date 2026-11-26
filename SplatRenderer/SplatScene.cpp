// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.

#include "SplatScene.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace splatrender {

namespace {

constexpr std::size_t kMinimumHeaderBytes = 4096;

/// Position just after `"key":` in a JSON text, or npos.
std::size_t FindValue(const std::string& text, const std::string& key) {
  const std::string pattern = "\"" + key + "\"";
  std::size_t pos = text.find(pattern);
  if (pos == std::string::npos) {
    return pos;
  }
  pos = text.find(':', pos + pattern.size());
  return pos == std::string::npos ? pos : pos + 1;
}

double NumberValue(const std::string& text, const std::string& key) {
  const std::size_t pos = FindValue(text, key);
  if (pos == std::string::npos) {
    throw std::runtime_error("Splat header is missing '" + key + "'");
  }
  return std::strtod(text.c_str() + pos, nullptr);
}

std::vector<double> ArrayValue(const std::string& text,
                               const std::string& key) {
  std::size_t pos = FindValue(text, key);
  if (pos == std::string::npos) {
    throw std::runtime_error("Splat header is missing '" + key + "'");
  }
  pos = text.find('[', pos);
  const std::size_t end = text.find(']', pos);
  if (pos == std::string::npos || end == std::string::npos) {
    throw std::runtime_error("Splat header field '" + key +
                             "' is not an array");
  }
  std::vector<double> values;
  const char* cursor = text.c_str() + pos + 1;
  const char* stop = text.c_str() + end;
  while (cursor < stop) {
    char* next = nullptr;
    const double value = std::strtod(cursor, &next);
    if (next == cursor) {
      ++cursor;  // skip separators
      continue;
    }
    values.push_back(value);
    cursor = next;
  }
  return values;
}

std::string StringValue(const std::string& text, const std::string& key) {
  std::size_t pos = FindValue(text, key);
  if (pos == std::string::npos) {
    return {};
  }
  pos = text.find('"', pos);
  const std::size_t end = text.find('"', pos + 1);
  if (pos == std::string::npos || end == std::string::npos) {
    return {};
  }
  return text.substr(pos + 1, end - pos - 1);
}

Vec3 Sub(const Vec3& a, const Vec3& b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
Vec3 Add(const Vec3& a, const Vec3& b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
Vec3 Scale(const Vec3& a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float DotV(const Vec3& a, const Vec3& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec3 Cross(const Vec3& a, const Vec3& b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
Vec3 Normalize(const Vec3& a) {
  const float n = std::sqrt(DotV(a, a));
  if (!(n > 0.0f)) {
    throw std::invalid_argument("Cannot normalize a zero-length vector");
  }
  return Scale(a, 1.0f / n);
}

struct ColorStop {
  float t, r, g, b;  // sRGB
};

std::vector<ColorStop> ColormapStops(const std::string& name) {
  if (name == "inferno") {
    return {{0.00f, 0.0015f, 0.0005f, 0.0139f},
            {0.25f, 0.258f, 0.039f, 0.406f},
            {0.50f, 0.735f, 0.216f, 0.330f},
            {0.75f, 0.978f, 0.557f, 0.035f},
            {1.00f, 0.988f, 0.998f, 0.645f}};
  }
  if (name == "viridis") {
    return {{0.00f, 0.267f, 0.005f, 0.329f},
            {0.25f, 0.229f, 0.322f, 0.546f},
            {0.50f, 0.128f, 0.567f, 0.551f},
            {0.75f, 0.369f, 0.789f, 0.383f},
            {1.00f, 0.993f, 0.906f, 0.144f}};
  }
  if (name == "coolwarm") {
    return {{0.0f, 0.230f, 0.299f, 0.754f},
            {0.5f, 0.865f, 0.865f, 0.865f},
            {1.0f, 0.706f, 0.016f, 0.150f}};
  }
  if (name == "gray") {
    return {{0.0f, 0.0f, 0.0f, 0.0f}, {1.0f, 1.0f, 1.0f, 1.0f}};
  }
  throw std::invalid_argument("Unknown colormap '" + name + "'");
}

float SrgbToLinear(float c) {
  return c <= 0.04045f ? c / 12.92f : std::pow((c + 0.055f) / 1.055f, 2.4f);
}

}  // namespace

SplatScene LoadSplatFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Unable to open splat file '" + path + "'");
  }
  std::string text(kMinimumHeaderBytes, '\0');
  in.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (in.gcount() != static_cast<std::streamsize>(text.size())) {
    throw std::runtime_error("Splat file '" + path + "' is too short");
  }
  if (text.find("\"amrVolumeRenderer.splats\"") == std::string::npos) {
    throw std::runtime_error("'" + path +
                             "' is not an amrVolumeRenderer splat file");
  }

  SplatScene scene;
  SplatFileHeader& header = scene.header;
  header.text = text.substr(0, text.find('\0'));
  header.recordCount =
      static_cast<std::uint64_t>(NumberValue(text, "recordCount"));
  header.headerBytes =
      static_cast<std::uint64_t>(NumberValue(text, "headerBytes"));
  header.recordBytes =
      static_cast<std::uint64_t>(NumberValue(text, "recordBytes"));
  if (header.recordBytes != sizeof(SplatRecord)) {
    throw std::runtime_error("Unsupported splat record size " +
                             std::to_string(header.recordBytes));
  }
  if (static_cast<int>(NumberValue(text, "version")) != 1) {
    throw std::runtime_error("Unsupported splat file version");
  }
  const auto lo = ArrayValue(text, "domainLo");
  if (lo.size() != 3) {
    throw std::runtime_error("Splat header 'domainLo' must have 3 entries");
  }
  for (int axis = 0; axis < 3; ++axis) {
    header.domainLo[axis] = lo[axis];
  }
  header.finestCellSize = NumberValue(text, "finestCellSize");
  header.topDepth = static_cast<int>(NumberValue(text, "topDepth"));
  header.phaseEdges = ArrayValue(text, "phaseEdgesLog10Temperature");
  header.lengthUnit = StringValue(text, "lengthUnit");

  scene.records.resize(header.recordCount);
  in.seekg(static_cast<std::streamoff>(header.headerBytes));
  in.read(
      reinterpret_cast<char*>(scene.records.data()),
      static_cast<std::streamsize>(scene.records.size() * sizeof(SplatRecord)));
  if (static_cast<std::uint64_t>(in.gcount()) !=
      scene.records.size() * sizeof(SplatRecord)) {
    throw std::runtime_error("Splat file '" + path + "' is truncated");
  }
  ComputeSceneRanges(scene);
  return scene;
}

void ComputeSceneRanges(SplatScene& scene) {
  float rhoLo = std::numeric_limits<float>::max();
  float rhoHi = std::numeric_limits<float>::lowest();
  float tLo = std::numeric_limits<float>::max();
  float tHi = std::numeric_limits<float>::lowest();
  Vec3 lo = {std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max(),
             std::numeric_limits<float>::max()};
  Vec3 hi = {std::numeric_limits<float>::lowest(),
             std::numeric_limits<float>::lowest(),
             std::numeric_limits<float>::lowest()};
  for (const SplatRecord& r : scene.records) {
    if (r.logDensity > -299.0f) {
      rhoLo = std::min(rhoLo, r.logDensity);
      rhoHi = std::max(rhoHi, r.logDensity);
    }
    tLo = std::min(tLo, r.logTemperature);
    tHi = std::max(tHi, r.logTemperature);
    lo = {std::min(lo.x, r.position[0]),
          std::min(lo.y, r.position[1]),
          std::min(lo.z, r.position[2])};
    hi = {std::max(hi.x, r.position[0]),
          std::max(hi.y, r.position[1]),
          std::max(hi.z, r.position[2])};
  }
  if (scene.records.empty()) {
    rhoLo = rhoHi = tLo = tHi = 0.0f;
    lo = hi = {0.0f, 0.0f, 0.0f};
  }
  scene.logDensityRange[0] = rhoLo;
  scene.logDensityRange[1] = rhoHi;
  scene.logTemperatureRange[0] = tLo;
  scene.logTemperatureRange[1] = tHi;
  scene.boundsLo = lo;
  scene.boundsHi = hi;
}

LodParams MakeLodParams(const SplatFileHeader& header,
                        const Vec3& eye,
                        float threshold) {
  LodParams lod{};
  lod.eye = eye;
  lod.threshold = threshold;
  for (int axis = 0; axis < 3; ++axis) {
    lod.domainLo[axis] = static_cast<float>(header.domainLo[axis]);
  }
  lod.finestCellSize = static_cast<float>(header.finestCellSize);
  lod.topDepth = header.topDepth;
  lod.enabled = 1;
  return lod;
}

ScreenCamera MakeScreenCamera(const Vec3& eye,
                              const Vec3& lowerLeft,
                              const Vec3& lowerRight,
                              const Vec3& upperLeft,
                              int width,
                              int height,
                              float nearDepth) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("Image size must be positive");
  }
  const Vec3 right = Normalize(Sub(lowerRight, lowerLeft));
  const Vec3 up = Normalize(Sub(upperLeft, lowerLeft));
  const Vec3 normal = Normalize(Cross(right, up));
  const Vec3 va = Sub(lowerLeft, eye);
  const Vec3 vb = Sub(lowerRight, eye);
  const Vec3 vc = Sub(upperLeft, eye);
  const float screenDistance = -DotV(va, normal);
  if (!(screenDistance > 0.0f)) {
    throw std::invalid_argument("The eye must be in front of the screen");
  }
  const float l = DotV(right, va);
  const float r = DotV(right, vb);
  const float b = DotV(up, va);
  const float t = DotV(up, vc);

  ScreenCamera camera{};
  camera.eye = eye;
  camera.right = right;
  camera.up = up;
  camera.normal = normal;
  camera.width = width;
  camera.height = height;
  camera.fx = screenDistance * static_cast<float>(width) / (r - l);
  camera.fy = screenDistance * static_cast<float>(height) / (t - b);
  camera.cx = -l * static_cast<float>(width) / (r - l);
  camera.cy = t * static_cast<float>(height) / (t - b);
  camera.nearDepth = nearDepth;
  const float marginX = 0.3f * (r - l) / screenDistance;
  const float marginY = 0.3f * (t - b) / screenDistance;
  camera.xOverDMin = l / screenDistance - marginX;
  camera.xOverDMax = r / screenDistance + marginX;
  camera.yOverDMin = b / screenDistance - marginY;
  camera.yOverDMax = t / screenDistance + marginY;
  return camera;
}

ScreenCamera MakeLookAtCamera(const Vec3& eye,
                              const Vec3& target,
                              const Vec3& up,
                              float verticalFovDegrees,
                              int width,
                              int height,
                              float nearDepth) {
  const Vec3 forward = Normalize(Sub(target, eye));
  const Vec3 right = Normalize(Cross(forward, up));
  const Vec3 trueUp = Cross(right, forward);
  const float halfHeight =
      std::tan(0.5f * verticalFovDegrees * 3.14159265358979f / 180.0f);
  const float halfWidth =
      halfHeight * static_cast<float>(width) / static_cast<float>(height);
  const Vec3 center = Add(eye, forward);
  const Vec3 lowerLeft =
      Sub(Sub(center, Scale(right, halfWidth)), Scale(trueUp, halfHeight));
  const Vec3 lowerRight =
      Sub(Add(center, Scale(right, halfWidth)), Scale(trueUp, halfHeight));
  const Vec3 upperLeft =
      Add(Sub(center, Scale(right, halfWidth)), Scale(trueUp, halfHeight));
  return MakeScreenCamera(
      eye, lowerLeft, lowerRight, upperLeft, width, height, nearDepth);
}

ScreenCamera HalfHeightEyeCamera(const ScreenCamera& full, int rowParity) {
  if (full.height % 2 != 0) {
    throw std::invalid_argument("Interleaved stereo needs an even height");
  }
  ScreenCamera eye = full;
  eye.height = full.height / 2;
  eye.fy = 0.5f * full.fy;
  // Half-resolution row k samples full-resolution row 2k + rowParity.
  eye.cy = 0.5f * full.cy + 0.25f - 0.5f * static_cast<float>(rowParity);
  return eye;
}

float LodThresholdForPixels(const ScreenCamera& camera, float pixels) {
  return pixels / std::max(std::fabs(camera.fx), std::fabs(camera.fy));
}

std::vector<std::string> ColormapNames() {
  return {"inferno", "viridis", "coolwarm", "gray"};
}

TransferFunction MakeTransferFunction(const TransferSettings& settings) {
  if (!(settings.logDensityMax > settings.logDensityMin) ||
      !(settings.logTemperatureMax > settings.logTemperatureMin)) {
    throw std::invalid_argument("Transfer function ranges must be increasing");
  }
  TransferFunction tf{};
  tf.logDensityMin = settings.logDensityMin;
  tf.logDensityMax = settings.logDensityMax;
  tf.logTemperatureMin = settings.logTemperatureMin;
  tf.logTemperatureMax = settings.logTemperatureMax;
  tf.sigmaScale = settings.sigmaScale;
  tf.phaseMask = settings.phaseMask;

  const auto stops = ColormapStops(settings.colormap);
  for (int i = 0; i < kTransferTableSize; ++i) {
    const float u = static_cast<float>(i) / (kTransferTableSize - 1);
    const float logRho = settings.logDensityMin +
                         u * (settings.logDensityMax - settings.logDensityMin);
    tf.extinction[i] =
        settings.extinctionAtMax *
        std::pow(10.0f,
                 settings.densityExponent * (logRho - settings.logDensityMax));

    std::size_t k = 0;
    while (k + 2 < stops.size() && u > stops[k + 1].t) {
      ++k;
    }
    const ColorStop& s0 = stops[k];
    const ColorStop& s1 = stops[k + 1];
    const float f = std::clamp((u - s0.t) / (s1.t - s0.t), 0.0f, 1.0f);
    tf.color[i][0] =
        settings.brightness * SrgbToLinear(s0.r + f * (s1.r - s0.r));
    tf.color[i][1] =
        settings.brightness * SrgbToLinear(s0.g + f * (s1.g - s0.g));
    tf.color[i][2] =
        settings.brightness * SrgbToLinear(s0.b + f * (s1.b - s0.b));
  }
  return tf;
}

void Image::Resize(int w, int h) {
  width = w;
  height = h;
  rgba.assign(4 * static_cast<std::size_t>(w) * h, 0.0f);
}

Image InterleaveRows(const Image& parity0, const Image& parity1) {
  if (parity0.width != parity1.width || parity0.height != parity1.height) {
    throw std::invalid_argument("Eye images must have the same size");
  }
  Image out;
  out.Resize(parity0.width, 2 * parity0.height);
  const std::size_t rowFloats = 4 * static_cast<std::size_t>(parity0.width);
  for (int k = 0; k < parity0.height; ++k) {
    std::memcpy(
        out.Pixel(0, 2 * k), parity0.Pixel(0, k), rowFloats * sizeof(float));
    std::memcpy(out.Pixel(0, 2 * k + 1),
                parity1.Pixel(0, k),
                rowFloats * sizeof(float));
  }
  return out;
}

void WritePpm(const Image& image, const std::string& path, float exposure) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    throw std::runtime_error("Unable to write '" + path + "'");
  }
  out << "P6\n" << image.width << ' ' << image.height << "\n255\n";
  std::vector<unsigned char> row(3 * static_cast<std::size_t>(image.width));
  for (int y = 0; y < image.height; ++y) {
    for (int x = 0; x < image.width; ++x) {
      const float* p = image.Pixel(x, y);
      for (int c = 0; c < 3; ++c) {
        const float v = std::clamp(p[c] * exposure, 0.0f, 1.0f);
        row[3 * x + c] = static_cast<unsigned char>(
            std::pow(v, 1.0f / 2.2f) * 255.0f + 0.5f);
      }
    }
    out.write(reinterpret_cast<const char*>(row.data()),
              static_cast<std::streamsize>(row.size()));
  }
}

}  // namespace splatrender
