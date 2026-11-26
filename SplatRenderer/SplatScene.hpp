// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Loading splat files, building cameras and transfer functions, and simple
// image utilities for the splat renderers.

#ifndef SPLAT_RENDERER_SPLAT_SCENE_HPP
#define SPLAT_RENDERER_SPLAT_SCENE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "SplatTypes.hpp"

namespace splatrender {

struct SplatFileHeader {
  std::uint64_t recordCount = 0;
  std::uint64_t headerBytes = 0;
  std::uint64_t recordBytes = 0;
  double domainLo[3] = {0.0, 0.0, 0.0};
  double finestCellSize = 1.0;
  int topDepth = 0;
  std::vector<double> phaseEdges;
  std::string lengthUnit;
  std::string text;  // full JSON header
};

struct SplatScene {
  SplatFileHeader header;
  std::vector<SplatRecord> records;

  /// Data ranges over all records (ignoring the -300 "zero density" marker).
  float logDensityRange[2] = {0.0f, 0.0f};
  float logTemperatureRange[2] = {0.0f, 0.0f};
  /// Bounding box of record positions.
  Vec3 boundsLo = {0.0f, 0.0f, 0.0f};
  Vec3 boundsHi = {0.0f, 0.0f, 0.0f};
};

/// Read a file written by gaussian_export. Throws on malformed input.
SplatScene LoadSplatFile(const std::string& path);

/// Fill the data ranges and bounds of a scene from its records.
void ComputeSceneRanges(SplatScene& scene);

/// LOD parameters for a scene (LOD enabled, given eye and threshold).
LodParams MakeLodParams(const SplatFileHeader& header,
                        const Vec3& eye,
                        float threshold);

/// Off-axis camera from an eye position and three corners of the screen
/// rectangle (lower-left, lower-right, upper-left) in world coordinates.
ScreenCamera MakeScreenCamera(const Vec3& eye,
                              const Vec3& lowerLeft,
                              const Vec3& lowerRight,
                              const Vec3& upperLeft,
                              int width,
                              int height,
                              float nearDepth);

/// Conventional symmetric perspective camera.
ScreenCamera MakeLookAtCamera(const Vec3& eye,
                              const Vec3& target,
                              const Vec3& up,
                              float verticalFovDegrees,
                              int width,
                              int height,
                              float nearDepth);

/// Camera that renders only the rows of one eye of a row-interleaved stereo
/// display: half the height, sampling full-resolution rows
/// rowParity, rowParity + 2, ...
ScreenCamera HalfHeightEyeCamera(const ScreenCamera& full, int rowParity);

/// LOD threshold (radians) at which a record's bounding sphere spans about
/// `pixels` pixels near the center of the view.
float LodThresholdForPixels(const ScreenCamera& camera, float pixels);

struct TransferSettings {
  float logDensityMin = -27.0f;
  float logDensityMax = -22.0f;
  /// Extinction (per unit length) at logDensityMax.
  float extinctionAtMax = 1.0f;
  /// Extinction scales as density^exponent (1 = physical, linear in mass).
  float densityExponent = 1.0f;
  float logTemperatureMin = 4.0f;
  float logTemperatureMax = 7.0f;
  std::string colormap = "inferno";
  float brightness = 1.0f;
  float sigmaScale = 1.8f;
  std::uint32_t phaseMask = 0xffffffffu;
};

/// Names accepted by MakeTransferFunction.
std::vector<std::string> ColormapNames();

TransferFunction MakeTransferFunction(const TransferSettings& settings);

/// Linear RGBA float image, row 0 at the top.
struct Image {
  int width = 0;
  int height = 0;
  std::vector<float> rgba;

  void Resize(int w, int h);
  float* Pixel(int x, int y) {
    return &rgba[4 * (static_cast<std::size_t>(y) * width + x)];
  }
  const float* Pixel(int x, int y) const {
    return &rgba[4 * (static_cast<std::size_t>(y) * width + x)];
  }
};

/// Combine two half-height eye images into a row-interleaved stereo image.
/// Full-resolution row 2k + p comes from the eye with row parity p.
Image InterleaveRows(const Image& parity0, const Image& parity1);

/// Write an 8-bit binary PPM with exposure scaling and sRGB-like gamma.
void WritePpm(const Image& image, const std::string& path, float exposure);

}  // namespace splatrender

#endif  // SPLAT_RENDERER_SPLAT_SCENE_HPP
