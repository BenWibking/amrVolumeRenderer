// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#include <Common/VolumePainterViskores.hpp>

#include <Common/Color.hpp>
#include <Common/ImageRGBAFloatColorDepthSort.hpp>

#include <AMReX_Gpu.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Math.H>
#include <AMReX_GpuLaunch.H>
#include <AMReX_RealVect.H>
#include <AMReX_SmallMatrix.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Vec3 = amrex::RealVect;
using Vec4 = amrex::SmallMatrix<float, 4, 1>;
using Matrix4x4 = amrex::SmallMatrix<float, 4, 4>;

constexpr int kColorTableSize = 256;
constexpr float kSoftClipTolerance = 1e-5f;
constexpr float kPi = 3.14159265358979323846f;

struct ColorTableEntry {
  float r = 0.0f;
  float g = 0.0f;
  float b = 0.0f;
  float a = 0.0f;
};

float saturateSoftTail(float value, float clipStart, float rolloffEnd) {
  const float clampedEnd = std::max(clipStart, rolloffEnd);
  if (!(clampedEnd > clipStart + kSoftClipTolerance)) {
    return std::clamp(value, 0.0f, clampedEnd);
  }
  float clampedValue = std::clamp(value, 0.0f, clampedEnd);
  if (!(clampedValue > clipStart)) {
    return clampedValue;
  }
  if (!(clampedValue < clampedEnd)) {
    return clampedEnd;
  }
  const float normalized =
      (clampedValue - clipStart) / (clampedEnd - clipStart);
  const float smooth =
      normalized + normalized * normalized - normalized * normalized * normalized;
  return clipStart + (clampedEnd - clipStart) * smooth;
}

bool shouldApplySoftClip(const std::vector<float>& values,
                         float clipStart,
                         float rolloffEnd) {
  if (!(rolloffEnd > clipStart + kSoftClipTolerance)) {
    return false;
  }
  for (float value : values) {
    if (!std::isfinite(value)) {
      continue;
    }
    if (value > clipStart) {
      return true;
    }
  }
  return false;
}

void applySoftClip(std::vector<float>& values,
                   float clipStart,
                   float rolloffEnd) {
  if (!(rolloffEnd > clipStart + kSoftClipTolerance)) {
    for (float& value : values) {
      if (std::isfinite(value)) {
        value = std::clamp(value, 0.0f, rolloffEnd);
      }
    }
    return;
  }
  for (float& value : values) {
    if (!std::isfinite(value)) {
      continue;
    }
    value = saturateSoftTail(value, clipStart, rolloffEnd);
  }
}

float computeScaledAlpha(float baseAlpha,
                         float alphaScale,
                         float normalizationFactor) {
  const float scaledBase = std::clamp(baseAlpha * alphaScale, 0.0f, 1.0f);
  if (normalizationFactor <= 0.0f || scaledBase <= 0.0f) {
    return 0.0f;
  }
  if (scaledBase >= 1.0f) {
    return 1.0f;
  }
  const double transmittance =
      std::pow(1.0 - static_cast<double>(scaledBase),
               static_cast<double>(normalizationFactor));
  float scaledAlpha = static_cast<float>(1.0 - transmittance);
  if (!std::isfinite(scaledAlpha)) {
    scaledAlpha = scaledBase;
  }
  return std::clamp(scaledAlpha, 0.0f, 1.0f);
}

float jetComponent(float t, float shift) {
  return std::clamp(1.5f - std::fabs(4.0f * t - shift), 0.0f, 1.0f);
}

ColorTableEntry sampleColorMap(
    float value,
    float alphaScale,
    float normalizationFactor,
    const amrVolumeRenderer::volume::ColorMap& colorMap) {
  if (colorMap.empty()) {
    return {};
  }

  if (value <= colorMap.front().value) {
    const auto& first = colorMap.front();
    return {std::clamp(first.red, 0.0f, 1.0f),
            std::clamp(first.green, 0.0f, 1.0f),
            std::clamp(first.blue, 0.0f, 1.0f),
            computeScaledAlpha(first.alpha, alphaScale, normalizationFactor)};
  }

  for (std::size_t idx = 1; idx < colorMap.size(); ++idx) {
    const auto& left = colorMap[idx - 1];
    const auto& right = colorMap[idx];
    if (value <= right.value) {
      const float span = right.value - left.value;
      const float t = (span > 0.0f) ? (value - left.value) / span : 0.0f;
      const float r = std::clamp(left.red + (right.red - left.red) * t,
                                 0.0f,
                                 1.0f);
      const float g = std::clamp(left.green + (right.green - left.green) * t,
                                 0.0f,
                                 1.0f);
      const float b = std::clamp(left.blue + (right.blue - left.blue) * t,
                                 0.0f,
                                 1.0f);
      const float aLeft =
          computeScaledAlpha(left.alpha, alphaScale, normalizationFactor);
      const float aRight =
          computeScaledAlpha(right.alpha, alphaScale, normalizationFactor);
      const float a =
          std::clamp(aLeft + (aRight - aLeft) * t, 0.0f, 1.0f);
      return {r, g, b, a};
    }
  }

  const auto& last = colorMap.back();
  return {std::clamp(last.red, 0.0f, 1.0f),
          std::clamp(last.green, 0.0f, 1.0f),
          std::clamp(last.blue, 0.0f, 1.0f),
          computeScaledAlpha(last.alpha, alphaScale, normalizationFactor)};
}

std::vector<ColorTableEntry> buildColorTable(
    float alphaScale,
    float normalizationFactor,
    const std::pair<float, float>& scalarRange,
    const amrVolumeRenderer::volume::ColorMap* colorMap) {
  std::vector<ColorTableEntry> table(static_cast<std::size_t>(kColorTableSize));
  const float rangeMin = scalarRange.first;
  const float rangeMax = scalarRange.second;
  const float rangeSpan = rangeMax - rangeMin;
  const float invSpan = (rangeSpan != 0.0f) ? (1.0f / rangeSpan) : 1.0f;

  if (colorMap != nullptr && !colorMap->empty()) {
    for (int i = 0; i < kColorTableSize; ++i) {
      const float t = static_cast<float>(i) /
                      static_cast<float>(kColorTableSize - 1);
      const float value = rangeMin + rangeSpan * t;
      table[static_cast<std::size_t>(i)] =
          sampleColorMap(value, alphaScale, normalizationFactor, *colorMap);
    }
    return table;
  }

  const std::array<float, 6> positions = {
      0.0f, 0.15f, 0.35f, 0.6f, 0.85f, 1.0f};
  const std::array<float, 6> alphaValues = {
      0.05f, 0.15f, 0.22f, 0.3f, 0.38f, 0.5f};

  std::array<float, 6> alphaPositions;
  std::array<float, 6> scaledAlphas;
  for (std::size_t idx = 0; idx < positions.size(); ++idx) {
    alphaPositions[idx] = positions[idx] * rangeSpan + rangeMin;
    scaledAlphas[idx] = computeScaledAlpha(alphaValues[idx],
                                           alphaScale,
                                           normalizationFactor);
  }

  for (int i = 0; i < kColorTableSize; ++i) {
    const float t = static_cast<float>(i) /
                    static_cast<float>(kColorTableSize - 1);
    const float value = rangeMin + rangeSpan * t;
    const float normalized =
        std::clamp((value - rangeMin) * invSpan, 0.0f, 1.0f);
    const float r = jetComponent(normalized, 3.0f);
    const float g = jetComponent(normalized, 2.0f);
    const float b = jetComponent(normalized, 1.0f);

    float alpha = scaledAlphas.back();
    if (value <= alphaPositions.front()) {
      alpha = scaledAlphas.front();
    } else {
      for (std::size_t idx = 1; idx < alphaPositions.size(); ++idx) {
        if (value <= alphaPositions[idx]) {
          const float span = alphaPositions[idx] - alphaPositions[idx - 1];
          const float localT =
              (span > 0.0f) ? (value - alphaPositions[idx - 1]) / span : 0.0f;
          alpha = std::clamp(scaledAlphas[idx - 1] +
                                 (scaledAlphas[idx] - scaledAlphas[idx - 1]) *
                                     localT,
                             0.0f,
                             1.0f);
          break;
        }
      }
    }

    table[static_cast<std::size_t>(i)] = {r, g, b, alpha};
  }

  return table;
}

Vec3 safeNormalize(const Vec3& input) {
  const amrex::Real length = input.vectorLength();
  if (length > 0.0 && std::isfinite(static_cast<double>(length))) {
    return input / length;
  }
  return Vec3(0.0, 0.0, -1.0);
}

Matrix4x4 makeViewMatrix(const Vec3& eye,
                         const Vec3& lookAt,
                         const Vec3& up) {
  const Vec3 forward = safeNormalize(lookAt - eye);
  Vec3 right = forward.crossProduct(up);
  const amrex::Real rightLength = right.vectorLength();
  if (rightLength > 0.0 && std::isfinite(static_cast<double>(rightLength))) {
    right /= rightLength;
  } else {
    right = Vec3(1.0, 0.0, 0.0);
  }
  const Vec3 upOrtho = right.crossProduct(forward);

  Matrix4x4 view = Matrix4x4::Identity();
  view(0, 0) = static_cast<float>(right[0]);
  view(1, 0) = static_cast<float>(right[1]);
  view(2, 0) = static_cast<float>(right[2]);
  view(3, 0) = static_cast<float>(-right.dotProduct(eye));

  view(0, 1) = static_cast<float>(upOrtho[0]);
  view(1, 1) = static_cast<float>(upOrtho[1]);
  view(2, 1) = static_cast<float>(upOrtho[2]);
  view(3, 1) = static_cast<float>(-upOrtho.dotProduct(eye));

  view(0, 2) = static_cast<float>(-forward[0]);
  view(1, 2) = static_cast<float>(-forward[1]);
  view(2, 2) = static_cast<float>(-forward[2]);
  view(3, 2) = static_cast<float>(forward.dotProduct(eye));

  view(0, 3) = 0.0f;
  view(1, 3) = 0.0f;
  view(2, 3) = 0.0f;
  view(3, 3) = 1.0f;

  return view;
}

Matrix4x4 makePerspectiveMatrix(float fovYDegrees,
                                float aspect,
                                float nearPlane,
                                float farPlane) {
  Matrix4x4 matrix = Matrix4x4::Identity();

  const float fovTangent = std::tan(fovYDegrees * kPi / 180.0f * 0.5f);
  const float size = nearPlane * fovTangent;
  const float left = -size * aspect;
  const float right = size * aspect;
  const float bottom = -size;
  const float top = size;

  matrix(0, 0) = 2.0f * nearPlane / (right - left);
  matrix(1, 1) = 2.0f * nearPlane / (top - bottom);
  matrix(0, 2) = (right + left) / (right - left);
  matrix(1, 2) = (top + bottom) / (top - bottom);
  matrix(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
  matrix(3, 2) = -1.0f;
  matrix(2, 3) = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
  matrix(3, 3) = 0.0f;

  return matrix;
}

}  // namespace

VolumePainterViskores::VolumePainterViskores() = default;
VolumePainterViskores::~VolumePainterViskores() = default;

void VolumePainterViskores::paint(
    const amrVolumeRenderer::volume::AmrBox& box,
    const amrVolumeRenderer::volume::VolumeBounds& bounds,
    const std::pair<float, float>& scalarRange,
    int rank,
    int numProcs,
    float boxTransparency,
    int antialiasing,
    float referenceSampleDistance,
    ImageFull& image,
    const amrVolumeRenderer::volume::CameraParameters& camera,
    const amrVolumeRenderer::volume::ColorMap* colorMap) {
  static_cast<void>(numProcs);
  static_cast<void>(antialiasing);

  auto* depthSortedImage = dynamic_cast<ImageRGBAFloatColorDepthSort*>(&image);
  if (depthSortedImage == nullptr) {
    throw std::runtime_error(
        "VolumePainterViskores expects ImageRGBAFloatColorDepthSort images.");
  }

  try {
    const Vec3 span = box.maxCorner - box.minCorner;
    Vec3 spacing(0.0);
    if (box.cellDimensions[0] > 0) {
      spacing[0] = span[0] /
                   static_cast<amrex::Real>(box.cellDimensions[0]);
    }
    if (box.cellDimensions[1] > 0) {
      spacing[1] = span[1] /
                   static_cast<amrex::Real>(box.cellDimensions[1]);
    }
    if (box.cellDimensions[2] > 0) {
      spacing[2] = span[2] /
                   static_cast<amrex::Real>(box.cellDimensions[2]);
    }

    float minSpacing = std::numeric_limits<float>::max();
    for (int component = 0; component < 3; ++component) {
      const float value = static_cast<float>(spacing[component]);
      if (value > 0.0f && value < minSpacing && std::isfinite(value)) {
        minSpacing = value;
      }
    }
    if (!(minSpacing > 0.0f && std::isfinite(minSpacing))) {
      const Vec3 fallbackSpan = bounds.maxCorner - bounds.minCorner;
      const float fallbackMin = static_cast<float>(
          std::min({fallbackSpan[0], fallbackSpan[1], fallbackSpan[2]}));
      minSpacing = std::max(1e-4f, fallbackMin * 0.01f);
    }

    const float sampleDistance = std::max(minSpacing * 0.5f, 1e-5f);
    float referenceDistance = referenceSampleDistance;
    if (!(referenceDistance > 0.0f &&
          std::isfinite(referenceDistance))) {
      referenceDistance = sampleDistance;
    }
    float normalizationFactor = sampleDistance / referenceDistance;
    if (!std::isfinite(normalizationFactor)) {
      normalizationFactor = 1.0f;
    }
    normalizationFactor = std::max(normalizationFactor, 0.0f);

    const float alphaScale =
        std::clamp(1.0f - boxTransparency, 0.0f, 1.0f);

    std::vector<ColorTableEntry> colorTable =
        buildColorTable(alphaScale, normalizationFactor, scalarRange, colorMap);

    const float clipStart = std::clamp(scalarRange.second, 0.0f, 1.0f);
    std::vector<float> adjustedValues;
    const bool applyClip =
        shouldApplySoftClip(box.cellValues, clipStart, 1.0f);
    if (applyClip) {
      adjustedValues = box.cellValues;
      applySoftClip(adjustedValues, clipStart, 1.0f);
    }

    const std::vector<float>& sourceValues =
        applyClip ? adjustedValues : box.cellValues;

    if (sourceValues.empty()) {
      depthSortedImage->clear();
      return;
    }

    const int width = image.getWidth();
    const int height = image.getHeight();
    if (width <= 0 || height <= 0) {
      return;
    }

    const float aspect =
        static_cast<float>(width) / static_cast<float>(std::max(height, 1));
    const Matrix4x4 view =
        makeViewMatrix(camera.eye, camera.lookAt, camera.up);
    const Matrix4x4 projection = makePerspectiveMatrix(
        camera.fovYDegrees, aspect, camera.nearPlane, camera.farPlane);

    const Vec3 forward = safeNormalize(camera.lookAt - camera.eye);
    Vec3 right = forward.crossProduct(camera.up);
    const amrex::Real rightLength = right.vectorLength();
    if (rightLength > 0.0 && std::isfinite(static_cast<double>(rightLength))) {
      right /= rightLength;
    } else {
      right = Vec3(1.0, 0.0, 0.0);
    }
    const Vec3 up = right.crossProduct(forward);

    const amrex::GpuArray<float, 3> eye = {
        static_cast<float>(camera.eye[0]),
        static_cast<float>(camera.eye[1]),
        static_cast<float>(camera.eye[2])};
    const amrex::GpuArray<float, 3> basisForward = {
        static_cast<float>(forward[0]),
        static_cast<float>(forward[1]),
        static_cast<float>(forward[2])};
    const amrex::GpuArray<float, 3> basisRight = {
        static_cast<float>(right[0]),
        static_cast<float>(right[1]),
        static_cast<float>(right[2])};
    const amrex::GpuArray<float, 3> basisUp = {
        static_cast<float>(up[0]),
        static_cast<float>(up[1]),
        static_cast<float>(up[2])};

    const amrex::GpuArray<float, 3> minCorner = {
        static_cast<float>(box.minCorner[0]),
        static_cast<float>(box.minCorner[1]),
        static_cast<float>(box.minCorner[2])};
    const amrex::GpuArray<float, 3> maxCorner = {
        static_cast<float>(box.maxCorner[0]),
        static_cast<float>(box.maxCorner[1]),
        static_cast<float>(box.maxCorner[2])};

    const int nx = box.cellDimensions[0];
    const int ny = box.cellDimensions[1];
    const int nz = box.cellDimensions[2];
    if (nx <= 0 || ny <= 0 || nz <= 0) {
      depthSortedImage->clear();
      return;
    }

    const float dx =
        (nx > 0) ? (maxCorner[0] - minCorner[0]) / static_cast<float>(nx)
                 : 1.0f;
    const float dy =
        (ny > 0) ? (maxCorner[1] - minCorner[1]) / static_cast<float>(ny)
                 : 1.0f;
    const float dz =
        (nz > 0) ? (maxCorner[2] - minCorner[2]) / static_cast<float>(nz)
                 : 1.0f;

    const float extentMag = std::sqrt(
        (maxCorner[0] - minCorner[0]) * (maxCorner[0] - minCorner[0]) +
        (maxCorner[1] - minCorner[1]) * (maxCorner[1] - minCorner[1]) +
        (maxCorner[2] - minCorner[2]) * (maxCorner[2] - minCorner[2]));
    const float meshEpsilon = extentMag * 0.0001f;

    amrex::Gpu::DeviceVector<float> deviceValues(sourceValues.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          sourceValues.begin(),
                          sourceValues.end(),
                          deviceValues.begin());

    std::vector<float> hostColorTable(
        static_cast<std::size_t>(kColorTableSize) * 4);
    for (int i = 0; i < kColorTableSize; ++i) {
      const std::size_t base = static_cast<std::size_t>(i) * 4;
      hostColorTable[base + 0] = colorTable[static_cast<std::size_t>(i)].r;
      hostColorTable[base + 1] = colorTable[static_cast<std::size_t>(i)].g;
      hostColorTable[base + 2] = colorTable[static_cast<std::size_t>(i)].b;
      hostColorTable[base + 3] = colorTable[static_cast<std::size_t>(i)].a;
    }

    amrex::Gpu::DeviceVector<float> deviceColorTable(
        hostColorTable.size());
    amrex::Gpu::copyAsync(amrex::Gpu::hostToDevice,
                          hostColorTable.begin(),
                          hostColorTable.end(),
                          deviceColorTable.begin());

    const int pixelCount = width * height;
    amrex::Gpu::DeviceVector<float> deviceColor(
        static_cast<std::size_t>(pixelCount) * 4);
    amrex::Gpu::DeviceVector<float> deviceDepth(
        static_cast<std::size_t>(pixelCount));

    const float rangeMin = scalarRange.first;
    const float rangeMax = scalarRange.second;
    float inverseRange = 1.0f;
    if (rangeMax != rangeMin) {
      inverseRange = 1.0f / (rangeMax - rangeMin);
    }

    const float tanHalfFov =
        std::tan(camera.fovYDegrees * 0.5f * kPi / 180.0f);
    const float invWidth = 1.0f / static_cast<float>(width);
    const float invHeight = 1.0f / static_cast<float>(height);

    const float* valuesPtr = deviceValues.data();
    const float* tablePtr = deviceColorTable.data();
    float* outColor = deviceColor.data();
    float* outDepth = deviceDepth.data();

    amrex::ParallelFor(
        pixelCount,
        [=] AMREX_GPU_DEVICE(int index) noexcept {
          const int px = index % width;
          const int py = index / width;

          const float ndcX =
              (static_cast<float>(px) + 0.5f) * invWidth * 2.0f - 1.0f;
          const float ndcY =
              (static_cast<float>(py) + 0.5f) * invHeight * 2.0f - 1.0f;

          const float planeX = ndcX * tanHalfFov * aspect;
          const float planeY = ndcY * tanHalfFov;

          float dirX = basisForward[0] + planeX * basisRight[0] +
                       planeY * basisUp[0];
          float dirY = basisForward[1] + planeX * basisRight[1] +
                       planeY * basisUp[1];
          float dirZ = basisForward[2] + planeX * basisRight[2] +
                       planeY * basisUp[2];

          const float dirLenSq = dirX * dirX + dirY * dirY + dirZ * dirZ;
          const float dirLength =
              (dirLenSq > 0.0f)
                  ? (1.0f / amrex::Math::rsqrt(dirLenSq))
                  : 0.0f;
          if (dirLength > 0.0f) {
            const float inv = 1.0f / dirLength;
            dirX *= inv;
            dirY *= inv;
            dirZ *= inv;
          }

          float tmin = -std::numeric_limits<float>::infinity();
          float tmax = std::numeric_limits<float>::infinity();

          const float originX = eye[0];
          const float originY = eye[1];
          const float originZ = eye[2];

          auto updateBounds = [&](float origin,
                                  float direction,
                                  float minVal,
                                  float maxVal) {
            if (amrex::Math::abs(direction) < 1e-8f) {
              if (origin < minVal || origin > maxVal) {
                tmin = std::numeric_limits<float>::infinity();
                tmax = -std::numeric_limits<float>::infinity();
              }
              return;
            }
            const float invDir = 1.0f / direction;
            float t1 = (minVal - origin) * invDir;
            float t2 = (maxVal - origin) * invDir;
            if (t1 > t2) {
              float tmp = t1;
              t1 = t2;
              t2 = tmp;
            }
            tmin = (tmin > t1) ? tmin : t1;
            tmax = (tmax < t2) ? tmax : t2;
          };

          updateBounds(originX, dirX, minCorner[0], maxCorner[0]);
          updateBounds(originY, dirY, minCorner[1], maxCorner[1]);
          updateBounds(originZ, dirZ, minCorner[2], maxCorner[2]);

          if (!(tmax >= tmin)) {
            outColor[index * 4 + 0] = 0.0f;
            outColor[index * 4 + 1] = 0.0f;
            outColor[index * 4 + 2] = 0.0f;
            outColor[index * 4 + 3] = 0.0f;
            outDepth[index] = std::numeric_limits<float>::infinity();
            return;
          }

          float distance = tmin + meshEpsilon;
          if (distance < 0.0f) {
            distance = meshEpsilon;
          }

          float accumR = 0.0f;
          float accumG = 0.0f;
          float accumB = 0.0f;
          float accumA = 0.0f;

          auto isInside = [&](float x, float y, float z) {
            return !(x < minCorner[0] || x > maxCorner[0] ||
                     y < minCorner[1] || y > maxCorner[1] ||
                     z < minCorner[2] || z > maxCorner[2]);
          };

          float posX = originX + dirX * distance;
          float posY = originY + dirY * distance;
          float posZ = originZ + dirZ * distance;
          while (distance < tmax && !isInside(posX, posY, posZ)) {
            distance += sampleDistance;
            posX = originX + dirX * distance;
            posY = originY + dirY * distance;
            posZ = originZ + dirZ * distance;
          }

          while (distance < tmax && accumA < 1.0f) {
            if (!isInside(posX, posY, posZ)) {
              distance += sampleDistance;
              posX = originX + dirX * distance;
              posY = originY + dirY * distance;
              posZ = originZ + dirZ * distance;
              continue;
            }

            float fx = (posX - minCorner[0]) / dx;
            float fy = (posY - minCorner[1]) / dy;
            float fz = (posZ - minCorner[2]) / dz;

            int i = static_cast<int>(amrex::Math::floor(fx));
            int j = static_cast<int>(amrex::Math::floor(fy));
            int k = static_cast<int>(amrex::Math::floor(fz));
            if (i < 0) {
              i = 0;
            } else if (i >= nx) {
              i = nx - 1;
            }
            if (j < 0) {
              j = 0;
            } else if (j >= ny) {
              j = ny - 1;
            }
            if (k < 0) {
              k = 0;
            } else if (k >= nz) {
              k = nz - 1;
            }

            if (i >= 0 && i < nx && j >= 0 && j < ny && k >= 0 && k < nz) {
              const int flat =
                  (k * ny + j) * nx + i;
              const float scalar = valuesPtr[flat];
              float normalized = (scalar - rangeMin) * inverseRange;
              normalized = (normalized < 0.0f) ? 0.0f : normalized;
              normalized = (normalized > 1.0f) ? 1.0f : normalized;
              int idx =
                  static_cast<int>(normalized *
                                   static_cast<float>(kColorTableSize - 1));
              idx = (idx < 0) ? 0 : idx;
              idx = (idx > kColorTableSize - 1) ? (kColorTableSize - 1) : idx;
              const int base = idx * 4;
              const float sampleR = tablePtr[base + 0];
              const float sampleG = tablePtr[base + 1];
              const float sampleB = tablePtr[base + 2];
              const float sampleA = tablePtr[base + 3];
              const float alpha = sampleA * (1.0f - accumA);
              accumR += sampleR * alpha;
              accumG += sampleG * alpha;
              accumB += sampleB * alpha;
              accumA += alpha;
            }

            distance += sampleDistance;
            posX = originX + dirX * distance;
            posY = originY + dirY * distance;
            posZ = originZ + dirZ * distance;
          }

          accumR = (accumR > 1.0f) ? 1.0f : accumR;
          accumG = (accumG > 1.0f) ? 1.0f : accumG;
          accumB = (accumB > 1.0f) ? 1.0f : accumB;
          accumA = (accumA > 1.0f) ? 1.0f : accumA;

          outColor[index * 4 + 0] = accumR;
          outColor[index * 4 + 1] = accumG;
          outColor[index * 4 + 2] = accumB;
          outColor[index * 4 + 3] = accumA;

          float depth = std::numeric_limits<float>::infinity();
          if (accumA > 0.0f) {
            const float entryX = originX + dirX * tmin;
            const float entryY = originY + dirY * tmin;
            const float entryZ = originZ + dirZ * tmin;
            depth = (entryX - eye[0]) * basisForward[0] +
                    (entryY - eye[1]) * basisForward[1] +
                    (entryZ - eye[2]) * basisForward[2];
          }
          outDepth[index] = depth;
        });

    amrex::Gpu::streamSynchronize();

    std::vector<float> hostColor(
        static_cast<std::size_t>(pixelCount) * 4);
    std::vector<float> hostDepth(static_cast<std::size_t>(pixelCount));
    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                          deviceColor.begin(),
                          deviceColor.end(),
                          hostColor.begin());
    amrex::Gpu::copyAsync(amrex::Gpu::deviceToHost,
                          deviceDepth.begin(),
                          deviceDepth.end(),
                          hostDepth.begin());
    amrex::Gpu::streamSynchronize();

    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const int pixelIndex = y * width + x;
        const std::size_t base =
            static_cast<std::size_t>(pixelIndex) * 4;
        const float red = std::clamp(hostColor[base + 0], 0.0f, 1.0f);
        const float green = std::clamp(hostColor[base + 1], 0.0f, 1.0f);
        const float blue = std::clamp(hostColor[base + 2], 0.0f, 1.0f);
        const float alpha = std::clamp(hostColor[base + 3], 0.0f, 1.0f);
        depthSortedImage->setColor(x, y, Color(red, green, blue, alpha));
        float depth = hostDepth[static_cast<std::size_t>(pixelIndex)];
        if (!std::isfinite(depth) || alpha <= 0.0f) {
          depth = std::numeric_limits<float>::infinity();
        }
        depthSortedImage->setDepthHint(x, y, depth);
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "VolumePainterViskores error on rank " << rank << ": "
              << error.what() << std::endl;
    throw;
  }
}
