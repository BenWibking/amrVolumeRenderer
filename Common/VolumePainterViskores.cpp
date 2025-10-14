// miniGraphics is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Copyright (c) 2017
// National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
// the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
// certain rights in this software.

#include <Common/VolumePainterViskores.hpp>

#include <Common/Color.hpp>
#include <Common/ImageRGBAFloatColorDepthSort.hpp>

#if defined(MINIGRAPHICS_ENABLE_VISKORES)

#include <viskores/Math.h>
#include <viskores/Matrix.h>
#include <viskores/VectorAnalysis.h>

#include <viskores/Types.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/DataSetBuilderUniform.h>
#include <viskores/cont/Initialize.h>
#include <viskores/Range.h>
#include <viskores/rendering/Actor.h>
#include <viskores/rendering/Camera.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/rendering/Color.h>
#include <viskores/rendering/MapperVolume.h>
#include <viskores/rendering/MatrixHelpers.h>
#include <viskores/rendering/Scene.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace {

using Vec3 = viskores::Vec3f_32;
using Vec4 = viskores::Vec4f_32;
using Matrix4x4 = viskores::Matrix<viskores::Float32, 4, 4>;

constexpr const char kDensityFieldName[] = "density";

class View3DNoColorBar : public viskores::rendering::View3D {
 public:
  using viskores::rendering::View3D::View3D;

  void RenderScreenAnnotations() override {}
  void RenderWorldAnnotations() override {}
};

}  // namespace

VolumePainterViskores::VolumePainterViskores() {
  static bool viskoresInitialized = false;
  if (!viskoresInitialized) {
    viskores::cont::Initialize();
    viskoresInitialized = true;
  }
}

VolumePainterViskores::~VolumePainterViskores() = default;

void VolumePainterViskores::paint(
    const minigraphics::volume::AmrBox& box,
    const minigraphics::volume::VolumeBounds& bounds,
    const std::pair<float, float>& scalarRange,
    int rank,
    int numProcs,
    float boxTransparency,
    int antialiasing,
    float referenceSampleDistance,
    ImageFull& image,
    const minigraphics::volume::CameraParameters& camera,
    const minigraphics::volume::ColorMap* colorMap) {
  try {
    viskores::cont::DataSet dataset = this->boxToDataSet(box);

    const Vec3 span = box.maxCorner - box.minCorner;
    Vec3 spacing(0.0f);
    Vec3 fallbackSpan = bounds.maxCorner - bounds.minCorner;
    if (box.cellDimensions[0] > 0) {
      spacing[0] =
          span[0] / static_cast<float>(box.cellDimensions[0]);
    }
    if (box.cellDimensions[1] > 0) {
      spacing[1] =
          span[1] / static_cast<float>(box.cellDimensions[1]);
    }
    if (box.cellDimensions[2] > 0) {
      spacing[2] =
          span[2] / static_cast<float>(box.cellDimensions[2]);
    }

    float minSpacing = std::numeric_limits<float>::max();
    for (int component = 0; component < 3; ++component) {
      if (spacing[component] > 0.0f &&
          spacing[component] < minSpacing &&
          std::isfinite(spacing[component])) {
        minSpacing = spacing[component];
      }
    }
    if (!(minSpacing > 0.0f && std::isfinite(minSpacing))) {
      const float fallbackMin =
          std::min({fallbackSpan[0], fallbackSpan[1], fallbackSpan[2]});
      minSpacing = std::max(1e-4f, fallbackMin * 0.01f);
    }

    const float sampleDistance =
        std::max(minSpacing * 0.5f, 1e-5f);
    float referenceDistance = referenceSampleDistance;
    if (!(referenceDistance > 0.0f &&
          std::isfinite(referenceDistance))) {
      referenceDistance = sampleDistance;
    }
    float normalizationFactor = sampleDistance / referenceDistance;
    if (!(std::isfinite(normalizationFactor))) {
      normalizationFactor = 1.0f;
    }
    normalizationFactor = std::max(normalizationFactor, 0.0f);

    viskores::cont::ColorTable colorTable =
        this->buildColorTable(1.0f - boxTransparency,
                              normalizationFactor,
                              scalarRange,
                              colorMap);

    viskores::rendering::CanvasRayTracer localCanvas(image.getWidth(),
                                                     image.getHeight());
    viskores::rendering::MapperVolume localMapper;
    localMapper.SetSampleDistance(sampleDistance);
    static_cast<void>(antialiasing);  // Supersampling is handled in screen space.
    localMapper.SetCompositeBackground(false);
    localMapper.SetActiveColorTable(colorTable);

    viskores::rendering::Scene localScene;
    viskores::rendering::Actor actor(dataset.GetCellSet(),
                                     dataset.GetCoordinateSystem(),
                                     dataset.GetField(kDensityFieldName),
                                     colorTable);
    actor.SetScalarRange(
        viskores::Range(scalarRange.first, scalarRange.second));
    localScene.AddActor(actor);

    View3DNoColorBar localView(localScene,
                               localMapper,
                               localCanvas,
                               viskores::rendering::Color(0.0f,
                                                          0.0f,
                                                          0.0f,
                                                          1.0f));
    localView.SetWorldAnnotationsEnabled(false);

    this->setupCamera(localView, camera);

    localView.Paint();

    this->canvasToImage(localCanvas, image);

  } catch (const std::exception& error) {
    std::cerr << "VolumePainterViskores error on rank " << rank << ": "
              << error.what() << std::endl;
    throw;
  }
}

viskores::cont::DataSet VolumePainterViskores::boxToDataSet(
    const minigraphics::volume::AmrBox& box) const {
  const Vec3 minCorner = box.minCorner;
  const Vec3 maxCorner = box.maxCorner;
  const Vec3 span = maxCorner - minCorner;
  const viskores::Id3 cellDims = box.cellDimensions;

  if (cellDims[0] <= 0 || cellDims[1] <= 0 || cellDims[2] <= 0) {
    throw std::invalid_argument(
        "AMR box dimensions must be positive along every axis.");
  }

  const std::size_t expectedValueCount =
      static_cast<std::size_t>(cellDims[0]) *
      static_cast<std::size_t>(cellDims[1]) *
      static_cast<std::size_t>(cellDims[2]);
  if (box.cellValues.size() != expectedValueCount) {
    std::ostringstream message;
    message << "AMR box provided " << box.cellValues.size()
            << " cell values but expected " << expectedValueCount;
    throw std::invalid_argument(message.str());
  }

  viskores::cont::DataSetBuilderUniform builder;
  const viskores::Id3 pointDims(cellDims[0] + 1,
                                cellDims[1] + 1,
                                cellDims[2] + 1);
  const Vec3 origin(minCorner[0], minCorner[1], minCorner[2]);

  const auto safeSpacing = [&](float component, viskores::Id cells) -> float {
    if (cells <= 0) {
      return 1.0f;
    }
    const float length = std::max(component, 1e-5f);
    return length / static_cast<float>(cells);
  };

  const Vec3 spacing(safeSpacing(span[0], cellDims[0]),
                     safeSpacing(span[1], cellDims[1]),
                     safeSpacing(span[2], cellDims[2]));

  viskores::cont::DataSet dataset = builder.Create(pointDims, origin, spacing);

  dataset.AddCellField(
      kDensityFieldName,
      viskores::cont::make_ArrayHandle(box.cellValues,
                                       viskores::CopyFlag::On));

  return dataset;
}

viskores::cont::ColorTable VolumePainterViskores::buildColorTable(
    float alphaScale,
    float normalizationFactor,
    const std::pair<float, float>& scalarRange,
    const minigraphics::volume::ColorMap* colorMap) const {
  const float clampedScale = std::clamp(alphaScale, 0.0f, 1.0f);
  float clampedFactor = normalizationFactor;
  if (!(std::isfinite(clampedFactor))) {
    clampedFactor = 1.0f;
  }
  clampedFactor = std::max(clampedFactor, 0.0f);
  const double rangeMin = static_cast<double>(scalarRange.first);
  const double rangeMax = static_cast<double>(scalarRange.second);

  const auto computeScaledAlpha = [&](float baseAlpha) -> float {
    const float scaledBase = std::clamp(baseAlpha * clampedScale, 0.0f, 1.0f);
    if (clampedFactor <= 0.0f || scaledBase <= 0.0f) {
      return 0.0f;
    }
    if (scaledBase >= 1.0f) {
      return 1.0f;
    }
    const double transmittance =
        std::pow(1.0 - static_cast<double>(scaledBase),
                 static_cast<double>(clampedFactor));
    float scaledAlpha =
        static_cast<float>(1.0 - transmittance);
    if (!std::isfinite(scaledAlpha)) {
      scaledAlpha = scaledBase;
    }
    return std::clamp(scaledAlpha, 0.0f, 1.0f);
  };

  if (colorMap != nullptr && !colorMap->empty()) {
    viskores::cont::ColorTable customTable;
    for (const auto& point : *colorMap) {
      const double position = static_cast<double>(point.value);
      const Vec3 rgb(std::clamp(point.red, 0.0f, 1.0f),
                     std::clamp(point.green, 0.0f, 1.0f),
                     std::clamp(point.blue, 0.0f, 1.0f));
      const float scaledAlpha = computeScaledAlpha(point.alpha);
      customTable.AddPoint(position, rgb);
      customTable.AddPointAlpha(position, scaledAlpha);
    }
    return customTable;
  }

  viskores::cont::ColorTable colorTable(
      viskores::cont::ColorTable::Preset::Jet);
  colorTable.ClearAlpha();

  const std::array<float, 6> positions = {
      0.0f, 0.15f, 0.35f, 0.6f, 0.85f, 1.0f};
  const std::array<float, 6> alphaValues = {
      0.05f, 0.15f, 0.22f, 0.3f, 0.38f, 0.5f};

  for (std::size_t i = 0; i < positions.size(); ++i) {
    const float scaledAlpha = computeScaledAlpha(alphaValues[i]);
    const double position =
        static_cast<double>(positions[i]) * (rangeMax - rangeMin) + rangeMin;
    colorTable.AddPointAlpha(position, scaledAlpha);
  }
  colorTable.RescaleToRange(viskores::Range(rangeMin, rangeMax));

  return colorTable;
}

void VolumePainterViskores::setupCamera(
    viskores::rendering::View3D& targetView,
    const minigraphics::volume::CameraParameters& cameraParams) {
  viskores::rendering::Camera camera;

  camera.SetPosition(cameraParams.eye);
  camera.SetLookAt(cameraParams.lookAt);
  camera.SetViewUp(cameraParams.up);
  camera.SetFieldOfView(cameraParams.fovYDegrees);
  camera.SetClippingRange(cameraParams.nearPlane, cameraParams.farPlane);

  targetView.SetCamera(camera);
}

void VolumePainterViskores::canvasToImage(
    const viskores::rendering::Canvas& viskoresCanvas,
    ImageFull& image) const {
  auto* depthSortedImage =
      dynamic_cast<ImageRGBAFloatColorDepthSort*>(&image);
  if (depthSortedImage == nullptr) {
    throw std::runtime_error(
        "VolumePainterViskores expects ImageRGBAFloatColorDepthSort images.");
  }

  const auto colorPortal = viskoresCanvas.GetColorBuffer().ReadPortal();
  const auto depthPortal = viskoresCanvas.GetDepthBuffer().ReadPortal();

  const int width = image.getWidth();
  const int height = image.getHeight();
  const auto clampUnit = [](float value) {
    return std::clamp(value, 0.0f, 1.0f);
  };

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int pixelIndex = y * width + x;
      const Vec4 color = colorPortal.Get(pixelIndex);
      const float alpha = clampUnit(static_cast<float>(color[3]));
      const float red = clampUnit(static_cast<float>(color[0]));
      const float green = clampUnit(static_cast<float>(color[1]));
      const float blue = clampUnit(static_cast<float>(color[2]));
      depthSortedImage->setColor(x, y, Color(red, green, blue, alpha));
      float depth = depthPortal.Get(pixelIndex);
      if (!std::isfinite(depth) || alpha <= 0.0f) {
        depth = std::numeric_limits<float>::infinity();
      }
      depthSortedImage->setDepthHint(x, y, depth);
    }
  }
}

#endif  // MINIGRAPHICS_ENABLE_VISKORES
