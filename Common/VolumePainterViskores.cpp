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
    const std::vector<minigraphics::volume::VolumeBox>& boxes,
    const minigraphics::volume::VolumeBounds& bounds,
    int samplesPerAxis,
    int rank,
    int numProcs,
    float boxTransparency,
    ImageFull& image,
    const minigraphics::volume::CameraParameters& camera,
    const Vec3* colorOverride) {
  try {
    viskores::cont::DataSet dataset =
        this->boxesToDataSet(boxes, bounds, samplesPerAxis);
    viskores::cont::ColorTable colorTable =
        this->buildColorTable(numProcs, 1.0f - boxTransparency);

    const Vec3 span = bounds.maxCorner - bounds.minCorner;
    const float minSpan = std::min({span[0], span[1], span[2]});
    const float minSpacing =
        std::max(1e-4f,
                 minSpan / static_cast<float>(std::max(samplesPerAxis - 1, 1)));

    viskores::rendering::CanvasRayTracer localCanvas(image.getWidth(),
                                                     image.getHeight());
    viskores::rendering::MapperVolume localMapper;
    localMapper.SetSampleDistance(minSpacing * 0.5f);
    localMapper.SetCompositeBackground(false);
    localMapper.SetActiveColorTable(colorTable);

    viskores::rendering::Scene localScene;
    viskores::rendering::Actor actor(dataset.GetCellSet(),
                                     dataset.GetCoordinateSystem(),
                                     dataset.GetField(kDensityFieldName),
                                     colorTable);
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

    this->canvasToImage(localCanvas, image, colorOverride);

    if (rank == 0) {
      std::cout << "VolumePainterViskores: Rendered volume with "
                << samplesPerAxis << "^3 samples across " << numProcs
                << " ranks" << std::endl;
    }
  } catch (const std::exception& error) {
    std::cerr << "VolumePainterViskores error on rank " << rank << ": "
              << error.what() << std::endl;
    throw;
  }
}

viskores::cont::DataSet VolumePainterViskores::boxesToDataSet(
    const std::vector<minigraphics::volume::VolumeBox>& boxes,
    const minigraphics::volume::VolumeBounds& bounds,
    int samplesPerAxis) const {
  const Vec3 minCorner = bounds.minCorner;
  const Vec3 maxCorner = bounds.maxCorner;
  const Vec3 span = maxCorner - minCorner;

  viskores::cont::DataSetBuilderUniform builder;
  const viskores::Id3 dimensions(samplesPerAxis,
                                 samplesPerAxis,
                                 samplesPerAxis);
  const Vec3 origin(minCorner[0], minCorner[1], minCorner[2]);

  const auto safeSpacing = [&](float component) -> float {
    if (samplesPerAxis <= 1) {
      return 1.0f;
    }
    const float length = std::max(component, 1e-5f);
    return length / static_cast<float>(samplesPerAxis - 1);
  };

  const Vec3 spacing(safeSpacing(span[0]),
                     safeSpacing(span[1]),
                     safeSpacing(span[2]));

  viskores::cont::DataSet dataset = builder.Create(dimensions, origin, spacing);

  const std::size_t pointCount =
      static_cast<std::size_t>(samplesPerAxis) *
      static_cast<std::size_t>(samplesPerAxis) *
      static_cast<std::size_t>(samplesPerAxis);

  std::vector<viskores::Float32> pointScalars(pointCount, 0.0f);
  for (int z = 0; z < samplesPerAxis; ++z) {
    const float posZ =
        minCorner[2] + static_cast<float>(z) * spacing[2];

    for (int y = 0; y < samplesPerAxis; ++y) {
      const float posY =
          minCorner[1] + static_cast<float>(y) * spacing[1];

      for (int x = 0; x < samplesPerAxis; ++x) {
        const float posX =
            minCorner[0] + static_cast<float>(x) * spacing[0];

        float sampleValue = 0.0f;
        for (const auto& box : boxes) {
          const bool inside =
              posX >= (box.minCorner[0] - 1e-4f) &&
              posX <= (box.maxCorner[0] + 1e-4f) &&
              posY >= (box.minCorner[1] - 1e-4f) &&
              posY <= (box.maxCorner[1] + 1e-4f) &&
              posZ >= (box.minCorner[2] - 1e-4f) &&
              posZ <= (box.maxCorner[2] + 1e-4f);
          if (inside) {
            sampleValue = std::max(sampleValue, box.scalarValue);
          }
        }

        const std::size_t linearIndex =
            static_cast<std::size_t>(z) * samplesPerAxis * samplesPerAxis +
            static_cast<std::size_t>(y) * samplesPerAxis +
            static_cast<std::size_t>(x);
        pointScalars[linearIndex] = sampleValue;
      }
    }
  }

  dataset.AddPointField(
      kDensityFieldName,
      viskores::cont::make_ArrayHandle(pointScalars, viskores::CopyFlag::On));

  return dataset;
}

viskores::cont::ColorTable VolumePainterViskores::buildColorTable(
    int numProcs, float alphaScale) const {
  (void)numProcs;
  const float clampedScale = std::clamp(alphaScale, 0.0f, 1.0f);
  viskores::cont::ColorTable colorTable("inferno");

  const std::array<float, 6> positions = {
      0.0f, 0.15f, 0.35f, 0.6f, 0.85f, 1.0f};
  const std::array<float, 6> alphaValues = {
      0.0f, 0.02f, 0.05f, 0.1f, 0.18f, 0.3f};

  for (std::size_t i = 0; i < positions.size(); ++i) {
    const float scaledAlpha =
        std::clamp(alphaValues[i] * clampedScale, 0.0f, 1.0f);
    colorTable.AddPointAlpha(positions[i], scaledAlpha);
  }

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
    ImageFull& image,
    const Vec3* colorOverride) const {
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
      float red = clampUnit(static_cast<float>(color[0]));
      float green = clampUnit(static_cast<float>(color[1]));
      float blue = clampUnit(static_cast<float>(color[2]));
      if (colorOverride != nullptr) {
        const float intensity = (red + green + blue) / 3.0f;
        red = clampUnit(intensity * (*colorOverride)[0]);
        green = clampUnit(intensity * (*colorOverride)[1]);
        blue = clampUnit(intensity * (*colorOverride)[2]);
      }
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
