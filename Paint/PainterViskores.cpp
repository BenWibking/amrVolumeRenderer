// ABOUTME: Implementation of Viskores-based painter for miniGraphics
// ABOUTME: Handles conversion between miniGraphics and Viskores data structures

// miniGraphics is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Copyright (c) 2017
// National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
// the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
// certain rights in this software.

#include "PainterViskores.hpp"

#include <array>
#include <cmath>
#include <iostream>
#include <map>

#if MINIGRAPHICS_HAS_VISKORES

#include <viskores/CellShape.h>
#include <viskores/Types.h>
#include <viskores/cont/ArrayHandle.h>
#include <viskores/cont/ArrayHandleBasic.h>
#include <viskores/cont/DataSetBuilderExplicit.h>
#include <viskores/cont/DeviceAdapter.h>
#include <viskores/cont/Initialize.h>
#include <viskores/cont/RuntimeDeviceTracker.h>
#include <viskores/rendering/Camera.h>
#include <viskores/rendering/Color.h>

#include <Common/Color.hpp>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace
{
constexpr const char kColorFieldName[] = "color_scalar";

// Custom view that skips on-screen annotations so the colorbar is never drawn.
class View3DNoColorBar : public viskores::rendering::View3D
{
public:
  using viskores::rendering::View3D::View3D;

  void RenderScreenAnnotations() override {}
  void RenderWorldAnnotations() override {}
};
}

PainterViskores::PainterViskores() 
{
  // Initialize MPI information
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
  
  // Initialize Viskores once per process
  static bool viskoresInitialized = false;
  if (!viskoresInitialized)
  {
    viskores::cont::Initialize();
    viskores::cont::GetRuntimeDeviceTracker().ForceDevice(viskores::cont::DeviceAdapterTagSerial{});
    viskoresInitialized = true;
  }
  
  // Initialize Viskores components (will be setup in paint method)
  scene = std::make_unique<viskores::rendering::Scene>();
  mapper = std::make_unique<viskores::rendering::MapperRayTracer>();
}

PainterViskores::~PainterViskores() = default;

viskores::cont::DataSet PainterViskores::meshToDataSet(const Mesh& mesh,
                                                       viskores::cont::ColorTable& colorTable) const
{
  // Convert miniGraphics mesh to Viskores DataSet
  const int numVertices = mesh.getNumberOfVertices();
  const int numTriangles = mesh.getNumberOfTriangles();
  
  // Create coordinate array
  std::vector<viskores::Vec3f_32> coordinates;
  coordinates.reserve(numVertices);
  
  for (int i = 0; i < numVertices; ++i)
  {
    glm::vec3 vertex = mesh.getPointCoordinates(i);
    coordinates.emplace_back(vertex.x, vertex.y, vertex.z);
  }
  
  // Create connectivity array
  std::vector<viskores::Id> connectivity;
  connectivity.reserve(numTriangles * 3);
  
  for (int i = 0; i < numTriangles; ++i)
  {
    const int* triangleIndices = mesh.getTriangleConnectionsBuffer(i);
    connectivity.push_back(triangleIndices[0]);
    connectivity.push_back(triangleIndices[1]);
    connectivity.push_back(triangleIndices[2]);
  }
  
  // Create color scalar array (per triangle) and matching color table entries
  std::vector<viskores::Float32> colorScalars;
  colorScalars.reserve(numTriangles);
  std::map<std::array<viskores::Float32, 4>, viskores::Float32> colorToScalar;
  viskores::Float32 nextScalar = 0.0f;
  
  for (int i = 0; i < numTriangles; ++i)
  {
    const float* triangleColors = mesh.getTriangleColorsBuffer(i);
    std::array<viskores::Float32, 4> rgba = {
      static_cast<viskores::Float32>(triangleColors[0]),
      static_cast<viskores::Float32>(triangleColors[1]),
      static_cast<viskores::Float32>(triangleColors[2]),
      static_cast<viskores::Float32>(triangleColors[3])
    };
    auto [it, inserted] = colorToScalar.emplace(rgba, nextScalar);
    if (inserted)
    {
      colorTable.AddPoint(nextScalar,
                          viskores::Vec3f_32(rgba[0], rgba[1], rgba[2]));
      colorTable.AddPointAlpha(nextScalar, rgba[3]);
      nextScalar += 1.0f;
    }
    colorScalars.push_back(it->second);
  }
  
  // Build the dataset using Viskores builder
  std::vector<viskores::UInt8> shapes(numTriangles, viskores::CELL_SHAPE_TRIANGLE);
  std::vector<viskores::IdComponent> numIndices(numTriangles, 3);
  viskores::cont::DataSet dataSet =
    viskores::cont::DataSetBuilderExplicit::Create(coordinates, shapes, numIndices, connectivity);
  
  // Add color scalar field (used with generated color table)
  dataSet.AddCellField(kColorFieldName,
                       viskores::cont::make_ArrayHandle(colorScalars, viskores::CopyFlag::On));
  
  // Add rank field for distributed visualization
  std::vector<viskores::Float32> rankData(numTriangles, static_cast<viskores::Float32>(rank));
  dataSet.AddCellField("rank",
                       viskores::cont::make_ArrayHandle(rankData, viskores::CopyFlag::On));
  
  return dataSet;
}

void PainterViskores::canvasToImage(const viskores::rendering::Canvas& canvas, ImageFull& image) const
{
  // Get color and depth buffers from Viskores canvas
  auto colorBuffer = canvas.GetColorBuffer();
  auto depthBuffer = canvas.GetDepthBuffer();
  
  // Get portal for reading
  auto colorPortal = colorBuffer.ReadPortal();
  auto depthPortal = depthBuffer.ReadPortal();
  
  const int width = image.getWidth();
  const int height = image.getHeight();
  
  // Copy pixels from Viskores to miniGraphics
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      const int pixelIndex = y * width + x;
      
      // Get color from Viskores (RGBA float)
      viskores::Vec4f viskoresColor = colorPortal.Get(pixelIndex);
      Color mgColor(viskoresColor[0], viskoresColor[1], viskoresColor[2], viskoresColor[3]);
      
      // Get depth from Viskores
      float depth = depthPortal.Get(pixelIndex);
      
      // Set in miniGraphics image
      image.setColor(x, y, mgColor);
      image.setDepth(x, y, depth);
    }
  }
}

void PainterViskores::setupCamera(const glm::mat4& modelview, const glm::mat4& projection)
{
  viskores::rendering::Camera camera;

  const glm::mat4 inverseModelview = glm::inverse(modelview);
  const glm::vec3 position =
    glm::vec3(inverseModelview * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
  const glm::vec3 forward =
    glm::normalize(glm::vec3(inverseModelview * glm::vec4(0.0f, 0.0f, -1.0f, 0.0f)));
  const glm::vec3 up =
    glm::normalize(glm::vec3(inverseModelview * glm::vec4(0.0f, 1.0f, 0.0f, 0.0f)));
  const glm::vec3 lookAt = position + forward;

  camera.SetPosition(viskores::Vec3f_32(position.x, position.y, position.z));
  camera.SetLookAt(viskores::Vec3f_32(lookAt.x, lookAt.y, lookAt.z));
  camera.SetViewUp(viskores::Vec3f_32(up.x, up.y, up.z));

  const float perspectiveScale = projection[1][1];
  if (perspectiveScale > 0.0f)
  {
    const float fovRadians = 2.0f * std::atan(1.0f / perspectiveScale);
    const float fovDegrees = fovRadians * (180.0f / glm::pi<float>());
    camera.SetFieldOfView(fovDegrees);
  }

  const float m22 = projection[2][2];
  const float m23 = projection[2][3];
  const float m32 = projection[3][2];
  if (std::abs(m32 + 1.0f) < 1e-5f && std::abs(m23) > 0.0f)
  {
    const float denominator = (m22 * m22) - 1.0f;
    if (std::abs(denominator) > 1e-5f)
    {
      const float range = -2.0f * m23 / denominator;
      const float sum = -m22 * range;
      const float nearPlane = (sum - range) * 0.5f;
      const float farPlane = (sum + range) * 0.5f;
      if (nearPlane > 0.0f && farPlane > nearPlane)
      {
        camera.SetClippingRange(nearPlane, farPlane);
      }
    }
  }

  if (view)
  {
    view->SetCamera(camera);
  }
}

void PainterViskores::paint(const Mesh& mesh,
                            ImageFull& image,
                            const glm::mat4& modelview,
                            const glm::mat4& projection)
{
  try
  {
    // Convert mesh to Viskores DataSet
    viskores::cont::ColorTable colorTable(viskores::ColorSpace::RGB);
    viskores::cont::DataSet dataSet = meshToDataSet(mesh, colorTable);
    
    // Create canvas with image dimensions
    canvas = std::make_unique<viskores::rendering::CanvasRayTracer>(image.getWidth(), image.getHeight());
    mapper->SetCanvas(canvas.get());

    // Reset scene so each frame starts with a clean set of actors
    scene = std::make_unique<viskores::rendering::Scene>();
    
    // Set up scene
    scene->AddActor(viskores::rendering::Actor(dataSet.GetCellSet(),
                                              dataSet.GetCoordinateSystem(),
                                              dataSet.GetField(kColorFieldName),
                                              colorTable));
    
    // Create view
    view = std::make_unique<View3DNoColorBar>(*scene,
                                              *mapper,
                                              *canvas,
                                              viskores::rendering::Color(0.0f, 0.0f, 0.0f, 1.0f));
    view->SetWorldAnnotationsEnabled(false);
    
    // Setup camera from matrices
    setupCamera(modelview, projection);
    
    // Render the scene
    view->Paint();
    
    // Convert result back to miniGraphics image
    canvasToImage(*canvas, image);

    // Release Viskores objects in a defined order to avoid dangling references
    view.reset();
    mapper->SetCanvas(nullptr);
    canvas.reset();
    scene.reset();
    
    // Output rank information for debugging
    if (rank == 0)
    {
      std::cout << "PainterViskores: Rendered " << mesh.getNumberOfTriangles() 
                << " triangles on " << numProcs << " ranks" << std::endl;
    }
  }
  catch (const std::exception& e)
  {
    if (mapper)
    {
      mapper->SetCanvas(nullptr);
    }
    view.reset();
    canvas.reset();
    scene.reset();
    std::cerr << "PainterViskores error on rank " << rank << ": " << e.what() << std::endl;
    throw;
  }
}

#else  // MINIGRAPHICS_HAS_VISKORES

#include <mpi.h>

PainterViskores::PainterViskores() : warnedOnce(false) {}

PainterViskores::~PainterViskores() = default;

void PainterViskores::paint(const Mesh&,
                            ImageFull&,
                            const glm::mat4&,
                            const glm::mat4&) {
  int commRank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
  if (!warnedOnce && commRank == 0) {
    std::cerr << "PainterViskores: Viskores support is not available in this "
                 "build. Enable MINIGRAPHICS_ENABLE_VISKORES to render."
              << std::endl;
  }
  warnedOnce = true;
  MPI_Abort(MPI_COMM_WORLD, 1);
}

#endif  // MINIGRAPHICS_HAS_VISKORES
