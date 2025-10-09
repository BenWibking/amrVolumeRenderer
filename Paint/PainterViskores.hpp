// ABOUTME: Viskores-based painter implementation for miniGraphics rank-based rendering
// ABOUTME: Uses Viskores rendering engine to paint meshes with distributed data support

// miniGraphics is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Copyright (c) 2017
// National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
// the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
// certain rights in this software.

#ifndef PAINTER_VISKORES_H
#define PAINTER_VISKORES_H

#include <miniGraphicsConfig.h>

#include "Painter.hpp"

#if defined(MINIGRAPHICS_ENABLE_VISKORES) && \
    __has_include(<viskores/rendering/Scene.h>)
#define MINIGRAPHICS_HAS_VISKORES 1

#include <viskores/rendering/Scene.h>
#include <viskores/rendering/View3D.h>
#include <viskores/rendering/Canvas.h>
#include <viskores/rendering/CanvasRayTracer.h>
#include <viskores/rendering/MapperRayTracer.h>
#include <viskores/rendering/Actor.h>
#include <viskores/cont/DataSet.h>
#include <viskores/cont/ColorTable.h>
#include <viskores/cont/Initialize.h>

#include <mpi.h>
#include <memory>

class PainterViskores : public Painter {
 private:
  // Viskores components
  std::unique_ptr<viskores::rendering::Scene> scene;
  std::unique_ptr<viskores::rendering::CanvasRayTracer> canvas;
  std::unique_ptr<viskores::rendering::MapperRayTracer> mapper;
  std::unique_ptr<viskores::rendering::View3D> view;

  // MPI information for rank-based rendering
  int rank;
  int numProcs;
  
  // Convert miniGraphics mesh to Viskores DataSet
  viskores::cont::DataSet meshToDataSet(const Mesh& mesh,
                                        viskores::cont::ColorTable& colorTable) const;
  
  // Convert Viskores canvas to miniGraphics image
  void canvasToImage(const viskores::rendering::Canvas& canvas, ImageFull& image) const;
  
  // Setup camera from OpenGL-style matrices
  void setupCamera(const glm::mat4& modelview, const glm::mat4& projection);

 public:
  PainterViskores();
  virtual ~PainterViskores();

  void paint(const Mesh& mesh,
             ImageFull& image,
             const glm::mat4& modelview,
             const glm::mat4& projection) override;
};

#else

#define MINIGRAPHICS_HAS_VISKORES 0

/// \brief Stub implementation used when Viskores headers are unavailable.
class PainterViskores : public Painter {
 public:
  PainterViskores();
  ~PainterViskores() override;

  void paint(const Mesh& mesh,
             ImageFull& image,
             const glm::mat4& modelview,
             const glm::mat4& projection) override;

 private:
  bool warnedOnce;
};

#endif  // MINIGRAPHICS_HAS_VISKORES

#endif  // PAINTER_VISKORES_H
