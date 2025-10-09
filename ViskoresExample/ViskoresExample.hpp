#ifndef VISKORES_EXAMPLE_HPP
#define VISKORES_EXAMPLE_HPP

#include <miniGraphicsConfig.h>

#include <Common/Color.hpp>
#include <Common/Compositor.hpp>
#include <Common/ImageFull.hpp>
#include <Common/Mesh.hpp>

#include <glm/mat4x4.hpp>

#ifndef MINIGRAPHICS_ENABLE_VISKORES
#error "ViskoresExample requires MINIGRAPHICS_ENABLE_VISKORES"
#endif

#include <Paint/PainterViskores.hpp>

#include <mpi.h>

/// \brief Example driver that renders rank-specific geometry with Viskores and
/// composites the results with miniGraphics.
class ViskoresExample {
 public:
  ViskoresExample();

  /// \brief Run the example application.
  int run(int argc, char** argv);

 private:
  void initialize() const;
  void createRankSpecificMesh(Mesh& mesh) const;
  double paint(ImageFull& image,
               const glm::mat4& modelview,
               const glm::mat4& projection);
  Compositor* getCompositor();

  int rank;
  int numProcs;

  PainterViskores painter;
};

#endif  // VISKORES_EXAMPLE_HPP
