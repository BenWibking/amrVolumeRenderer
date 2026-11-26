## amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
## Gaussian splat exporter for AMReX plotfiles.

cmake_minimum_required(VERSION 3.14)

include(../CMake/amrVolumeRendererMacros.cmake)

# Octree logic without AMReX or MPI dependencies.
add_library(amrVolumeRendererSplatOctree STATIC
  SplatOctree.cpp
  SplatOctree.hpp
)
amrVolumeRenderer_set_output_dirs(amrVolumeRendererSplatOctree)
target_include_directories(amrVolumeRendererSplatOctree PUBLIC
  ${amrVolumeRenderer_SOURCE_DIR})
set_target_properties(amrVolumeRendererSplatOctree PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
  POSITION_INDEPENDENT_CODE ON
)

# Distributed plotfile reader and file writer.
add_library(amrVolumeRendererGaussianExport STATIC
  PlotfileSplatExporter.cpp
  PlotfileSplatExporter.hpp
)
amrVolumeRenderer_set_output_dirs(amrVolumeRendererGaussianExport)
amrVolumeRenderer_target_features(amrVolumeRendererGaussianExport)
amrVolumeRenderer_cuda_target_sources(amrVolumeRendererGaussianExport
  PlotfileSplatExporter.cpp)
set_target_properties(amrVolumeRendererGaussianExport PROPERTIES
  POSITION_INDEPENDENT_CODE ON)
target_link_libraries(amrVolumeRendererGaussianExport
  PUBLIC amrVolumeRendererSplatOctree amrex)

add_executable(gaussian_export main.cpp)
amrVolumeRenderer_set_output_dirs(gaussian_export)
amrVolumeRenderer_target_features(gaussian_export)
amrVolumeRenderer_cuda_target_sources(gaussian_export main.cpp)
target_link_libraries(gaussian_export PRIVATE amrVolumeRendererGaussianExport amrex)

if (AMRVOLUMERENDERER_ENABLE_TESTING)
  add_subdirectory(Testing)
endif()
