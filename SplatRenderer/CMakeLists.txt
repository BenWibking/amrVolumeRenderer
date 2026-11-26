## amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
## Gaussian splat renderer (CPU reference + CUDA backend). Builds on its own
## (no AMReX or MPI) so that it can be linked into display applications:
##
##   cmake -S SplatRenderer -B build-splat -DSPLAT_RENDERER_ENABLE_CUDA=ON
##
## or as part of amrVolumeRenderer.

cmake_minimum_required(VERSION 3.18)
project(SplatRenderer LANGUAGES CXX)

option(SPLAT_RENDERER_ENABLE_CUDA "Build the CUDA backend" OFF)
option(SPLAT_RENDERER_ENABLE_TESTING "Build splat renderer tests" ON)

if (SPLAT_RENDERER_ENABLE_CUDA)
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
endif()

find_package(Threads REQUIRED)

set(_splat_sources
  SplatScene.cpp
  CpuSplatRenderer.cpp
  SplatKernels.hpp
  SplatRenderer.hpp
  SplatScene.hpp
  SplatTypes.hpp
)
if (SPLAT_RENDERER_ENABLE_CUDA)
  list(APPEND _splat_sources CudaSplatRenderer.cu)
else()
  list(APPEND _splat_sources CudaSplatRendererStub.cpp)
endif()

add_library(splat_renderer STATIC ${_splat_sources})
target_include_directories(splat_renderer PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_features(splat_renderer PUBLIC cxx_std_17)
set_target_properties(splat_renderer PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(splat_renderer PUBLIC Threads::Threads)
if (SPLAT_RENDERER_ENABLE_CUDA)
  set_target_properties(splat_renderer PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(splat_renderer PUBLIC SPLAT_RENDERER_HAVE_CUDA=1)
  target_link_libraries(splat_renderer PUBLIC CUDA::cudart)
endif()

add_executable(splat_render main.cpp)
target_link_libraries(splat_render PRIVATE splat_renderer)
set_target_properties(splat_render PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if (SPLAT_RENDERER_ENABLE_TESTING)
  if (CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    enable_testing()
  endif()

  add_executable(SplatRendererTest Testing/SplatRendererTest.cpp)
  target_link_libraries(SplatRendererTest PRIVATE splat_renderer)
  set_target_properties(SplatRendererTest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

  # Cross-check the LOD rule against the exporter when its sources exist.
  set(_exporter_octree ${CMAKE_CURRENT_SOURCE_DIR}/../GaussianExport/SplatOctree.cpp)
  if (TARGET amrVolumeRendererSplatOctree)
    target_link_libraries(SplatRendererTest PRIVATE amrVolumeRendererSplatOctree)
    target_compile_definitions(SplatRendererTest PRIVATE SPLAT_RENDERER_TEST_WITH_EXPORTER=1)
  elseif (EXISTS ${_exporter_octree})
    add_library(splat_exporter_octree STATIC ${_exporter_octree})
    target_include_directories(splat_exporter_octree PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
    target_compile_features(splat_exporter_octree PUBLIC cxx_std_17)
    target_link_libraries(SplatRendererTest PRIVATE splat_exporter_octree)
    target_compile_definitions(SplatRendererTest PRIVATE SPLAT_RENDERER_TEST_WITH_EXPORTER=1)
  endif()
  add_test(NAME SplatRendererTest COMMAND SplatRendererTest)

  if (SPLAT_RENDERER_ENABLE_CUDA)
    add_executable(CudaCpuMatchTest Testing/CudaCpuMatchTest.cpp)
    target_link_libraries(CudaCpuMatchTest PRIVATE splat_renderer)
    set_target_properties(CudaCpuMatchTest PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_test(NAME CudaCpuMatchTest COMMAND CudaCpuMatchTest)
  endif()
endif()
