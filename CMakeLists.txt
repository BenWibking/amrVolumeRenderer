## amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
## Copyright (c) 2017
## National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
## the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
## certain rights in this software.
## Additional contributions (C) 2025 Ben Wibking.

cmake_minimum_required(VERSION 3.14)

project(amrVolumeRenderer)

option(AMRVOLUMERENDERER_ENABLE_TESTING "Turn on/off building tests." ON)
if (AMRVOLUMERENDERER_ENABLE_TESTING)
  enable_testing()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Suppress GCC PSA ABI noise when building AMReX headers under C++17+
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-psabi>")
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

set(Python_EXECUTABLE ${Python3_EXECUTABLE})
set(Python_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
set(Python_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
set(Python_LIBRARY ${Python3_LIBRARIES})
set(Python_LIBRARIES ${Python3_LIBRARIES})

include(FetchContent)

# Disable nanobind tests when building as a subproject.
set(NB_TEST OFF CACHE BOOL "" FORCE)
set(NB_TEST_STABLE_ABI OFF CACHE BOOL "" FORCE)
set(NB_PYTHON_TARGET Python3::Module CACHE INTERNAL "Python target for nanobind")
FetchContent_Declare(nanobind
  GIT_REPOSITORY https://github.com/wjakob/nanobind.git
  GIT_TAG dd350fe81931a1b362196cb415d188c36422766e
)
FetchContent_MakeAvailable(nanobind)

option(AMRVOLUMERENDERER_SUPERBUILD_VISKORES
       "Download and build Viskores with FetchContent for the renderer." ON)
set(AMRVOLUMERENDERER_VISKORES_GIT_REPOSITORY
    "https://github.com/Viskores/viskores.git"
    CACHE STRING "Git repository used to fetch Viskores.")
set(AMRVOLUMERENDERER_VISKORES_GIT_TAG
    "v1.1.0-rc1"
    CACHE STRING "Git tag or commit for the Viskores superbuild.")
option(AMRVOLUMERENDERER_VISKORES_ENABLE_LOGGING
       "Enable Viskores' internal logging when building from source."
       OFF)

if(AMRVOLUMERENDERER_SUPERBUILD_VISKORES)
  set(_amrvr_fetch_viskores OFF)
  if(NOT TARGET viskores::rendering)
    set(_amrvr_fetch_viskores ON)
    FetchContent_Declare(viskores
      GIT_REPOSITORY ${AMRVOLUMERENDERER_VISKORES_GIT_REPOSITORY}
      GIT_TAG ${AMRVOLUMERENDERER_VISKORES_GIT_TAG}
      GIT_SHALLOW TRUE
    )
  endif()

  FetchContent_GetProperties(viskores)
  if(NOT viskores_POPULATED)
    set(Viskores_BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_RENDERING ON CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_MPI ON CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_LOGGING ${AMRVOLUMERENDERER_VISKORES_ENABLE_LOGGING}
        CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_TBB OFF CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_OPENMP OFF CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_ANARI OFF CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_GL_CONTEXT OFF CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_OSMESA_CONTEXT OFF CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_EGL_CONTEXT OFF CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_TESTING_LIBRARY OFF CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_DEVELOPER_FLAGS OFF CACHE BOOL "" FORCE)
    set(Viskores_ENABLE_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
  endif()

  set(_amrvr_saved_cxx_flags "${CMAKE_CXX_FLAGS}")
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w" CACHE STRING "" FORCE)
  endif()

  FetchContent_MakeAvailable(viskores)
  if(DEFINED _amrvr_saved_cxx_flags)
    set(CMAKE_CXX_FLAGS "${_amrvr_saved_cxx_flags}" CACHE STRING "" FORCE)
  endif()
  unset(_amrvr_saved_cxx_flags)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(_amrvr_viskores_targets
        viskores_cont
        viskores_filter_clean_grid
        viskores_filter_connected_components
        viskores_filter_contour
        viskores_filter_core
        viskores_filter_density_estimate
        viskores_filter_entity_extraction
        viskores_filter_field_conversion
        viskores_filter_field_transform
        viskores_filter_flow
        viskores_filter_geometry_refinement
        viskores_filter_image_processing
        viskores_filter_mesh_info
        viskores_filter_multi_block
        viskores_filter_resampling
        viskores_filter_scalar_topology
        viskores_filter_uncertainty
        viskores_filter_vector_analysis
        viskores_filter_zfp
        viskores_io
        viskores_lodepng
        viskores_rendering
        viskores_source
        viskores_worklet)
    foreach(_amrvr_target IN LISTS _amrvr_viskores_targets)
      if(TARGET ${_amrvr_target})
        set_property(TARGET ${_amrvr_target} APPEND PROPERTY COMPILE_OPTIONS -w)
      endif()
    endforeach()
    unset(_amrvr_viskores_targets)
    unset(_amrvr_target)
  endif()
  unset(_amrvr_fetch_viskores)

  if(TARGET viskores::rendering OR TARGET viskores_rendering)
    set(AMRVR_VISKORES_FROM_SUPERBUILD ON CACHE INTERNAL
        "amrVolumeRenderer is building Viskores from source." FORCE)
  endif()

  if(TARGET viskores_cont AND NOT TARGET viskores::cont)
    add_library(viskores::cont ALIAS viskores_cont)
  endif()
  if(TARGET viskores_rendering AND NOT TARGET viskores::rendering)
    add_library(viskores::rendering ALIAS viskores_rendering)
  endif()

  if(DEFINED viskores_BINARY_DIR)
    list(PREPEND CMAKE_PREFIX_PATH "${viskores_BINARY_DIR}")
    set(Viskores_DIR "${viskores_BINARY_DIR}" CACHE PATH
        "Path to ViskoresConfig.cmake generated by the superbuild." FORCE)
  endif()

  if(TARGET viskores::rendering AND NOT DEFINED Viskores_LIBRARIES)
    set(Viskores_LIBRARIES viskores::rendering CACHE INTERNAL
        "Viskores rendering target provided by the superbuild.")
  endif()

  if(TARGET viskores::cont AND NOT DEFINED Viskores_INCLUDE_DIRS)
    set(Viskores_INCLUDE_DIRS
        "$<TARGET_PROPERTY:viskores::cont,INTERFACE_INCLUDE_DIRECTORIES>"
        CACHE INTERNAL
        "Viskores include directories propagated from the superbuild.")
  endif()

  if(TARGET viskores::rendering)
    set(Viskores_FOUND ON CACHE INTERNAL "Flag indicating Viskores is available." FORCE)
  endif()
endif()

# These macros are not stricly necessary for this top-level CMake file, but
# by including it here, we can find the necessary packages once and this
# will significantly speed up when this file is included in subdirectories.
include(CMake/amrVolumeRendererMacros.cmake)

add_subdirectory(ViskoresVolumeRenderer)

if (TARGET ViskoresVolumeRenderer)
  add_subdirectory(python)
endif()
