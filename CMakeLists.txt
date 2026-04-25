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

set(CMAKE_CXX_STANDARD 20 CACHE STRING
    "C++ standard used to build amrVolumeRenderer and AMReX.")
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(AMReX_GPU_BACKEND STREQUAL "CUDA")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 20 CACHE STRING
      "CUDA C++ standard used to build amrVolumeRenderer and AMReX.")
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the build type, e.g. Release, Debug, or RelWithDebInfo.")
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
               "Release" "Debug" "MinSizeRel" "RelWithDebInfo")
endif()

option(AMRVOLUMERENDERER_ENABLE_TESTING "Turn on/off building tests." ON)
if (AMRVOLUMERENDERER_ENABLE_TESTING)
  enable_testing()
endif()

option(AMRVOLUMERENDERER_ENABLE_PYTHON "Turn on/off building Python bindings." ON)

include(FetchContent)

if (NOT TARGET amrex AND NOT TARGET AMReX::amrex)
  set(AMRVOLUMERENDERER_AMREX_RELEASE_TAG "26.04" CACHE STRING
      "AMReX release tag to fetch from GitHub releases.")
  set(AMRVOLUMERENDERER_AMREX_RELEASE_URL
      "https://github.com/AMReX-Codes/amrex/releases/download/${AMRVOLUMERENDERER_AMREX_RELEASE_TAG}/amrex-${AMRVOLUMERENDERER_AMREX_RELEASE_TAG}.tar.gz"
      CACHE STRING "AMReX release tarball URL.")
  FetchContent_Declare(amrex
    URL ${AMRVOLUMERENDERER_AMREX_RELEASE_URL}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )

  set(AMReX_SPACEDIM 3 CACHE STRING "AMReX dimension" FORCE)
  set(AMReX_MPI ON CACHE BOOL "Enable MPI in AMReX" FORCE)
  set(AMReX_FORTRAN OFF CACHE BOOL "Disable AMReX Fortran interfaces" FORCE)
  set(AMReX_OMP OFF CACHE BOOL "Disable AMReX OpenMP" FORCE)
  set(AMReX_ENABLE_PIC ON CACHE BOOL "Build AMReX with position independent code" FORCE)

  FetchContent_MakeAvailable(amrex)
endif()

if(TARGET AMReX::amrex AND NOT TARGET amrex)
  add_library(amrex ALIAS AMReX::amrex)
endif()

if(TARGET amrex)
  get_target_property(_amrex_alias amrex ALIASED_TARGET)
  if(_amrex_alias)
    set_target_properties(${_amrex_alias} PROPERTIES POSITION_INDEPENDENT_CODE ON)
  else()
    set_target_properties(amrex PROPERTIES POSITION_INDEPENDENT_CODE ON)
  endif()
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Suppress GCC PSA ABI noise when building AMReX headers under C++20.
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-psabi>")
endif()

if (AMRVOLUMERENDERER_ENABLE_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

  set(Python_EXECUTABLE ${Python3_EXECUTABLE})
  set(Python_INCLUDE_DIR ${Python3_INCLUDE_DIRS})
  set(Python_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
  set(Python_LIBRARY ${Python3_LIBRARIES})
  set(Python_LIBRARIES ${Python3_LIBRARIES})

  # Disable nanobind tests when building as a subproject.
  set(NB_TEST OFF CACHE BOOL "" FORCE)
  set(NB_TEST_STABLE_ABI OFF CACHE BOOL "" FORCE)
  set(NB_PYTHON_TARGET Python3::Module CACHE INTERNAL "Python target for nanobind")
  FetchContent_Declare(nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG dd350fe81931a1b362196cb415d188c36422766e
    GIT_SUBMODULES_RECURSE TRUE
    GIT_PROGRESS TRUE
  )
  FetchContent_MakeAvailable(nanobind)
endif()

# These macros are not stricly necessary for this top-level CMake file, but
# by including it here, we can find the necessary packages once and this
# will significantly speed up when this file is included in subdirectories.
include(CMake/amrVolumeRendererMacros.cmake)

add_subdirectory(VolumeRenderer)

if (TARGET volume_renderer AND AMRVOLUMERENDERER_ENABLE_PYTHON)
  add_subdirectory(python)
endif()
