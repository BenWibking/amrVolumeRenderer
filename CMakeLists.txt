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

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Suppress GCC PSA ABI noise when building AMReX headers under C++17+
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wno-psabi>")
endif()

if (AMRVOLUMERENDERER_ENABLE_PYTHON)
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
