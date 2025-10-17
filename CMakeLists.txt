## amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
## Copyright (c) 2017
## National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
## the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
## certain rights in this software.
## Additional contributions (C) 2025 Ben Wibking.

cmake_minimum_required(VERSION 3.10)

project(amrVolumeRenderer)

option(AMRVOLUMERENDERER_ENABLE_TESTING "Turn on/off building tests." ON)
if (AMRVOLUMERENDERER_ENABLE_TESTING)
  enable_testing()
endif()

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
add_subdirectory(nanobind EXCLUDE_FROM_ALL)

# These macros are not stricly necessary for this top-level CMake file, but
# by including it here, we can find the necessary packages once and this
# will significantly speed up when this file is included in subdirectories.
include(CMake/amrVolumeRendererMacros.cmake)

add_subdirectory(ViskoresVolumeRenderer)

if (TARGET ViskoresVolumeRenderer)
  add_subdirectory(python)
endif()
