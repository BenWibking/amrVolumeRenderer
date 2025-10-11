## miniGraphics is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
## Copyright (c) 2017
## National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
## the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
## certain rights in this software.

cmake_minimum_required(VERSION 3.10)

include(../CMake/miniGraphicsMacros.cmake)

# Allow users to force Viskores usage from the command line.
set(_miniGraphicsRequireViskores OFF)
if(DEFINED MINIGRAPHICS_ENABLE_VISKORES)
  if(MINIGRAPHICS_ENABLE_VISKORES)
    set(_miniGraphicsRequireViskores ON)
  endif()
endif()

find_package(Viskores QUIET)

if(NOT Viskores_FOUND AND _miniGraphicsRequireViskores)
  message(FATAL_ERROR "MINIGRAPHICS_ENABLE_VISKORES=ON but Viskores was not found. "
                      "Provide Viskores via CMake package discovery to build ViskoresVolumeExample.")
endif()

if(Viskores_FOUND)
  message(STATUS "Building ViskoresVolumeExample with Viskores support")

  miniGraphics_executable(ViskoresVolumeExample
    SOURCES
      main.cpp
      ../Common/VolumePainterViskores.cpp
      ../Common/VisibilityOrdering.cpp
      ../DirectSend/Base/DirectSendBase.cpp
    HEADERS
      ViskoresVolumeExample.hpp
      ../DirectSend/Base/DirectSendBase.hpp
    )

  if(TARGET viskores::rendering)
    target_link_libraries(ViskoresVolumeExample PRIVATE viskores::rendering)
  endif()

  if(DEFINED Viskores_INCLUDE_DIRS)
    target_include_directories(ViskoresVolumeExample PRIVATE ${Viskores_INCLUDE_DIRS})
  endif()

  if(DEFINED Viskores_LIBRARIES)
    target_link_libraries(ViskoresVolumeExample PRIVATE ${Viskores_LIBRARIES})
  endif()

  target_compile_definitions(ViskoresVolumeExample PRIVATE MINIGRAPHICS_ENABLE_VISKORES)
else()
  message(STATUS "Skipping ViskoresVolumeExample (Viskores not found)")
endif()
