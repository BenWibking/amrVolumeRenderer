## miniGraphics is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
## Copyright (c) 2017
## National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
## the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
## certain rights in this software.

cmake_minimum_required(VERSION 3.5)

include(../CMake/miniGraphicsMacros.cmake)

# Allow users to force Viskores usage from the command line.
set(_miniGraphicsRequireViskores OFF)
if(DEFINED MINIGRAPHICS_ENABLE_VISKORES)
  if(MINIGRAPHICS_ENABLE_VISKORES)
    set(_miniGraphicsRequireViskores ON)
  endif()
endif()

# Only build this example if Viskores is available
find_package(Viskores QUIET)

if(NOT Viskores_FOUND AND _miniGraphicsRequireViskores)
  message(FATAL_ERROR "MINIGRAPHICS_ENABLE_VISKORES=ON but Viskores was not found. "
                      "Provide Viskores via CMake package discovery to build ViskoresExample.")
endif()

if(Viskores_FOUND)
  message(STATUS "Building ViskoresExample with Viskores support")
  
  miniGraphics_executable(ViskoresExample
    SOURCES
      main.cpp
      ../DirectSend/Base/DirectSendBase.cpp
    HEADERS
      ViskoresExample.hpp
      ../DirectSend/Base/DirectSendBase.hpp
    )

  target_compile_definitions(ViskoresExample PRIVATE MINIGRAPHICS_ENABLE_VISKORES)
    
else()
  message(STATUS "Skipping ViskoresExample (Viskores not found)")
endif()
