## miniGraphics is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
## Copyright (c) 2017
## National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
## the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
## certain rights in this software.

# This file should only be called from miniGraphicsMacros.cmake

cmake_minimum_required(VERSION 3.10)

project(miniGraphicsPaint CXX)

set(srcs
  PainterViskores.cpp
  )

set(headers
  ${CMAKE_CURRENT_BINARY_DIR}/miniGraphicsConfig.h
  Painter.hpp
  PainterViskores.hpp
  )

# Find Viskores
find_package(Viskores QUIET)

set(include_dirs)
set(libs)

# Add Viskores if available
if(Viskores_FOUND)
  if(TARGET viskores::rendering)
    list(APPEND libs viskores::rendering)
  endif()
  if(DEFINED Viskores_INCLUDE_DIRS)
    list(APPEND include_dirs ${Viskores_INCLUDE_DIRS})
  endif()
  if(DEFINED Viskores_LIBRARIES)
    list(APPEND libs ${Viskores_LIBRARIES})
  endif()
endif()

miniGraphics_create_config_header("paint library")

add_library(miniGraphicsPaint ${srcs} ${headers})

# Add Viskores compile definition after target creation
if(Viskores_FOUND)
  target_compile_definitions(miniGraphicsPaint PRIVATE MINIGRAPHICS_ENABLE_VISKORES)
endif()

miniGraphics_target_features(miniGraphicsPaint)

target_include_directories(miniGraphicsPaint PRIVATE ${include_dirs})

target_link_libraries(miniGraphicsPaint PRIVATE ${libs})

set_source_files_properties(miniGraphicsPaint HEADER_ONLY TRUE)
