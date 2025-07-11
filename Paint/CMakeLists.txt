## miniGraphics is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
## Copyright (c) 2017
## National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
## the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
## certain rights in this software.

# This file should only be called from miniGraphicsMacros.cmake

cmake_minimum_required(VERSION 3.5)

project(miniGraphicsPaint CXX)

set(srcs
  PainterSimple.cpp
  )

set(headers
  ${CMAKE_CURRENT_BINARY_DIR}/miniGraphicsConfig.h
  Painter.hpp
  PainterSimple.hpp
  )

set(include_dirs)
set(libs)

miniGraphics_create_config_header("paint library")

add_library(miniGraphicsPaint ${srcs} ${headers})

miniGraphics_target_features(miniGraphicsPaint)

target_include_directories(miniGraphicsPaint PRIVATE ${include_dirs})

target_link_libraries(miniGraphicsPaint PRIVATE ${libs})

set_source_files_properties(miniGraphicsPaint HEADER_ONLY TRUE)
