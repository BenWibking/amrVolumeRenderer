## miniGraphics is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
## Copyright (c) 2017
## National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
## the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
## certain rights in this software.

# This file should only be called from miniGraphicsMacros.cmake

cmake_minimum_required(VERSION 3.10)

project(miniGraphicsCommon CXX)

set(srcs
  Image.cpp
  ImageRGBAFloatColorOnly.cpp
  ImageRGBAFloatColorDepthSort.cpp
  ImageRGBAUByteColorFloatDepth.cpp
  ImageRGBAUByteColorOnly.cpp
  ImageRGBFloatColorDepth.cpp
  ImageSparse.cpp
  LayeredVolumeImage.cpp
  SavePPM.cpp
  )

set(headers
  ${CMAKE_CURRENT_BINARY_DIR}/miniGraphicsConfig.h
  Color.hpp
  Compositor.hpp
  Image.hpp
  ImageColorDepth.hpp
  ImageColorOnly.hpp
  ImageFull.hpp
  ImageRGBAFloatColorOnly.hpp
  ImageRGBAFloatColorDepthSort.hpp
  ImageRGBAUByteColorFloatDepth.hpp
  ImageRGBAUByteColorOnly.hpp
  ImageRGBFloatColorDepth.hpp
  LayeredImageInterface.hpp
  LayeredVolumeImage.hpp
  ImageSparse.hpp
  ImageSparseColorDepth.hpp
  ImageSparseColorOnly.hpp
  VolumePainterViskores.hpp
  VolumeTypes.hpp
  VisibilityOrdering.hpp
  SavePPM.hpp
  Viewport.hpp
  )

miniGraphics_create_config_header("common library")

add_library(miniGraphicsCommon ${srcs} ${headers})

set_source_files_properties(${headers} HEADER_ONLY TRUE)

miniGraphics_target_features(miniGraphicsCommon)

if(MINIGRAPHICS_ENABLE_TESTING)
  add_subdirectory(Testing)
endif()
