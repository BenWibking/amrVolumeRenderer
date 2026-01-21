## amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
## Copyright (c) 2017
## National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
## the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
## certain rights in this software.
## Additional contributions (C) 2025 Ben Wibking.

# This file should only be called from amrVolumeRendererMacros.cmake

cmake_minimum_required(VERSION 3.10)

project(amrVolumeRendererCommon CXX)

find_package(PNG REQUIRED)

set(srcs
  Image.cpp
  ImageRGBAFloatColorOnly.cpp
  ImageRGBAFloatColorDepthSort.cpp
  ImageRGBAUByteColorFloatDepth.cpp
  ImageRGBAUByteColorOnly.cpp
  ImageRGBFloatColorDepth.cpp
  ImageSparse.cpp
  LayeredVolumeImage.cpp
  SavePNG.cpp
  SavePPM.cpp
  )

set(headers
  ${CMAKE_CURRENT_BINARY_DIR}/amrVolumeRendererConfig.h
  CameraUtils.hpp
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
  SavePNG.hpp
  Viewport.hpp
  )

amrVolumeRenderer_create_config_header("common library")

add_library(amrVolumeRendererCommon ${srcs} ${headers})
set_target_properties(amrVolumeRendererCommon PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_source_files_properties(${headers} HEADER_ONLY TRUE)

amrVolumeRenderer_target_features(amrVolumeRendererCommon)
target_link_libraries(amrVolumeRendererCommon PRIVATE PNG::PNG)

install(TARGETS amrVolumeRendererCommon
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

if(AMRVOLUMERENDERER_ENABLE_TESTING)
  add_subdirectory(Testing)
endif()
