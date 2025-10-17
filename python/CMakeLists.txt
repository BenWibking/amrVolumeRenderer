cmake_minimum_required(VERSION 3.15)

include(../CMake/amrVolumeRendererMacros.cmake)

find_package(Viskores QUIET)
if(NOT Viskores_FOUND)
  message(STATUS "Skipping amrVolumeRenderer Python bindings (Viskores not found)")
  return()
endif()

amrVolumeRenderer_create_config_header(amrVolumeRendererPythonBindings)

nanobind_add_module(amrVolumeRenderer_ext MODULE
  amrVolumeRenderer/module.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/amrVolumeRendererConfig.h
  ../ViskoresVolumeRenderer/ViskoresVolumeRenderer.cpp
  ../Common/VolumePainterViskores.cpp
  ../Common/VisibilityOrdering.cpp
  ../DirectSend/Base/DirectSendBase.cpp
)

amrVolumeRenderer_target_features(amrVolumeRenderer_ext)

target_link_libraries(amrVolumeRenderer_ext
  PRIVATE
    amrVolumeRendererCommon
    amrex
    Python3::Module
)

if(TARGET viskores::rendering)
  target_link_libraries(amrVolumeRenderer_ext PRIVATE viskores::rendering)
endif()

if(DEFINED Viskores_INCLUDE_DIRS)
  target_include_directories(amrVolumeRenderer_ext PRIVATE ${Viskores_INCLUDE_DIRS})
endif()

if(DEFINED Viskores_LIBRARIES)
  target_link_libraries(amrVolumeRenderer_ext PRIVATE ${Viskores_LIBRARIES})
endif()

target_compile_definitions(amrVolumeRenderer_ext PRIVATE AMRVOLUMERENDERER_ENABLE_VISKORES)

set_target_properties(amrVolumeRenderer_ext PROPERTIES
  OUTPUT_NAME "amrVolumeRenderer_ext"
)

install(TARGETS amrVolumeRenderer_ext
  LIBRARY DESTINATION amrVolumeRenderer
  RUNTIME DESTINATION amrVolumeRenderer
  ARCHIVE DESTINATION amrVolumeRenderer
)
