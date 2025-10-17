cmake_minimum_required(VERSION 3.15)

include(../CMake/miniGraphicsMacros.cmake)

find_package(Viskores QUIET)
if(NOT Viskores_FOUND)
  message(STATUS "Skipping miniGraphics Python bindings (Viskores not found)")
  return()
endif()

miniGraphics_create_config_header(miniGraphicsPythonBindings)

nanobind_add_module(miniGraphics_ext MODULE
  miniGraphics/module.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/miniGraphicsConfig.h
  ../ViskoresVolumeRenderer/ViskoresVolumeRenderer.cpp
  ../Common/VolumePainterViskores.cpp
  ../Common/VisibilityOrdering.cpp
  ../DirectSend/Base/DirectSendBase.cpp
)

miniGraphics_target_features(miniGraphics_ext)

target_link_libraries(miniGraphics_ext
  PRIVATE
    miniGraphicsCommon
    amrex
    Python3::Module
)

if(TARGET viskores::rendering)
  target_link_libraries(miniGraphics_ext PRIVATE viskores::rendering)
endif()

if(DEFINED Viskores_INCLUDE_DIRS)
  target_include_directories(miniGraphics_ext PRIVATE ${Viskores_INCLUDE_DIRS})
endif()

if(DEFINED Viskores_LIBRARIES)
  target_link_libraries(miniGraphics_ext PRIVATE ${Viskores_LIBRARIES})
endif()

target_compile_definitions(miniGraphics_ext PRIVATE MINIGRAPHICS_ENABLE_VISKORES)

set_target_properties(miniGraphics_ext PROPERTIES
  OUTPUT_NAME "miniGraphics_ext"
)

install(TARGETS miniGraphics_ext
  LIBRARY DESTINATION miniGraphics
  RUNTIME DESTINATION miniGraphics
  ARCHIVE DESTINATION miniGraphics
)
