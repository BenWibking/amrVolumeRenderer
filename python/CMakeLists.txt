cmake_minimum_required(VERSION 3.15)

include(../CMake/amrVolumeRendererMacros.cmake)

amrVolumeRenderer_create_config_header(amrVolumeRendererPythonBindings)

nanobind_add_module(amrVolumeRenderer_ext MODULE
  amrVolumeRenderer/module.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/amrVolumeRendererConfig.h
  ../VolumeRenderer/VolumeRenderer.cpp
  ../Common/VolumePainter.cpp
  ../Common/VisibilityOrdering.cpp
  ../DirectSend/Base/DirectSendBase.cpp
)

amrVolumeRenderer_target_features(amrVolumeRenderer_ext)

set(_amrvr_nanobind_system_includes "")
if(DEFINED nanobind_SOURCE_DIR)
  list(APPEND _amrvr_nanobind_system_includes
       "${nanobind_SOURCE_DIR}/include")
endif()
if(TARGET nanobind::nanobind)
  get_target_property(_amrvr_nanobind_iface nanobind::nanobind
                      INTERFACE_INCLUDE_DIRECTORIES)
  if(_amrvr_nanobind_iface)
    list(APPEND _amrvr_nanobind_system_includes "${_amrvr_nanobind_iface}")
  endif()
endif()
if(_amrvr_nanobind_system_includes)
  target_include_directories(amrVolumeRenderer_ext SYSTEM PRIVATE
                             ${_amrvr_nanobind_system_includes})
endif()
unset(_amrvr_nanobind_system_includes)
unset(_amrvr_nanobind_iface)

target_link_libraries(amrVolumeRenderer_ext
  PRIVATE
    amrVolumeRendererCommon
    amrex
    Python3::Module
)

set_target_properties(amrVolumeRenderer_ext PROPERTIES
  OUTPUT_NAME "amrVolumeRenderer_ext"
)

install(TARGETS amrVolumeRenderer_ext
  LIBRARY DESTINATION amrVolumeRenderer
  RUNTIME DESTINATION amrVolumeRenderer
  ARCHIVE DESTINATION amrVolumeRenderer
)
