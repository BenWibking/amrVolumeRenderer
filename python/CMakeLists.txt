cmake_minimum_required(VERSION 3.15)

include(../CMake/amrVolumeRendererMacros.cmake)

set(_amrvr_python_have_viskores OFF)
if(AMRVR_VISKORES_FROM_SUPERBUILD)
  set(_amrvr_python_have_viskores ON)
elseif(TARGET viskores::rendering)
  set(Viskores_FOUND ON)
  set(_amrvr_python_have_viskores ON)
else()
  find_package(Viskores QUIET)
  if(Viskores_FOUND)
    set(_amrvr_python_have_viskores ON)
  endif()
endif()

if(NOT _amrvr_python_have_viskores)
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

if(TARGET viskores::rendering)
  target_link_libraries(amrVolumeRenderer_ext PRIVATE viskores::rendering)
endif()

if(DEFINED Viskores_INCLUDE_DIRS)
  target_include_directories(amrVolumeRenderer_ext SYSTEM PRIVATE
                             ${Viskores_INCLUDE_DIRS})
endif()

if(DEFINED Viskores_LIBRARIES)
  target_link_libraries(amrVolumeRenderer_ext PRIVATE ${Viskores_LIBRARIES})
endif()

target_compile_definitions(amrVolumeRenderer_ext PRIVATE AMRVOLUMERENDERER_ENABLE_VISKORES)

# Propagate runtime search paths so the extension can find the Viskores binaries.
set(_amrVolumeRenderer_ext_rpaths "")

if(TARGET viskores::rendering)
  get_target_property(_amrvr_viskores_imported viskores::rendering IMPORTED)
  if(_amrvr_viskores_imported)
    get_target_property(_amrvr_viskores_configs viskores::rendering IMPORTED_CONFIGURATIONS)
    if(NOT _amrvr_viskores_configs OR _amrvr_viskores_configs STREQUAL "NOTFOUND")
      set(_amrvr_viskores_configs Release)
    endif()
    foreach(_amrvr_cfg ${_amrvr_viskores_configs})
      string(TOUPPER "${_amrvr_cfg}" _amrvr_cfg_upper)
      get_target_property(_amrvr_viskores_location viskores::rendering "IMPORTED_LOCATION_${_amrvr_cfg_upper}")
      if(NOT _amrvr_viskores_location OR _amrvr_viskores_location STREQUAL "NOTFOUND")
        get_target_property(_amrvr_viskores_location viskores::rendering IMPORTED_LOCATION)
      endif()
      if(_amrvr_viskores_location AND NOT _amrvr_viskores_location STREQUAL "NOTFOUND")
        get_filename_component(_amrvr_viskores_dir "${_amrvr_viskores_location}" DIRECTORY)
        list(APPEND _amrVolumeRenderer_ext_rpaths "${_amrvr_viskores_dir}")
      endif()
    endforeach()
  else()
    get_target_property(_amrvr_viskores_runtime_dir viskores::rendering RUNTIME_OUTPUT_DIRECTORY)
    if(_amrvr_viskores_runtime_dir AND NOT _amrvr_viskores_runtime_dir STREQUAL "NOTFOUND")
      list(APPEND _amrVolumeRenderer_ext_rpaths "${_amrvr_viskores_runtime_dir}")
    endif()
    get_target_property(_amrvr_viskores_library_dir viskores::rendering LIBRARY_OUTPUT_DIRECTORY)
    if(_amrvr_viskores_library_dir AND NOT _amrvr_viskores_library_dir STREQUAL "NOTFOUND")
      list(APPEND _amrVolumeRenderer_ext_rpaths "${_amrvr_viskores_library_dir}")
    endif()
  endif()
endif()

if(DEFINED Viskores_LIBRARIES)
  foreach(_amrvr_library ${Viskores_LIBRARIES})
    if(IS_ABSOLUTE "${_amrvr_library}")
      get_filename_component(_amrvr_library_dir "${_amrvr_library}" DIRECTORY)
      list(APPEND _amrVolumeRenderer_ext_rpaths "${_amrvr_library_dir}")
    endif()
  endforeach()
endif()

list(REMOVE_DUPLICATES _amrVolumeRenderer_ext_rpaths)

if(APPLE)
  list(APPEND _amrVolumeRenderer_ext_rpaths "@loader_path/../lib")
  list(PREPEND _amrVolumeRenderer_ext_rpaths "@loader_path")
elseif(UNIX)
  list(APPEND _amrVolumeRenderer_ext_rpaths "$ORIGIN/../lib")
  list(PREPEND _amrVolumeRenderer_ext_rpaths "$ORIGIN")
endif()

list(REMOVE_DUPLICATES _amrVolumeRenderer_ext_rpaths)

if(_amrVolumeRenderer_ext_rpaths)
  set_target_properties(amrVolumeRenderer_ext PROPERTIES
    BUILD_RPATH "${_amrVolumeRenderer_ext_rpaths}"
    INSTALL_RPATH "${_amrVolumeRenderer_ext_rpaths}"
  )
endif()

set_target_properties(amrVolumeRenderer_ext PROPERTIES
  OUTPUT_NAME "amrVolumeRenderer_ext"
)

install(TARGETS amrVolumeRenderer_ext
  LIBRARY DESTINATION amrVolumeRenderer
  RUNTIME DESTINATION amrVolumeRenderer
  ARCHIVE DESTINATION amrVolumeRenderer
)
