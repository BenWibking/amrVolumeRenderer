## amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
##

cmake_minimum_required(VERSION 3.14)

include(../CMake/amrVolumeRendererMacros.cmake)
include(FetchContent)

FetchContent_Declare(amrex
  GIT_REPOSITORY https://github.com/AMReX-Codes/amrex.git
  GIT_TAG 546a888ef0417648b5086a13b3eb3cd13f185070
)

# Allow users to force Viskores usage from the command line.
set(_amrVolumeRendererRequireViskores OFF)
if(DEFINED AMRVOLUMERENDERER_ENABLE_VISKORES)
  if(AMRVOLUMERENDERER_ENABLE_VISKORES)
    set(_amrVolumeRendererRequireViskores ON)
  endif()
endif()

find_package(Viskores QUIET)

if(NOT Viskores_FOUND AND _amrVolumeRendererRequireViskores)
  message(FATAL_ERROR "AMRVOLUMERENDERER_ENABLE_VISKORES=ON but Viskores was not found. "
                      "Provide Viskores via CMake package discovery to build ViskoresVolumeRenderer.")
endif()

if(Viskores_FOUND)
  message(STATUS "Building ViskoresVolumeRenderer with Viskores support")

  set(AMReX_SPACEDIM 3 CACHE STRING "AMReX dimension" FORCE)
  set(AMReX_MPI ON CACHE BOOL "Enable MPI in AMReX" FORCE)
  set(AMReX_FORTRAN OFF CACHE BOOL "Disable AMReX Fortran interfaces" FORCE)
  set(AMReX_OMP OFF CACHE BOOL "Disable AMReX OpenMP" FORCE)
  set(AMReX_GPU_BACKEND "NONE" CACHE STRING "Disable AMReX GPU backends" FORCE)
  set(AMReX_ENABLE_PIC ON CACHE BOOL "Build AMReX with position independent code" FORCE)

  if(NOT TARGET amrex)
    FetchContent_MakeAvailable(amrex)
  endif()

  if(TARGET amrex)
    get_target_property(_amrex_alias amrex ALIASED_TARGET)
    if(_amrex_alias)
      set_target_properties(${_amrex_alias} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    else()
      set_target_properties(amrex PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
  endif()

  amrVolumeRenderer_executable(ViskoresVolumeRenderer
    SOURCES
      ViskoresVolumeRenderer.cpp
      main.cpp
      ../Common/VolumePainterViskores.cpp
      ../Common/VisibilityOrdering.cpp
      ../DirectSend/Base/DirectSendBase.cpp
    HEADERS
      ViskoresVolumeRenderer.hpp
      ../DirectSend/Base/DirectSendBase.hpp
    )

  if(TARGET viskores::rendering)
    target_link_libraries(ViskoresVolumeRenderer PRIVATE viskores::rendering)
  endif()

  if(DEFINED Viskores_INCLUDE_DIRS)
    target_include_directories(ViskoresVolumeRenderer PRIVATE ${Viskores_INCLUDE_DIRS})
  endif()

  if(DEFINED Viskores_LIBRARIES)
    target_link_libraries(ViskoresVolumeRenderer PRIVATE ${Viskores_LIBRARIES})
  endif()

  target_link_libraries(ViskoresVolumeRenderer PRIVATE amrex)
  target_compile_definitions(ViskoresVolumeRenderer PRIVATE AMRVOLUMERENDERER_ENABLE_VISKORES)
else()
  message(STATUS "Skipping ViskoresVolumeRenderer (Viskores not found)")
endif()
