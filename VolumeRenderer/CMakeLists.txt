## amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
##

cmake_minimum_required(VERSION 3.14)

include(../CMake/amrVolumeRendererMacros.cmake)
include(FetchContent)

set(AMRVOLUMERENDERER_AMREX_RELEASE_TAG "26.01" CACHE STRING
    "AMReX release tag to fetch from GitHub releases.")
set(AMRVOLUMERENDERER_AMREX_RELEASE_URL
    "https://github.com/AMReX-Codes/amrex/releases/download/${AMRVOLUMERENDERER_AMREX_RELEASE_TAG}/amrex-${AMRVOLUMERENDERER_AMREX_RELEASE_TAG}.tar.gz"
    CACHE STRING "AMReX release tarball URL.")
FetchContent_Declare(amrex
  URL ${AMRVOLUMERENDERER_AMREX_RELEASE_URL}
)

set(AMReX_SPACEDIM 3 CACHE STRING "AMReX dimension" FORCE)
set(AMReX_MPI ON CACHE BOOL "Enable MPI in AMReX" FORCE)
set(AMReX_FORTRAN OFF CACHE BOOL "Disable AMReX Fortran interfaces" FORCE)
set(AMReX_OMP OFF CACHE BOOL "Disable AMReX OpenMP" FORCE)
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

amrVolumeRenderer_executable(volume_renderer
  SOURCES
    VolumeRenderer.cpp
    main.cpp
    ../Common/VolumePainter.cpp
    ../Common/VisibilityOrdering.cpp
    ../DirectSend/Base/DirectSendBase.cpp
  HEADERS
    VolumeRenderer.hpp
    VolumeRendererApi.hpp
    ../DirectSend/Base/DirectSendBase.hpp
)

target_link_libraries(volume_renderer PRIVATE amrex)

add_library(amrVolumeRendererApi STATIC
  VolumeRenderer.cpp
  VolumeRendererApi.cpp
  ../Common/VolumePainter.cpp
  ../Common/VisibilityOrdering.cpp
  ../DirectSend/Base/DirectSendBase.cpp
  VolumeRenderer.hpp
  VolumeRendererApi.hpp
  ../DirectSend/Base/DirectSendBase.hpp
)
set_target_properties(amrVolumeRendererApi PROPERTIES POSITION_INDEPENDENT_CODE ON)
amrVolumeRenderer_target_features(amrVolumeRendererApi)
target_link_libraries(amrVolumeRendererApi PRIVATE amrVolumeRendererCommon amrex)

add_executable(insitu_example
  Examples/RenderFromMultiFab.cpp
)
amrVolumeRenderer_target_features(insitu_example)
target_link_libraries(insitu_example PRIVATE amrVolumeRendererApi amrex)
