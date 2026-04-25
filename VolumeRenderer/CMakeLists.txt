## amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.
##
##

cmake_minimum_required(VERSION 3.14)

include(../CMake/amrVolumeRendererMacros.cmake)

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

amrVolumeRenderer_cuda_target_sources(volume_renderer
  VolumeRenderer.cpp
  main.cpp
  ../Common/VisibilityOrdering.cpp
  ../Common/VolumePainter.cpp)
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
amrVolumeRenderer_set_output_dirs(amrVolumeRendererApi)
amrVolumeRenderer_target_features(amrVolumeRendererApi)
amrVolumeRenderer_cuda_target_sources(amrVolumeRendererApi
  VolumeRenderer.cpp
  VolumeRendererApi.cpp
  ../Common/VisibilityOrdering.cpp
  ../Common/VolumePainter.cpp)
target_link_libraries(amrVolumeRendererApi PUBLIC amrVolumeRendererCommon amrex)

add_executable(insitu_example
  Examples/RenderFromMultiFab.cpp
)
amrVolumeRenderer_set_output_dirs(insitu_example)
amrVolumeRenderer_target_features(insitu_example)
amrVolumeRenderer_cuda_target_sources(insitu_example Examples/RenderFromMultiFab.cpp)
target_link_libraries(insitu_example PRIVATE amrVolumeRendererApi amrex)
