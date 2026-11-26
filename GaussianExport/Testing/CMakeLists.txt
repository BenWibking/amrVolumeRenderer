## amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
## See LICENSE.txt for details.

add_executable(SplatOctreeTest SplatOctreeTest.cpp)
amrVolumeRenderer_set_output_dirs(SplatOctreeTest)
set_target_properties(SplatOctreeTest PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF
)
target_link_libraries(SplatOctreeTest PRIVATE amrVolumeRendererSplatOctree)
add_test(NAME SplatOctreeTest COMMAND SplatOctreeTest)

add_executable(PlotfileSplatExportTest PlotfileSplatExportTest.cpp)
amrVolumeRenderer_set_output_dirs(PlotfileSplatExportTest)
amrVolumeRenderer_target_features(PlotfileSplatExportTest)
amrVolumeRenderer_cuda_target_sources(PlotfileSplatExportTest
  PlotfileSplatExportTest.cpp)
target_link_libraries(PlotfileSplatExportTest
  PRIVATE amrVolumeRendererGaussianExport amrex)

set(_splat_test_np 4)
if (MPIEXEC_MAX_NUMPROCS AND MPIEXEC_MAX_NUMPROCS LESS _splat_test_np)
  set(_splat_test_np ${MPIEXEC_MAX_NUMPROCS})
endif()
add_test(NAME PlotfileSplatExportTest
  COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${_splat_test_np}
          ${MPIEXEC_PREFLAGS} $<TARGET_FILE:PlotfileSplatExportTest>
          ${MPIEXEC_POSTFLAGS}
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
set_tests_properties(PlotfileSplatExportTest PROPERTIES TIMEOUT 300)
