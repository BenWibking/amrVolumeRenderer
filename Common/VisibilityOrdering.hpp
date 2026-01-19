// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#ifndef AMRVOLUMERENDERER_VISIBILITY_ORDERING_HPP
#define AMRVOLUMERENDERER_VISIBILITY_ORDERING_HPP

#include <Common/VolumeTypes.hpp>

#include <mpi.h>

#include <vector>

MPI_Group BuildVisibilityOrderedGroup(
    const amrVolumeRenderer::volume::CameraParameters& camera,
    float aspect,
    MPI_Group baseGroup,
    int rank,
    int numProcs,
    bool useVisibilityGraph,
    bool writeVisibilityGraph,
    const std::vector<amrVolumeRenderer::volume::AmrBox>& localBoxes,
    MPI_Comm communicator);

#endif  // AMRVOLUMERENDERER_VISIBILITY_ORDERING_HPP
