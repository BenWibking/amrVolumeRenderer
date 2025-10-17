// miniGraphics is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#ifndef MINIGRAPHICS_VISIBILITY_ORDERING_HPP
#define MINIGRAPHICS_VISIBILITY_ORDERING_HPP

#include <miniGraphicsConfig.h>

#if defined(MINIGRAPHICS_ENABLE_VISKORES)

#include <Common/VolumeTypes.hpp>

#include <mpi.h>

#include <vector>

MPI_Group BuildVisibilityOrderedGroup(
    const minigraphics::volume::CameraParameters& camera,
    float aspect,
    MPI_Group baseGroup,
    int rank,
    int numProcs,
    bool useVisibilityGraph,
    bool writeVisibilityGraph,
    const std::vector<minigraphics::volume::AmrBox>& localBoxes,
    MPI_Comm communicator);

#endif  // MINIGRAPHICS_ENABLE_VISKORES

#endif  // MINIGRAPHICS_VISIBILITY_ORDERING_HPP
