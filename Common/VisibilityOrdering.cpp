// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#include <Common/VisibilityOrdering.hpp>

#include <Common/CameraUtils.hpp>

#include <AMReX_RealVect.H>
#include <AMReX_SmallMatrix.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using Vec3 = amrex::RealVect;
using Vec4 = amrex::SmallMatrix<float, 4, 1>;
using Matrix4x4 = amrex::SmallMatrix<float, 4, 4>;
using amrVolumeRenderer::camera::makeViewMatrix;
using amrVolumeRenderer::camera::safeNormalize;

Matrix4x4 makePerspectiveMatrix(float fovYDegrees,
                                float aspect,
                                float nearPlane,
                                float farPlane) {
  Matrix4x4 matrix = Matrix4x4::Identity();

  constexpr float kPi = 3.14159265358979323846f;
  const float fovTangent =
      std::tan(fovYDegrees * kPi / 180.0f * 0.5f);
  const float size = nearPlane * fovTangent;
  const float left = -size * aspect;
  const float right = size * aspect;
  const float bottom = -size;
  const float top = size;

  matrix(0, 0) = 2.0f * nearPlane / (right - left);
  matrix(1, 1) = 2.0f * nearPlane / (top - bottom);
  matrix(0, 2) = (right + left) / (right - left);
  matrix(1, 2) = (top + bottom) / (top - bottom);
  matrix(2, 2) = -(farPlane + nearPlane) / (farPlane - nearPlane);
  matrix(3, 2) = -1.0f;
  matrix(2, 3) = -(2.0f * farPlane * nearPlane) / (farPlane - nearPlane);
  matrix(3, 3) = 0.0f;

  return matrix;
}

}  // namespace

MPI_Group BuildVisibilityOrderedGroup(
    const amrVolumeRenderer::volume::CameraParameters& camera,
    float aspect,
    MPI_Group baseGroup,
    int rank,
    int numProcs,
    bool useVisibilityGraph,
    bool writeVisibilityGraph,
    const std::vector<amrVolumeRenderer::volume::AmrBox>& localBoxes,
    MPI_Comm communicator) {
  const Matrix4x4 modelview =
      makeViewMatrix(camera.eye, camera.lookAt, camera.up);
  const Matrix4x4 projection = makePerspectiveMatrix(camera.fovYDegrees,
                                                     aspect,
                                                     camera.nearPlane,
                                                     camera.farPlane);

  std::vector<int> defaultOrder(static_cast<std::size_t>(numProcs));
  std::iota(defaultOrder.begin(), defaultOrder.end(), 0);

  auto attemptVisibilityGraphOrdering =
      [&]() -> std::pair<bool, std::vector<int>> {
    const int localBoxCount =
        static_cast<int>(localBoxes.size());
    std::vector<int> allBoxCounts(static_cast<std::size_t>(numProcs), 0);
    MPI_Allgather(&localBoxCount,
                  1,
                  MPI_INT,
                  allBoxCounts.data(),
                  1,
                  MPI_INT,
                  communicator);

    const int totalBoxes =
        std::accumulate(allBoxCounts.begin(), allBoxCounts.end(), 0);
    if (totalBoxes <= 0) {
      return {true, defaultOrder};
    }

    std::vector<int> boxDispls(static_cast<std::size_t>(numProcs), 0);
    for (int i = 1; i < numProcs; ++i) {
      boxDispls[static_cast<std::size_t>(i)] =
          boxDispls[static_cast<std::size_t>(i) - 1] +
          allBoxCounts[static_cast<std::size_t>(i) - 1];
    }

    std::vector<float> localBoxBounds(
        static_cast<std::size_t>(localBoxCount) * 6, 0.0f);
    for (int i = 0; i < localBoxCount; ++i) {
      const auto& box = localBoxes[static_cast<std::size_t>(i)];
      const std::size_t base = static_cast<std::size_t>(i) * 6;
      localBoxBounds[base + 0] = box.minCorner[0];
      localBoxBounds[base + 1] = box.minCorner[1];
      localBoxBounds[base + 2] = box.minCorner[2];
      localBoxBounds[base + 3] = box.maxCorner[0];
      localBoxBounds[base + 4] = box.maxCorner[1];
      localBoxBounds[base + 5] = box.maxCorner[2];
    }

    std::vector<float> allBoxBounds(
        static_cast<std::size_t>(totalBoxes) * 6, 0.0f);
    std::vector<int> countsBounds(static_cast<std::size_t>(numProcs), 0);
    std::vector<int> displsBounds(static_cast<std::size_t>(numProcs), 0);
    for (int i = 0; i < numProcs; ++i) {
      countsBounds[static_cast<std::size_t>(i)] =
          allBoxCounts[static_cast<std::size_t>(i)] * 6;
      displsBounds[static_cast<std::size_t>(i)] =
          boxDispls[static_cast<std::size_t>(i)] * 6;
    }

    MPI_Allgatherv(localBoxBounds.data(),
                   localBoxCount * 6,
                   MPI_FLOAT,
                   allBoxBounds.data(),
                   countsBounds.data(),
                   displsBounds.data(),
                   MPI_FLOAT,
                   communicator);

    std::vector<int> localOwners(static_cast<std::size_t>(localBoxCount),
                                 rank);
    std::vector<int> allOwners(static_cast<std::size_t>(totalBoxes), 0);
    MPI_Allgatherv(localOwners.data(),
                   localBoxCount,
                   MPI_INT,
                   allOwners.data(),
                   allBoxCounts.data(),
                   boxDispls.data(),
                   MPI_INT,
                   communicator);

    struct BoxInfo {
      Vec3 minCorner;
      Vec3 maxCorner;
      int ownerRank = -1;
      float minDepth = std::numeric_limits<float>::infinity();
      float maxDepth = std::numeric_limits<float>::infinity();
    };

    std::vector<BoxInfo> globalBoxes(static_cast<std::size_t>(totalBoxes));

    auto computeDepthRange = [&](const Vec3& minCorner,
                                 const Vec3& maxCorner) {
      float minDepth = std::numeric_limits<float>::infinity();
      float maxDepth = -std::numeric_limits<float>::infinity();
      for (int cornerIndex = 0; cornerIndex < 8; ++cornerIndex) {
        const Vec3 corner((cornerIndex & 1) ? maxCorner[0] : minCorner[0],
                          (cornerIndex & 2) ? maxCorner[1] : minCorner[1],
                          (cornerIndex & 4) ? maxCorner[2] : minCorner[2]);
        const Vec4 homogeneousCorner(static_cast<float>(corner[0]),
                                      static_cast<float>(corner[1]),
                                      static_cast<float>(corner[2]),
                                      1.0f);
        const Vec4 viewSpace = modelview * homogeneousCorner;
        const Vec4 clipSpace = projection * viewSpace;
        if (clipSpace[3] != 0.0f) {
          const float normalizedDepth = clipSpace[2] / clipSpace[3];
          minDepth = std::min(minDepth, normalizedDepth);
          maxDepth = std::max(maxDepth, normalizedDepth);
        }
      }

      if (!std::isfinite(minDepth) || !std::isfinite(maxDepth)) {
        minDepth = std::numeric_limits<float>::infinity();
        maxDepth = std::numeric_limits<float>::infinity();
      }
      return std::make_pair(minDepth, maxDepth);
    };

    for (int boxIndex = 0; boxIndex < totalBoxes; ++boxIndex) {
      const std::size_t base = static_cast<std::size_t>(boxIndex) * 6;
      BoxInfo info;
      info.minCorner = Vec3(allBoxBounds[base + 0],
                            allBoxBounds[base + 1],
                            allBoxBounds[base + 2]);
      info.maxCorner = Vec3(allBoxBounds[base + 3],
                            allBoxBounds[base + 4],
                            allBoxBounds[base + 5]);
      info.ownerRank = allOwners[static_cast<std::size_t>(boxIndex)];
      const auto depthRange =
          computeDepthRange(info.minCorner, info.maxCorner);
      info.minDepth = depthRange.first;
      info.maxDepth = depthRange.second;
      globalBoxes[static_cast<std::size_t>(boxIndex)] = info;
    }

    Vec3 viewDir = safeNormalize(camera.lookAt - camera.eye);

    auto nearlyEqual = [](float a, float b) {
      const float scale = std::max({1.0f, std::fabs(a), std::fabs(b)});
      return std::fabs(a - b) <= 1e-5f * scale;
    };

    auto overlaps = [](float aMin, float aMax, float bMin, float bMax) {
      const float overlapMin = std::max(aMin, bMin);
      const float overlapMax = std::min(aMax, bMax);
      const float scale = std::max(
          {1.0f,
           std::fabs(aMin),
           std::fabs(aMax),
           std::fabs(bMin),
           std::fabs(bMax),
           std::fabs(overlapMin),
           std::fabs(overlapMax)});
      return (overlapMax - overlapMin) > 1e-5f * scale;
    };

    auto updateBoxDepth = [&](BoxInfo& info) {
      const auto depthRange = computeDepthRange(info.minCorner, info.maxCorner);
      info.minDepth = depthRange.first;
      info.maxDepth = depthRange.second;
    };

    std::vector<BoxInfo> boxes = globalBoxes;

    auto compareBoxes = [&](int lhs, int rhs,
                            const std::vector<BoxInfo>& current) {
      const BoxInfo& a = current[static_cast<std::size_t>(lhs)];
      const BoxInfo& b = current[static_cast<std::size_t>(rhs)];
      const bool aFinite = std::isfinite(a.minDepth);
      const bool bFinite = std::isfinite(b.minDepth);
      if (aFinite != bFinite) {
        return aFinite && !bFinite;
      }
      if (a.minDepth == b.minDepth) {
        if (a.maxDepth == b.maxDepth) {
          if (a.ownerRank == b.ownerRank) {
            return lhs < rhs;
          }
          return a.ownerRank < b.ownerRank;
        }
        return a.maxDepth < b.maxDepth;
      }
      return a.minDepth < b.minDepth;
    };

    constexpr float kDirectionTolerance = 1e-6f;

    auto rebuildAdjacency = [&](const std::vector<BoxInfo>& currentBoxes,
                                std::vector<std::vector<int>>& adjacency,
                                std::vector<int>& indegree) {
      const int boxCount = static_cast<int>(currentBoxes.size());
      adjacency.assign(static_cast<std::size_t>(boxCount), {});
      indegree.assign(static_cast<std::size_t>(boxCount), 0);

      auto addEdge = [&](int from, int to) {
        if (from == to) {
          return;
        }
        auto& edges = adjacency[static_cast<std::size_t>(from)];
        if (std::find(edges.begin(), edges.end(), to) == edges.end()) {
          edges.push_back(to);
          ++indegree[static_cast<std::size_t>(to)];
        }
      };

      for (int i = 0; i < boxCount; ++i) {
        const BoxInfo& a = currentBoxes[static_cast<std::size_t>(i)];
        for (int j = i + 1; j < boxCount; ++j) {
          const BoxInfo& b = currentBoxes[static_cast<std::size_t>(j)];

          for (int axis = 0; axis < 3; ++axis) {
            const int axis1 = (axis + 1) % 3;
            const int axis2 = (axis + 2) % 3;
            const bool overlapAxis1 = overlaps(a.minCorner[axis1],
                                               a.maxCorner[axis1],
                                               b.minCorner[axis1],
                                               b.maxCorner[axis1]);
            const bool overlapAxis2 = overlaps(a.minCorner[axis2],
                                               a.maxCorner[axis2],
                                               b.minCorner[axis2],
                                               b.maxCorner[axis2]);
            if (!overlapAxis1 || !overlapAxis2) {
              continue;
            }

            const float dirComponent = viewDir[axis];

            if (nearlyEqual(a.maxCorner[axis], b.minCorner[axis])) {
              if (dirComponent > kDirectionTolerance) {
                addEdge(j, i);
              } else if (dirComponent < -kDirectionTolerance) {
                addEdge(i, j);
              }
            } else if (nearlyEqual(b.maxCorner[axis], a.minCorner[axis])) {
              if (dirComponent > kDirectionTolerance) {
                addEdge(i, j);
              } else if (dirComponent < -kDirectionTolerance) {
                addEdge(j, i);
              }
            }
          }
        }
      }
    };

    auto exportGraph = [&](const std::vector<BoxInfo>& currentBoxes,
                           const std::vector<std::vector<int>>& adjacency) {
      if (!writeVisibilityGraph || rank != 0) {
        return;
      }
      static int graphFileCounter = 0;
      std::ostringstream filename;
      filename << "visibility_graph_" << graphFileCounter++ << ".dot";
      std::ofstream dotFile(filename.str());
      if (!dotFile) {
        std::cerr << "Failed to write visibility graph to '" << filename.str()
                  << "'" << std::endl;
        return;
      }
      dotFile << "digraph VisibilityGraph {\n";
      dotFile << "  rankdir=LR;\n";
      dotFile << std::fixed << std::setprecision(6);
      const int boxCount = static_cast<int>(currentBoxes.size());
      for (int idx = 0; idx < boxCount; ++idx) {
        const BoxInfo& info = currentBoxes[static_cast<std::size_t>(idx)];
        dotFile << "  box" << idx << " [label=\"box " << idx
                << "\\nrank " << info.ownerRank
                << "\\nminDepth " << info.minDepth
                << "\\nmaxDepth " << info.maxDepth << "\"];\n";
      }
      for (std::size_t from = 0; from < adjacency.size(); ++from) {
        for (int to : adjacency[from]) {
          dotFile << "  box" << from << " -> box" << to << ";\n";
        }
      }
      dotFile << "}\n";
      std::cout << "Wrote visibility graph to '" << filename.str() << "'"
                << std::endl;
    };

    struct TopoResult {
      bool success = false;
      std::vector<int> order;
      std::vector<int> residualIndegree;
    };

    auto topoSortBoxes =
        [&](const std::vector<std::vector<int>>& adjacency,
            const std::vector<int>& indegreeInitial,
            const std::vector<BoxInfo>& currentBoxes) -> TopoResult {
      TopoResult result;
      const int boxCount = static_cast<int>(currentBoxes.size());
      std::vector<int> indegreeCopy = indegreeInitial;
      std::vector<int> ready;
      ready.reserve(static_cast<std::size_t>(boxCount));
      for (int i = 0; i < boxCount; ++i) {
        if (indegreeCopy[static_cast<std::size_t>(i)] == 0) {
          ready.push_back(i);
        }
      }

      auto sortReady = [&]() {
        std::sort(
            ready.begin(),
            ready.end(),
            [&](int lhs, int rhs) { return compareBoxes(lhs, rhs, currentBoxes); });
      };

      sortReady();
      while (!ready.empty()) {
        const int current = ready.front();
        ready.erase(ready.begin());
        result.order.push_back(current);

        for (int next : adjacency[static_cast<std::size_t>(current)]) {
          int& in = indegreeCopy[static_cast<std::size_t>(next)];
          --in;
          if (in == 0) {
            ready.push_back(next);
          }
        }
        sortReady();
      }

      result.residualIndegree = std::move(indegreeCopy);
      result.success = static_cast<int>(result.order.size()) == boxCount;
      return result;
    };

    auto findCycle = [&](const std::vector<std::vector<int>>& adjacency,
                         const std::vector<int>& residualIndegree)
        -> std::vector<int> {
      const int boxCount = static_cast<int>(adjacency.size());
      std::vector<int> state(static_cast<std::size_t>(boxCount), 0);
      std::vector<int> parent(static_cast<std::size_t>(boxCount), -1);
      std::vector<int> cycle;

      std::function<bool(int)> dfs = [&](int node) -> bool {
        state[static_cast<std::size_t>(node)] = 1;
        for (int next : adjacency[static_cast<std::size_t>(node)]) {
          if (state[static_cast<std::size_t>(next)] == 0) {
            parent[static_cast<std::size_t>(next)] = node;
            if (dfs(next)) {
              return true;
            }
          } else if (state[static_cast<std::size_t>(next)] == 1) {
            cycle.clear();
            cycle.push_back(next);
            for (int cur = node; cur != next && cur != -1;
                 cur = parent[static_cast<std::size_t>(cur)]) {
              cycle.push_back(cur);
            }
              std::reverse(cycle.begin(), cycle.end());
            return true;
          }
        }
        state[static_cast<std::size_t>(node)] = 2;
        return false;
      };

      for (int node = 0; node < boxCount; ++node) {
        if (residualIndegree[static_cast<std::size_t>(node)] > 0 &&
            state[static_cast<std::size_t>(node)] == 0) {
          if (dfs(node)) {
            break;
          }
        }
      }
      return cycle;
    };

    auto breakCycle = [&](const std::vector<int>& cycleNodes,
                          std::vector<BoxInfo>& mutableBoxes) -> bool {
      if (cycleNodes.size() < 2) {
        return false;
      }

      int chosenAxis = 0;
      float bestAlignment = std::fabs(viewDir[0]);
      for (int axis = 1; axis < 3; ++axis) {
        const float alignment = std::fabs(viewDir[axis]);
        if (alignment > bestAlignment) {
          bestAlignment = alignment;
          chosenAxis = axis;
        }
      }

      if (bestAlignment <= kDirectionTolerance) {
        float widestLength = -1.0f;
        for (int axis = 0; axis < 3; ++axis) {
          for (int index : cycleNodes) {
            const BoxInfo& box = mutableBoxes[static_cast<std::size_t>(index)];
            const float length = box.maxCorner[axis] - box.minCorner[axis];
            if (length > widestLength) {
              widestLength = length;
              chosenAxis = axis;
            }
          }
        }
      }

      const float dirComponent = viewDir[chosenAxis];
      if (std::fabs(dirComponent) <= kDirectionTolerance) {
        return false;
      }

      const float minLengthTolerance = 1e-6f;
      int targetIndex = cycleNodes.front();
      float targetLength = -1.0f;
      for (int index : cycleNodes) {
        const BoxInfo& box = mutableBoxes[static_cast<std::size_t>(index)];
        const float length =
            box.maxCorner[chosenAxis] - box.minCorner[chosenAxis];
        if (length > targetLength && length > minLengthTolerance) {
          targetLength = length;
          targetIndex = index;
        }
      }

      if (targetLength <= minLengthTolerance) {
        return false;
      }

      const BoxInfo targetBox =
          mutableBoxes[static_cast<std::size_t>(targetIndex)];
      const float minVal = targetBox.minCorner[chosenAxis];
      const float maxVal = targetBox.maxCorner[chosenAxis];
      const float length = maxVal - minVal;
      const float epsilon = std::max(1e-5f * length, 1e-6f);

      std::vector<float> candidates;
      for (int index : cycleNodes) {
        if (index == targetIndex) {
          continue;
        }
        const BoxInfo& other = mutableBoxes[static_cast<std::size_t>(index)];
        const float otherMin = other.minCorner[chosenAxis];
        const float otherMax = other.maxCorner[chosenAxis];
        if (otherMin > minVal + epsilon && otherMin < maxVal - epsilon) {
          candidates.push_back(otherMin);
        }
        if (otherMax > minVal + epsilon && otherMax < maxVal - epsilon) {
          candidates.push_back(otherMax);
        }
      }

      float split = 0.5f * (minVal + maxVal);
      if (!candidates.empty()) {
        if (dirComponent > 0.0f) {
          split = *std::max_element(candidates.begin(), candidates.end());
        } else {
          split = *std::min_element(candidates.begin(), candidates.end());
        }
      }

      if (split <= minVal + epsilon) {
        split = minVal + epsilon;
      }
      if (split >= maxVal - epsilon) {
        split = maxVal - epsilon;
      }

      if (!(split > minVal && split < maxVal)) {
        return false;
      }

      BoxInfo nearBox = targetBox;
      BoxInfo farBox = targetBox;
      if (dirComponent > 0.0f) {
        nearBox.maxCorner[chosenAxis] = split;
        farBox.minCorner[chosenAxis] = split;
      } else {
        nearBox.minCorner[chosenAxis] = split;
        farBox.maxCorner[chosenAxis] = split;
      }

      updateBoxDepth(nearBox);
      updateBoxDepth(farBox);

      mutableBoxes[static_cast<std::size_t>(targetIndex)] = nearBox;
      mutableBoxes.push_back(farBox);

      if (rank == 0) {
        std::cout << "Split box owned by rank " << targetBox.ownerRank
                  << " along axis " << chosenAxis << " at " << split
                  << " to break visibility cycle." << std::endl;
      }

      return true;
    };

    const int maxIterations =
        static_cast<int>(std::max<std::size_t>(globalBoxes.size(), 1)) * 8 + 32;
    std::vector<std::vector<int>> adjacency;
    std::vector<int> indegree;
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
      rebuildAdjacency(boxes, adjacency, indegree);
      exportGraph(boxes, adjacency);

      const TopoResult topo =
          topoSortBoxes(adjacency, indegree, boxes);
      if (topo.success) {
        std::vector<int> rankVisited(static_cast<std::size_t>(numProcs), 0);
        std::vector<int> rankOrder;
        rankOrder.reserve(static_cast<std::size_t>(numProcs));
        for (int boxIndex : topo.order) {
          const int owner =
              boxes[static_cast<std::size_t>(boxIndex)].ownerRank;
          if (owner >= 0 &&
              rankVisited[static_cast<std::size_t>(owner)] == 0) {
            rankVisited[static_cast<std::size_t>(owner)] = 1;
            rankOrder.push_back(owner);
          }
        }
        for (int owner : defaultOrder) {
          if (rankVisited[static_cast<std::size_t>(owner)] == 0) {
            rankVisited[static_cast<std::size_t>(owner)] = 1;
            rankOrder.push_back(owner);
          }
        }
        return {true, std::move(rankOrder)};
      }

      auto cycleNodes = findCycle(adjacency, topo.residualIndegree);
      if (cycleNodes.empty()) {
        return {false, {}};
      }

      if (!breakCycle(cycleNodes, boxes)) {
        return {false, {}};
      }
    }

    return {false, {}};
  };

  std::vector<int> rankOrder;
  if (useVisibilityGraph) {
    auto result = attemptVisibilityGraphOrdering();
    if (result.first) {
      rankOrder = std::move(result.second);
    } else {
      static bool warnedGraphFailure = false;
      if (!warnedGraphFailure && rank == 0) {
        std::cerr << "Visibility graph ordering failed; "
                     "falling back to default MPI rank order."
                  << std::endl;
      }
      warnedGraphFailure = true;
      rankOrder = defaultOrder;
    }
  } else {
    rankOrder = defaultOrder;
  }

  MPI_Group orderedGroup = MPI_GROUP_NULL;
  MPI_Group_incl(baseGroup, numProcs, rankOrder.data(), &orderedGroup);
  return orderedGroup;
}
