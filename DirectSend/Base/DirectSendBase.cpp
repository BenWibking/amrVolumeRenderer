// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Copyright (c) 2017
// National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
// the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
// certain rights in this software.

#include "DirectSendBase.hpp"

#include <Common/LayeredImageInterface.hpp>

#include <algorithm>
#include <cassert>
#include <limits>
#include <memory>
#include <numeric>
#include <vector>

constexpr int DEFAULT_MAX_IMAGE_SPLIT = 1000000;

namespace {

struct LayerEntry {
  float depth;
  int owningRank;
  int localIndex;
};

}  // namespace

static int getRealRank(MPI_Group group, int rank, MPI_Comm communicator) {
  MPI_Group commGroup;
  MPI_Comm_group(communicator, &commGroup);

  int realRank;
  MPI_Group_translate_ranks(group, 1, &rank, commGroup, &realRank);

  MPI_Group_free(&commGroup);
  return realRank;
}

static void getPieceRange(int imageSize,
                          int pieceIndex,
                          int numPieces,
                          int& rangeBeginOut,
                          int& rangeEndOut) {
  assert(pieceIndex >= 0);
  assert(pieceIndex < numPieces);

  int pieceSize = imageSize / numPieces;
  rangeBeginOut = pieceSize * pieceIndex;
  if (pieceIndex < numPieces - 1) {
    rangeEndOut = rangeBeginOut + pieceSize;
  } else {
    rangeEndOut = imageSize;
  }
}

static void PostReceives(
    Image* localImage,
    MPI_Group sendGroup,
    MPI_Group recvGroup,
    MPI_Comm communicator,
    std::vector<MPI_Request>& requestsOut,
    std::vector<std::unique_ptr<const Image>>& incomingImagesOut) {
  int localRank;
  MPI_Comm_rank(communicator, &localRank);
  int recvGroupRank;
  MPI_Group_rank(recvGroup, &recvGroupRank);
  if (recvGroupRank == MPI_UNDEFINED) {
    // I am not receiving anything. Just create an "empty" incoming image
    incomingImagesOut.push_back(localImage->window(0, 0));
    return;
  }
  int recvGroupSize;
  MPI_Group_size(recvGroup, &recvGroupSize);

  int sendGroupRank;
  MPI_Group_rank(sendGroup, &sendGroupRank);
  int sendGroupSize;
  MPI_Group_size(sendGroup, &sendGroupSize);

  int rangeBegin;
  int rangeEnd;
  getPieceRange(localImage->getNumberOfPixels(),
                recvGroupRank,
                recvGroupSize,
                rangeBegin,
                rangeEnd);

  incomingImagesOut.resize(sendGroupSize);
  for (int sendGroupIndex = 0; sendGroupIndex < sendGroupSize;
       ++sendGroupIndex) {
    if (sendGroupIndex != sendGroupRank) {
      int sourceRank =
          getRealRank(sendGroup, sendGroupIndex, communicator);
      std::unique_ptr<Image> recvImageBuffer =
          localImage->createNew(rangeBegin, rangeEnd);
      std::vector<MPI_Request> newRequests = recvImageBuffer->IReceive(
          sourceRank, communicator);
      requestsOut.insert(
          requestsOut.end(), newRequests.begin(), newRequests.end());
      incomingImagesOut[sendGroupIndex].reset(recvImageBuffer.release());
    } else {
      // "Sending" to self. Just record a shallow copy of the image.
      incomingImagesOut[sendGroupIndex] =
          localImage->window(rangeBegin, rangeEnd);
    }
  }
}

static void PostSends(
    Image* localImage,
    MPI_Group sendGroup,
    MPI_Group recvGroup,
    MPI_Comm communicator,
    std::vector<MPI_Request>& requestsOut,
    std::vector<std::unique_ptr<const Image>>& outgoingImagesOut) {
  int localRank;
  MPI_Comm_rank(communicator, &localRank);
  int sendGroupRank;
  MPI_Group_rank(sendGroup, &sendGroupRank);
  if (sendGroupRank == MPI_UNDEFINED) {
    // I am not sending anything. Nothing to do.
    return;
  }
  int sendGroupSize;
  MPI_Group_size(sendGroup, &sendGroupSize);

  int recvGroupRank;
  MPI_Group_rank(recvGroup, &recvGroupRank);
  int recvGroupSize;
  MPI_Group_size(recvGroup, &recvGroupSize);

  outgoingImagesOut.resize(recvGroupSize);
  for (int recvGroupIndex = 0; recvGroupIndex < recvGroupSize;
       ++recvGroupIndex) {
    if (recvGroupIndex != recvGroupRank) {
      int rangeBegin;
      int rangeEnd;
      getPieceRange(localImage->getNumberOfPixels(),
                    recvGroupIndex,
                    recvGroupSize,
                    rangeBegin,
                    rangeEnd);
      std::unique_ptr<const Image> outImage =
          localImage->window(rangeBegin, rangeEnd);
      int destRank =
          getRealRank(recvGroup, recvGroupIndex, communicator);
      std::vector<MPI_Request> newRequests = outImage->ISend(
          destRank, communicator);
      requestsOut.insert(
          requestsOut.end(), newRequests.begin(), newRequests.end());
      outgoingImagesOut[recvGroupIndex].swap(outImage);
    } else {
      // Do not need to send. PostReceives just did a shallow copy of the data.
    }
  }
}

static std::unique_ptr<Image> ProcessIncomingImages(
    std::vector<MPI_Request>& requests,
    std::vector<std::unique_ptr<const Image>>& incomingImages) {
  if (requests.size() > 0) {
    MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
  }

  assert(incomingImages.size() > 0);
  if (incomingImages.size() == 1) {
    // Unexpected corner case where there is just one image.
    return incomingImages[0]->deepCopy();
  }

  std::unique_ptr<Image> workingImage =
      incomingImages[0]->blend(*incomingImages[1]);
  for (int imageIndex = 2; imageIndex < incomingImages.size(); ++imageIndex) {
    workingImage = workingImage->blend(*incomingImages[imageIndex]);
  }

  return workingImage;
}

std::unique_ptr<Image> DirectSendBase::compose(Image* localImage,
                                               MPI_Group sendGroup,
                                               MPI_Group recvGroup,
                                               MPI_Comm communicator) {
  std::vector<MPI_Request> recvRequests;
  std::vector<std::unique_ptr<const Image>> incomingImages;
  PostReceives(localImage,
               sendGroup,
               recvGroup,
               communicator,
               recvRequests,
               incomingImages);

  std::vector<MPI_Request> sendRequests;
  std::vector<std::unique_ptr<const Image>> outgoingImages;
  PostSends(localImage,
            sendGroup,
            recvGroup,
            communicator,
            sendRequests,
            outgoingImages);

  std::unique_ptr<Image> resultImage =
      ProcessIncomingImages(recvRequests, incomingImages);

  if (sendRequests.size() > 0) {
    MPI_Waitall(sendRequests.size(), sendRequests.data(), MPI_STATUSES_IGNORE);
  }

  return resultImage;
}

DirectSendBase::DirectSendBase() = default;

std::unique_ptr<Image> DirectSendBase::compose(Image* localImage,
                                               MPI_Group group,
                                               MPI_Comm communicator) {
  if (auto* layered = dynamic_cast<LayeredImageInterface*>(localImage)) {
    return this->composeLayered(localImage, *layered, group, communicator);
  }

  int groupSize;
  MPI_Group_size(group, &groupSize);

  MPI_Group recvGroup;
  int procRange[1][3] = {
      {0, std::min(DEFAULT_MAX_IMAGE_SPLIT, groupSize) - 1, 1}};
  MPI_Group_range_incl(group, 1, procRange, &recvGroup);

  std::unique_ptr<Image> result =
      this->compose(localImage, group, recvGroup, communicator);

  MPI_Group_free(&recvGroup);

  return result;
}

std::unique_ptr<Image> DirectSendBase::composeLayered(
    Image* layeredImage,
    LayeredImageInterface& layers,
    MPI_Group group,
    MPI_Comm communicator) {
  int communicatorSize = 0;
  int communicatorRank = 0;
  MPI_Comm_size(communicator, &communicatorSize);
  MPI_Comm_rank(communicator, &communicatorRank);

  int localLayerCount = layers.getLayerCount();
  std::vector<int> allLayerCounts(static_cast<std::size_t>(communicatorSize), 0);
  MPI_Allgather(&localLayerCount,
                1,
                MPI_INT,
                allLayerCounts.data(),
                1,
                MPI_INT,
                communicator);

  std::vector<int> layerDisplacements(static_cast<std::size_t>(communicatorSize),
                                      0);
  int totalLayerCount = 0;
  for (int proc = 0; proc < communicatorSize; ++proc) {
    layerDisplacements[static_cast<std::size_t>(proc)] = totalLayerCount;
    totalLayerCount += allLayerCounts[static_cast<std::size_t>(proc)];
  }

  std::vector<float> localDepths(static_cast<std::size_t>(localLayerCount),
                                 std::numeric_limits<float>::infinity());
  for (int layerIndex = 0; layerIndex < localLayerCount; ++layerIndex) {
    localDepths[static_cast<std::size_t>(layerIndex)] =
        layers.getLayerDepthHint(layerIndex);
  }

  std::vector<float> allDepths(static_cast<std::size_t>(totalLayerCount),
                               std::numeric_limits<float>::infinity());
  MPI_Allgatherv(localDepths.data(),
                 localLayerCount,
                 MPI_FLOAT,
                 allDepths.data(),
                 allLayerCounts.data(),
                 layerDisplacements.data(),
                 MPI_FLOAT,
                 communicator);

  std::vector<LayerEntry> globalOrder;
  globalOrder.reserve(static_cast<std::size_t>(totalLayerCount));
  for (int proc = 0; proc < communicatorSize; ++proc) {
    const int count = allLayerCounts[static_cast<std::size_t>(proc)];
    const int offset = layerDisplacements[static_cast<std::size_t>(proc)];
    for (int layerIndex = 0; layerIndex < count; ++layerIndex) {
      const int globalIndex = offset + layerIndex;
      LayerEntry entry;
      entry.depth = allDepths[static_cast<std::size_t>(globalIndex)];
      entry.owningRank = proc;
      entry.localIndex = layerIndex;
      globalOrder.push_back(entry);
    }
  }

  std::sort(globalOrder.begin(),
            globalOrder.end(),
            [](const LayerEntry& a, const LayerEntry& b) {
              if (a.depth == b.depth) {
                if (a.owningRank == b.owningRank) {
                  return a.localIndex < b.localIndex;
                }
                return a.owningRank < b.owningRank;
              }
              return a.depth < b.depth;
            });

  int groupSize = 0;
  MPI_Group_size(group, &groupSize);

  MPI_Group recvGroup;
  int procRange[1][3] = {
      {0, std::min(DEFAULT_MAX_IMAGE_SPLIT, groupSize) - 1, 1}};
  MPI_Group_range_incl(group, 1, procRange, &recvGroup);

  std::unique_ptr<Image> accumulatedImage;

  std::size_t index = 0;
  while (index < globalOrder.size()) {
    const int owningRank = globalOrder[index].owningRank;
    const std::size_t runStart = index;
    while (index < globalOrder.size() &&
           globalOrder[index].owningRank == owningRank) {
      ++index;
    }

    Image* localLayerImage = nullptr;
    std::unique_ptr<Image> combinedLocalLayers;
    std::unique_ptr<Image> emptyLayer;

    if (owningRank == communicatorRank) {
      const int firstLocalIndex = globalOrder[runStart].localIndex;
      if (index == runStart + 1) {
        localLayerImage = layers.getLayer(firstLocalIndex);
      } else {
        combinedLocalLayers = layers.getLayer(firstLocalIndex)->deepCopy();
        for (std::size_t runIndex = runStart + 1; runIndex < index; ++runIndex) {
          const int nextLocalIndex = globalOrder[runIndex].localIndex;
          combinedLocalLayers =
              combinedLocalLayers->blend(*layers.getLayer(nextLocalIndex));
        }
        localLayerImage = combinedLocalLayers.get();
      }
    } else {
      emptyLayer = layers.createEmptyLayer(layeredImage->getRegionBegin(),
                                           layeredImage->getRegionEnd());
      emptyLayer->clear(Color(0.0f, 0.0f, 0.0f, 0.0f));
      localLayerImage = emptyLayer.get();
    }

    std::unique_ptr<Image> layerResult = DirectSendBase::compose(
        localLayerImage, group, recvGroup, communicator);

    if (!layerResult) {
      continue;
    }

    if (!accumulatedImage) {
      accumulatedImage = std::move(layerResult);
    } else {
      accumulatedImage = accumulatedImage->blend(*layerResult);
    }
  }

  MPI_Group_free(&recvGroup);

  if (!accumulatedImage) {
    std::unique_ptr<Image> empty = layers.createEmptyLayer(
        layeredImage->getRegionBegin(), layeredImage->getRegionEnd());
    empty->clear(Color(0.0f, 0.0f, 0.0f, 0.0f));
    return empty;
  }

  return accumulatedImage;
}
