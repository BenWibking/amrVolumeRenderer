// miniGraphics is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Copyright (c) 2017
// National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
// the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
// certain rights in this software.

#include <Common/LayeredVolumeImage.hpp>

#include <Common/Color.hpp>

#include <stdexcept>

LayeredVolumeImage::LayeredVolumeImage(
    int width,
    int height,
    std::vector<std::unique_ptr<ImageRGBAFloatColorDepthSort>> layersIn,
    std::vector<float> depthHintsIn,
    std::unique_ptr<ImageRGBAFloatColorDepthSort> prototypeIn)
    : Image(width, height),
      layers(std::move(layersIn)),
      depthHints(std::move(depthHintsIn)),
      prototype(std::move(prototypeIn)) {}

LayeredVolumeImage::~LayeredVolumeImage() = default;

int LayeredVolumeImage::getLayerCount() const {
  return static_cast<int>(this->layers.size());
}

Image* LayeredVolumeImage::getLayer(int layerIndex) {
  return this->layers.at(static_cast<std::size_t>(layerIndex)).get();
}

const Image* LayeredVolumeImage::getLayer(int layerIndex) const {
  return this->layers.at(static_cast<std::size_t>(layerIndex)).get();
}

float LayeredVolumeImage::getLayerDepthHint(int layerIndex) const {
  return this->depthHints.at(static_cast<std::size_t>(layerIndex));
}

std::unique_ptr<Image> LayeredVolumeImage::createEmptyLayer(
    int regionBegin, int regionEnd) const {
  std::unique_ptr<Image> empty =
      this->prototype->createNew(regionBegin, regionEnd);
  empty->clear(Color(0.0f, 0.0f, 0.0f, 0.0f));
  return empty;
}

void LayeredVolumeImage::clearImpl(const Color& color, float depth) {
  for (auto& layer : this->layers) {
    layer->clear(color, depth);
  }
}

std::unique_ptr<Image> LayeredVolumeImage::createNewImpl(int,
                                                         int,
                                                         int,
                                                         int) const {
  throw std::logic_error("LayeredVolumeImage::createNewImpl not supported");
}

std::unique_ptr<const Image> LayeredVolumeImage::shallowCopyImpl() const {
  throw std::logic_error("LayeredVolumeImage::shallowCopyImpl not supported");
}

std::unique_ptr<Image> LayeredVolumeImage::copySubrange(int, int) const {
  throw std::logic_error("LayeredVolumeImage::copySubrange not supported");
}

std::unique_ptr<const Image> LayeredVolumeImage::window(int, int) const {
  throw std::logic_error("LayeredVolumeImage::window not supported");
}

std::vector<MPI_Request> LayeredVolumeImage::ISend(int, MPI_Comm) const {
  throw std::logic_error("LayeredVolumeImage::ISend not supported");
}

std::vector<MPI_Request> LayeredVolumeImage::IReceive(int, MPI_Comm) {
  throw std::logic_error("LayeredVolumeImage::IReceive not supported");
}

std::unique_ptr<Image> LayeredVolumeImage::blend(const Image&) const {
  throw std::logic_error("LayeredVolumeImage::blend not supported");
}

bool LayeredVolumeImage::blendIsOrderDependent() const {
  return true;
}
