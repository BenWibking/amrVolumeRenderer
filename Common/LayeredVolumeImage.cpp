// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.

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

std::unique_ptr<const Image> LayeredVolumeImage::window(int subregionBegin,
                                                        int subregionEnd) const {
  if (subregionBegin < 0 || subregionEnd < subregionBegin) {
    throw std::out_of_range("LayeredVolumeImage::window invalid subregion");
  }

  const int pixelCount = this->getNumberOfPixels();
  if (subregionEnd > pixelCount) {
    throw std::out_of_range("LayeredVolumeImage::window subregion past end");
  }

  std::vector<std::unique_ptr<ImageRGBAFloatColorDepthSort>> windowLayers;
  windowLayers.reserve(this->layers.size());
  for (const auto& layer : this->layers) {
    std::unique_ptr<const Image> layerWindow =
        layer->window(subregionBegin, subregionEnd);
    const auto* typedWindow =
        dynamic_cast<const ImageRGBAFloatColorDepthSort*>(layerWindow.release());
    if (typedWindow == nullptr) {
      throw std::logic_error(
          "LayeredVolumeImage::window unexpected layer image type");
    }
    windowLayers.emplace_back(
        const_cast<ImageRGBAFloatColorDepthSort*>(typedWindow));
  }

  std::unique_ptr<ImageRGBAFloatColorDepthSort> prototypeCopy;
  if (this->prototype) {
    prototypeCopy.reset(new ImageRGBAFloatColorDepthSort(*this->prototype));
  }

  auto windowedImage = std::make_unique<LayeredVolumeImage>(
      this->getWidth(),
      this->getHeight(),
      std::move(windowLayers),
      this->depthHints,
      std::move(prototypeCopy));
  windowedImage->resizeRegion(subregionBegin + this->getRegionBegin(),
                              subregionEnd + this->getRegionBegin());
  windowedImage->setValidViewport(this->getValidViewport());

  return std::unique_ptr<const Image>(windowedImage.release());
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
