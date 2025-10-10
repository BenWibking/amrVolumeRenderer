#ifndef LAYERED_IMAGE_INTERFACE_HPP
#define LAYERED_IMAGE_INTERFACE_HPP

#include <memory>

class Image;

/// \brief Interface for images that provide multiple compositing layers.
class LayeredImageInterface {
 public:
  virtual ~LayeredImageInterface() = default;

  /// \brief Returns the number of layers stored on this rank.
  virtual int getLayerCount() const = 0;

  /// \brief Returns a mutable pointer to the image for the given layer index.
  virtual Image* getLayer(int layerIndex) = 0;

  /// \brief Returns a const pointer to the image for the given layer index.
  virtual const Image* getLayer(int layerIndex) const = 0;

  /// \brief Returns the depth hint for the layer. Smaller values are nearer.
  virtual float getLayerDepthHint(int layerIndex) const = 0;

  /// \brief Creates an empty image layer with the given region.
  virtual std::unique_ptr<Image> createEmptyLayer(int regionBegin,
                                                  int regionEnd) const = 0;
};

#endif  // LAYERED_IMAGE_INTERFACE_HPP
