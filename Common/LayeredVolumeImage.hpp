// miniGraphics is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#ifndef MINIGRAPHICS_LAYERED_VOLUME_IMAGE_HPP
#define MINIGRAPHICS_LAYERED_VOLUME_IMAGE_HPP

#include <Common/Image.hpp>
#include <Common/ImageRGBAFloatColorDepthSort.hpp>
#include <Common/LayeredImageInterface.hpp>

#include <memory>
#include <vector>

#include <mpi.h>

/// \brief Image composed of ordered RGBA layers with per-layer depth hints.
class LayeredVolumeImage : public Image, public LayeredImageInterface {
 public:
  LayeredVolumeImage(
      int width,
      int height,
      std::vector<std::unique_ptr<ImageRGBAFloatColorDepthSort>> layersIn,
      std::vector<float> depthHintsIn,
      std::unique_ptr<ImageRGBAFloatColorDepthSort> prototypeIn);

  ~LayeredVolumeImage() override;

  int getLayerCount() const override;
  Image* getLayer(int layerIndex) override;
  const Image* getLayer(int layerIndex) const override;
  float getLayerDepthHint(int layerIndex) const override;

  std::unique_ptr<Image> createEmptyLayer(int regionBegin,
                                          int regionEnd) const override;

 protected:
  void clearImpl(const Color& color, float depth) override;
  std::unique_ptr<Image> createNewImpl(int,
                                       int,
                                       int,
                                       int) const override;
  std::unique_ptr<const Image> shallowCopyImpl() const override;
  std::unique_ptr<Image> copySubrange(int, int) const override;
  std::unique_ptr<const Image> window(int, int) const override;
  std::vector<MPI_Request> ISend(int, MPI_Comm) const override;
  std::vector<MPI_Request> IReceive(int, MPI_Comm) override;

 public:
  std::unique_ptr<Image> blend(const Image&) const override;
  bool blendIsOrderDependent() const override;

 private:
  std::vector<std::unique_ptr<ImageRGBAFloatColorDepthSort>> layers;
  std::vector<float> depthHints;
  std::unique_ptr<ImageRGBAFloatColorDepthSort> prototype;
};

#endif  // MINIGRAPHICS_LAYERED_VOLUME_IMAGE_HPP
