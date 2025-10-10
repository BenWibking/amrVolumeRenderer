#include "ImageRGBAFloatColorDepthSort.hpp"

#include "ImageSparseColorOnly.hpp"

ImageRGBAFloatColorDepthSort::ImageRGBAFloatColorDepthSort(int _width,
                                                           int _height)
    : ImageColorOnly(_width, _height) {}

ImageRGBAFloatColorDepthSort::ImageRGBAFloatColorDepthSort(int _width,
                                                           int _height,
                                                           int _regionBegin,
                                                           int _regionEnd)
    : ImageColorOnly(_width, _height, _regionBegin, _regionEnd) {}

std::unique_ptr<ImageSparse> ImageRGBAFloatColorDepthSort::compress() const {
  return std::unique_ptr<ImageSparse>(
      new ImageSparseColorOnly<ImageRGBAFloatColorDepthSortFeatures>(*this));
}

void ImageRGBAFloatColorDepthSort::setDepthHint(int pixelIndex, float depth) {
  ImageRGBAFloatColorDepthSortFeatures::ColorType* buffer =
      this->getColorBuffer(pixelIndex);
  buffer[4] = depth;
}

void ImageRGBAFloatColorDepthSort::setDepthHint(int x, int y, float depth) {
  this->setDepthHint(this->pixelIndex(x, y), depth);
}

std::unique_ptr<Image> ImageRGBAFloatColorDepthSort::createNewImpl(
    int _width, int _height, int _regionBegin, int _regionEnd) const {
  return std::unique_ptr<Image>(new ImageRGBAFloatColorDepthSort(
      _width, _height, _regionBegin, _regionEnd));
}

std::unique_ptr<const Image> ImageRGBAFloatColorDepthSort::shallowCopyImpl()
    const {
  return std::unique_ptr<const Image>(
      new ImageRGBAFloatColorDepthSort(*this));
}
