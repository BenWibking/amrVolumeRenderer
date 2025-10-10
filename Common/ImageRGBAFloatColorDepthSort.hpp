#ifndef IMAGERGBAFLOATCOLORDEPTHSORT_HPP
#define IMAGERGBAFLOATCOLORDEPTHSORT_HPP

#include "ImageColorOnly.hpp"

#include <algorithm>
#include <limits>

struct ImageRGBAFloatColorDepthSortFeatures {
  using ColorType = float;
  static constexpr int ColorVecSize = 5;

  static void blend(const ColorType topColor[ColorVecSize],
                    const ColorType bottomColor[ColorVecSize],
                    ColorType outColor[ColorVecSize]) {
    const float topDepth = topColor[4];
    const float bottomDepth = bottomColor[4];
    const bool topIsFront = topDepth <= bottomDepth;
    const ColorType* front = topIsFront ? topColor : bottomColor;
    const ColorType* back = topIsFront ? bottomColor : topColor;

    for (int component = 0; component < 4; ++component) {
      outColor[component] =
          front[component] + back[component] * (1.0f - front[3]);
    }
    outColor[4] = std::min(topDepth, bottomDepth);
  }

  static void encodeColor(const Color& color,
                          ColorType colorComponents[ColorVecSize]) {
    colorComponents[0] = color.Components[0];
    colorComponents[1] = color.Components[1];
    colorComponents[2] = color.Components[2];
    colorComponents[3] = color.Components[3];
    colorComponents[4] = std::numeric_limits<float>::infinity();
  }

  static Color decodeColor(const ColorType colorComponents[ColorVecSize]) {
    return Color(colorComponents[0],
                 colorComponents[1],
                 colorComponents[2],
                 colorComponents[3]);
  }
};

class ImageRGBAFloatColorDepthSort
    : public ImageColorOnly<ImageRGBAFloatColorDepthSortFeatures> {
 public:
  ImageRGBAFloatColorDepthSort(int _width, int _height);
  ImageRGBAFloatColorDepthSort(int _width,
                               int _height,
                               int _regionBegin,
                               int _regionEnd);
  ~ImageRGBAFloatColorDepthSort() = default;

  std::unique_ptr<ImageSparse> compress() const final;

  void setDepthHint(int pixelIndex, float depth);
  void setDepthHint(int x, int y, float depth);

 protected:
  std::unique_ptr<Image> createNewImpl(int _width,
                                       int _height,
                                       int _regionBegin,
                                       int _regionEnd) const final;

  std::unique_ptr<const Image> shallowCopyImpl() const final;
};

#endif  // IMAGERGBAFLOATCOLORDEPTHSORT_HPP
