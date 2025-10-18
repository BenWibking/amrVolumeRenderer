// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Copyright (c) 2017
// National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
// the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
// certain rights in this software.
// Additional contributions (C) 2025 Ben Wibking.

#include "SavePNG.hpp"

#include <Common/ImageFull.hpp>
#include <Common/ImageSparse.hpp>

#include <png.h>

#include <cstdio>
#include <memory>
#include <vector>

namespace {

bool doSavePNG(const ImageFull& image, const std::string& filename) {
  FILE* file = std::fopen(filename.c_str(), "wb");
  if (file == nullptr) {
    return false;
  }

  png_structp pngPtr =
      png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
  if (pngPtr == nullptr) {
    std::fclose(file);
    return false;
  }

  png_infop infoPtr = png_create_info_struct(pngPtr);
  if (infoPtr == nullptr) {
    png_destroy_write_struct(&pngPtr, nullptr);
    std::fclose(file);
    return false;
  }

  if (setjmp(png_jmpbuf(pngPtr))) {
    png_destroy_write_struct(&pngPtr, &infoPtr);
    std::fclose(file);
    return false;
  }

  png_init_io(pngPtr, file);
  png_set_IHDR(pngPtr,
               infoPtr,
               static_cast<png_uint_32>(image.getWidth()),
               static_cast<png_uint_32>(image.getHeight()),
               8,
               PNG_COLOR_TYPE_RGB,
               PNG_INTERLACE_NONE,
               PNG_COMPRESSION_TYPE_DEFAULT,
               PNG_FILTER_TYPE_DEFAULT);
  png_write_info(pngPtr, infoPtr);

  std::vector<png_byte> row(
      static_cast<std::size_t>(image.getWidth()) * 3u, png_byte{0});

  for (int y = image.getHeight() - 1; y >= 0; --y) {
    png_bytep rowPtr = row.data();
    for (int x = 0; x < image.getWidth(); ++x) {
      Color color = image.getColor(x, y);
      *rowPtr++ = color.GetComponentAsByte(0);
      *rowPtr++ = color.GetComponentAsByte(1);
      *rowPtr++ = color.GetComponentAsByte(2);
    }

    png_write_row(pngPtr, row.data());
  }

  png_write_end(pngPtr, nullptr);
  png_destroy_write_struct(&pngPtr, &infoPtr);
  std::fclose(file);
  return true;
}

bool doSavePNG(const Image& image, const std::string& filename) {
  const auto* fullImage = dynamic_cast<const ImageFull*>(&image);
  if (fullImage != nullptr) {
    return doSavePNG(*fullImage, filename);
  }

  const auto* sparseImage = dynamic_cast<const ImageSparse*>(&image);
  if (sparseImage != nullptr) {
    return doSavePNG(*sparseImage->uncompress(), filename);
  }

  return false;
}

}  // namespace

bool SavePNG(const Image& image, const std::string& filename) {
  const int totalPixels = image.getWidth() * image.getHeight();
  if ((image.getRegionBegin() == 0) && (image.getRegionEnd() == totalPixels)) {
    return doSavePNG(image, filename);
  }

  // If we only have a region of the image, blend it to a clear image to fill
  // it to its width and height.
  std::unique_ptr<Image> blankImage = image.createNew(0, totalPixels);
  blankImage->clear();
  return doSavePNG(*image.blend(*blankImage), filename);
}
