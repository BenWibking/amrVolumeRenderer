// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Copyright (c) 2017
// National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
// the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
// certain rights in this software.
// Additional contributions (C) 2025 Ben Wibking.

#ifndef SAVEPNG_HPP
#define SAVEPNG_HPP

#include <string>

class Image;

/// \brief Saves the given Image data to a PNG file.
bool SavePNG(const Image& image, const std::string& filename);

#endif  // SAVEPNG_HPP
