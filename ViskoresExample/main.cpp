// ABOUTME: Main entry point for Viskores painter example
// ABOUTME: Demonstrates distributed rendering with rank-specific data

// miniGraphics is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Copyright (c) 2017
// National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
// the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
// certain rights in this software.

#include "ViskoresExample.hpp"

#include <DirectSend/Base/DirectSendBase.hpp>

#include <Common/ImageRGBAUByteColorFloatDepth.hpp>
#include <Common/ImageSparse.hpp>
#include <Common/MakeBox.hpp>
#include <Common/SavePPM.hpp>
#include <Common/YamlWriter.hpp>

#include <glm/gtc/constants.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  int width = 512;
  int height = 512;
  int trials = 1;
  std::string yamlOutput;
};

void printUsage() {
  std::cout << "Usage: ViskoresExample [--width W] [--height H] "
               "[--trials N] [--yaml-output FILE]\n"
            << "  --width W        Image width (default: 512)\n"
            << "  --height H       Image height (default: 512)\n"
            << "  --trials N       Number of render trials (default: 1)\n"
            << "  --yaml-output F  Write timing information to YAML file F\n"
            << "  -h, --help       Show this help message\n";
}

Options parseOptions(int argc, char** argv, int rank, bool& exitEarly) {
  Options options;
  exitEarly = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg(argv[i]);
    if (arg == "--width") {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for --width");
      }
      options.width = std::stoi(argv[++i]);
      if (options.width <= 0) {
        throw std::runtime_error("image width must be positive");
      }
    } else if (arg == "--height") {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for --height");
      }
      options.height = std::stoi(argv[++i]);
      if (options.height <= 0) {
        throw std::runtime_error("image height must be positive");
      }
    } else if (arg == "--trials") {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for --trials");
      }
      options.trials = std::stoi(argv[++i]);
      if (options.trials <= 0) {
        throw std::runtime_error("number of trials must be positive");
      }
    } else if (arg == "--yaml-output") {
      if (i + 1 >= argc) {
        throw std::runtime_error("missing value for --yaml-output");
      }
      options.yamlOutput = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      if (rank == 0) {
        printUsage();
      }
      exitEarly = true;
      return options;
    } else {
      std::ostringstream message;
      message << "unknown option '" << arg << "'";
      throw std::runtime_error(message.str());
    }
  }

  return options;
}

struct MpiGroupGuard {
  MpiGroupGuard() : group(MPI_GROUP_NULL) {}
  ~MpiGroupGuard() {
    if (group != MPI_GROUP_NULL) {
      MPI_Group_free(&group);
    }
  }

  MPI_Group group;
};

constexpr unsigned int kDefaultCameraSeed = 53321u;

}  // namespace

ViskoresExample::ViskoresExample() : rank(0), numProcs(1) {
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
}

void ViskoresExample::initialize() const {
  if (rank == 0) {
    std::cout << "ViskoresExample: Using Viskores painter on " << numProcs
              << " ranks" << std::endl;
  }
}

void ViskoresExample::createRankSpecificMesh(Mesh& mesh) const {
  constexpr int boxesX = 2;
  constexpr int boxesY = 2;
  constexpr int boxesZ = 2;
  constexpr int totalBoxes = boxesX * boxesY * boxesZ;
  const int ranks = std::max(1, numProcs);
  const int boxesPerRank = totalBoxes / ranks;
  const int remainder = totalBoxes % ranks;
  const int localBoxCount = boxesPerRank + ((rank < remainder) ? 1 : 0);
  const int firstBoxIndex =
      boxesPerRank * rank + std::min(rank, remainder);

  if (localBoxCount <= 0 || firstBoxIndex >= totalBoxes) {
    mesh = Mesh();
    return;
  }

  const int clampedBoxCount =
      std::min(localBoxCount, totalBoxes - firstBoxIndex);
  if (clampedBoxCount <= 0) {
    mesh = Mesh();
    return;
  }

  Mesh combined;
  const int boxesPerLayer = boxesX * boxesY;
  const float spacing = 2.0f;
  const float boxScale = 0.8f;
  const float twoPi = glm::two_pi<float>();

  for (int localIndex = 0; localIndex < clampedBoxCount; ++localIndex) {
    const int boxIndex = firstBoxIndex + localIndex;
    Mesh boxMesh;
    MakeBox(boxMesh);

    const int layer = boxIndex / boxesPerLayer;
    const int inLayerIndex = boxIndex % boxesPerLayer;
    const int row = inLayerIndex / boxesX;
    const int col = inLayerIndex % boxesX;

    glm::vec3 offset(0.0f);
    offset.x =
        (static_cast<float>(col) - (boxesX - 1) * 0.5f) * spacing;
    offset.y =
        ((boxesY - 1) * 0.5f - static_cast<float>(row)) * spacing;
    offset.z =
        (static_cast<float>(layer) - (boxesZ - 1) * 0.5f) * spacing;

    glm::mat4 transform(1.0f);
    transform = glm::translate(transform, offset);
    transform = glm::scale(transform, glm::vec3(boxScale));
    transform = glm::translate(transform, glm::vec3(-0.5f));
    boxMesh.transform(transform);

    const float hue = static_cast<float>(boxIndex) / static_cast<float>(totalBoxes);
    const Color boxColor(0.5f + 0.5f * std::sin(hue * twoPi),
                         0.5f + 0.5f * std::sin(hue * twoPi + twoPi / 3.0f),
                         0.5f + 0.5f * std::sin(hue * twoPi + 2.0f * twoPi / 3.0f),
                         1.0f);
    boxMesh.setHomogeneousColor(boxColor);

    combined.append(boxMesh);
  }

  mesh = combined;
}

double ViskoresExample::paint(ImageFull& image,
                              const glm::mat4& modelview,
                              const glm::mat4& projection) {
  Mesh mesh;
  createRankSpecificMesh(mesh);

  const auto startTime = std::chrono::high_resolution_clock::now();
  painter.paint(mesh, image, modelview, projection);
  const auto endTime = std::chrono::high_resolution_clock::now();

  const double seconds =
      std::chrono::duration_cast<std::chrono::duration<double>>(endTime - startTime)
          .count();

  return seconds;
}

Compositor* ViskoresExample::getCompositor() {
  static DirectSendBase compositor;
  return &compositor;
}

int ViskoresExample::run(int argc, char** argv) {
  bool exitEarly = false;
  Options options;

  try {
    options = parseOptions(argc, argv, rank, exitEarly);
  } catch (const std::exception& error) {
    if (rank == 0) {
      std::cerr << "Error parsing options: " << error.what() << std::endl;
      std::cerr << "Use --help to list available options." << std::endl;
    }
    return 1;
  }

  if (exitEarly) {
    return 0;
  }

  initialize();

  Compositor* compositor = getCompositor();
  if (compositor == nullptr) {
    if (rank == 0) {
      std::cerr << "No compositor available for the example." << std::endl;
    }
    return 1;
  }

  std::ofstream yamlFile;
  std::ostringstream yamlBuffer;
  std::ostream* yamlStream = nullptr;
  if (!options.yamlOutput.empty()) {
    yamlFile.open(options.yamlOutput.c_str());
    if (!yamlFile.good()) {
      if (rank == 0) {
        std::cerr << "Failed to open YAML output file '" << options.yamlOutput
                  << "'." << std::endl;
      }
      return 1;
    }
    yamlStream = &yamlFile;
  } else {
    yamlStream = &yamlBuffer;
  }
  YamlWriter yaml(*yamlStream);

  MpiGroupGuard groupGuard;
  MPI_Comm_group(MPI_COMM_WORLD, &groupGuard.group);

  double accumulatedMaxPaintSeconds = 0.0;
  for (int trial = 0; trial < options.trials; ++trial) {
    ImageRGBAUByteColorFloatDepth localImage(options.width, options.height);

    const float aspect =
        static_cast<float>(options.width) / static_cast<float>(options.height);
    constexpr int boxesX = 2;
    constexpr int boxesY = 2;
    constexpr int boxesZ = 2;
    constexpr float spacing = 2.0f;
    constexpr float boxScale = 0.8f;
    const float gridHalfWidth =
        ((boxesX - 1) * spacing) * 0.5f + boxScale * 0.5f;
    const float gridHalfHeight =
        ((boxesY - 1) * spacing) * 0.5f + boxScale * 0.5f;
    const float gridHalfDepth =
        ((boxesZ - 1) * spacing) * 0.5f + boxScale * 0.5f;
    const float horizontalRadius =
        std::sqrt(gridHalfWidth * gridHalfWidth +
                  gridHalfDepth * gridHalfDepth);
    const float boundingRadius =
        std::sqrt(horizontalRadius * horizontalRadius +
                  gridHalfHeight * gridHalfHeight);
    const float fovY = glm::radians(45.0f);
    const float cameraDistance =
        boundingRadius / std::tan(fovY * 0.5f) + boxScale;
    std::mt19937 trialRng(kDefaultCameraSeed + static_cast<unsigned int>(trial));
    std::uniform_real_distribution<float> angleDistribution(
        0.0f, glm::two_pi<float>());
    const float angle = angleDistribution(trialRng);
    const glm::vec3 eye(cameraDistance * std::sin(angle),
                        0.0f,
                        cameraDistance * std::cos(angle));
    glm::mat4 modelview =
        glm::lookAt(eye,
                    glm::vec3(0.0f),
                    glm::vec3(0.0f, 1.0f, 0.0f));

    const glm::mat4 projection =
        glm::perspective(fovY, aspect, 0.1f, 100.0f);

    const double localPaintSeconds = paint(localImage, modelview, projection);

    double maxPaintSeconds = 0.0;
    MPI_Allreduce(&localPaintSeconds,
                  &maxPaintSeconds,
                  1,
                  MPI_DOUBLE,
                  MPI_MAX,
                  MPI_COMM_WORLD);
    accumulatedMaxPaintSeconds += maxPaintSeconds;

    if (rank == 0) {
      std::cout << "Trial " << trial << ": paint time (max across ranks) = "
                << maxPaintSeconds << " s" << std::endl;
    }

    std::unique_ptr<Image> compositedImage =
        compositor->compose(&localImage, groupGuard.group, MPI_COMM_WORLD, yaml);

    if (compositedImage) {
      std::unique_ptr<ImageFull> fullImage;
      if (auto asFull = dynamic_cast<ImageFull*>(compositedImage.get())) {
        fullImage.reset(asFull);
        compositedImage.release();
      } else if (auto asSparse =
                     dynamic_cast<ImageSparse*>(compositedImage.get())) {
        fullImage = asSparse->uncompress();
      } else {
        if (rank == 0) {
          std::cerr << "Unsupported image type returned by compositor."
                    << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
      }

      std::unique_ptr<ImageFull> gatheredImage =
          fullImage->Gather(0, MPI_COMM_WORLD);

      if (rank == 0 && gatheredImage) {
        const int pixels = gatheredImage->getNumberOfPixels();
        std::cout << "Trial " << trial << ": composed " << pixels
                  << " pixels on rank 0" << std::endl;

        std::ostringstream filename;
        filename << "viskores-composite-trial-" << trial << ".ppm";
        if (SavePPM(*gatheredImage, filename.str())) {
          std::cout << "Saved trial " << trial << " composited image to '"
                    << filename.str() << "'" << std::endl;
        } else {
          std::cerr << "Failed to save trial " << trial
                    << " composited image to '" << filename.str() << "'"
                    << std::endl;
        }

        // keep gatheredImage alive for potential future use
      }
    }
  }

  if (rank == 0 && options.trials > 0) {
    const double averageMaxPaintSeconds =
        accumulatedMaxPaintSeconds / static_cast<double>(options.trials);
    std::cout << "Average paint time (max across ranks): "
              << averageMaxPaintSeconds << " s" << std::endl;
  }

  return 0;
}

int main(int argc, char* argv[]) {
  std::vector<std::string> originalArgs(argv, argv + argc);
  std::vector<char*> argvCopy;
  argvCopy.reserve(originalArgs.size());
  for (std::size_t i = 0; i < originalArgs.size(); ++i) {
    argvCopy.push_back(const_cast<char*>(originalArgs[i].c_str()));
  }

  MPI_Init(&argc, &argv);

  int exitCode = 0;
  try {
    ViskoresExample example;
    const int argcCopy = static_cast<int>(argvCopy.size());
    exitCode = example.run(argcCopy, argvCopy.data());
  } catch (const std::exception& error) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::cerr << "Error on rank " << rank << ": " << error.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return exitCode;
}
