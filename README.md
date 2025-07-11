# miniGraphics #

The miniGraphics miniapp demonstrates parallel rendering in an MPI
environment using a sort-last parallel rendering approach. The main
intention of miniGraphics is to demonstrate and compare the image
compositing step where multiple images are combined to a single image
(essentially an optimized reduce operation); however, miniGraphics also has
multiple rendering backends that can be used in conjunction with the
compositing algorithms.

The miniGraphics miniapp implements the DirectSend parallel rendering
algorithm, which is a straightforward sort-last parallel rendering approach
where each process sends its rendered image directly to the compositing
process.

## Compiling ##

The following are the minimum requirements for building miniGraphics:

  * CMake version 3.3 or better
  * A C++ compiler (C++11 compliant)
  * MPI

To compile miniGraphics, simply run CMake for the base directory of
miniGraphics (the directory containing this file). The basic steps for
compiling with CMake are (1) create a build directory, (2) run cmake in that
directory, and (3) run the build program for the project files generated
(usually make or ninja). The following are typical commands although they can
vary between systems.

    mkdir miniGraphics-build
    cd miniGraphics-build
    cmake ../miniGraphics
    make -j

It is also possible to independently compile the DirectSend variants by
following the same steps but for the subdirectory containing the specific
variant in question.

## Directories ##

The miniGraphics implementation is organized as follows:

  * **DirectSend** Contains the DirectSend compositing algorithm
    implementation. DirectSend is a straightforward algorithm where each
    process sends its rendered image directly to the designated compositing
    process. Within this directory are subdirectories with different variants:
    - **Base** The basic DirectSend implementation
    - **Overlap** An optimized version that overlaps communication

In addition to the DirectSend compositing algorithm, there are also some
supporting directories containing code that is used by the miniGraphics
application.

  * **Paint** Contains the drawing algorithms used in the parallel
    rendering. sort-last parallel rendering happens in 2 phases: a local
    geometry rendering (what we call here a "paint" to prevent name
    overloading) and a parallel image compositing. The painting algorithms
    are located in this directory (and contributed painting algorithms
    should also go here).
  * **Common** A collection of common objects used by miniGraphics.
    Examples include image objects, mesh objects, and boilerplate main loop
    code.
  * **CMake** Contains auxiliary CMake scripts used for building.
  * **ThirdParty** Contains code imported from third party sources that are
    used by miniGraphics.

## License ##

miniGraphics is distributed under the OSI-approved BSD 3-clause License.
See [LICENSE.txt]() for details.

Copyright (c) 2017
National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under
the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
certain rights in this software.

SDR# 2261
