#include <AMReX.H>
#include <mpi.h>

#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include "ViskoresVolumeRenderer.hpp"

int main(int argc, char* argv[]) {
  std::vector<std::string> originalArgs(argv, argv + argc);
  std::vector<char*> argvCopy;
  argvCopy.reserve(originalArgs.size());
  for (std::size_t i = 0; i < originalArgs.size(); ++i) {
    argvCopy.push_back(const_cast<char*>(originalArgs[i].c_str()));
  }

  MPI_Init(&argc, &argv);
  amrex::Initialize(argc, argv, false, MPI_COMM_WORLD);

  int exitCode = 0;
  try {
    ViskoresVolumeRenderer example;
    const int argcCopy = static_cast<int>(argvCopy.size());
    exitCode = example.run(argcCopy, argvCopy.data());
  } catch (const std::exception& error) {
    int commRank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &commRank);
    std::cerr << "Error on rank " << commRank << ": " << error.what()
              << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  amrex::Finalize();
  MPI_Finalize();
  return exitCode;
}
