// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//

#include <AMReX.H>
#include <AMReX_Box.H>
#include <AMReX_Geometry.H>
#include <AMReX_Gpu.H>
#include <AMReX_IntVect.H>
#include <AMReX_MultiFab.H>
#include <AMReX_RealBox.H>

#include <VolumeRenderer/VolumeRendererApi.hpp>

#include <vector>

int main(int argc, char** argv) {
  amrex::Initialize(argc, argv);
  {
    const int nCells = 32;
    const int nComp = 1;
    const int nGhost = 0;

    amrex::Box domain(amrex::IntVect(0),
                      amrex::IntVect(nCells - 1));
    amrex::RealBox realBox({0.0, 0.0, 0.0}, {1.0, 1.0, 1.0});
    amrex::Array<int, AMREX_SPACEDIM> isPer = {0, 0, 0};
    amrex::Geometry geom(domain, realBox, amrex::CoordSys::cartesian, isPer);

    amrex::BoxArray boxArray(domain);
    boxArray.maxSize(16);
    amrex::DistributionMapping distMap(boxArray);

    amrex::MultiFab multifab(boxArray, distMap, nComp, nGhost);
    multifab.setVal(0.0);

    for (amrex::MFIter mfi(multifab); mfi.isValid(); ++mfi) {
      const amrex::Box& box = mfi.validbox();
      auto array = multifab.array(mfi);
      amrex::ParallelFor(
          box, [=] AMREX_GPU_DEVICE(int i, int j, int k) noexcept {
            const amrex::Real x = static_cast<amrex::Real>(i) / (nCells - 1);
            const amrex::Real y = static_cast<amrex::Real>(j) / (nCells - 1);
            const amrex::Real z = static_cast<amrex::Real>(k) / (nCells - 1);
            array(i, j, k, 0) = x * x + y * y + z * z;
          });
    }

    amrVolumeRenderer::api::AmrData data;
    data.levelData.push_back(&multifab);
    data.levelGeometry.push_back(geom);

    amrVolumeRenderer::api::RenderOptions options;
    options.width = 512;
    options.height = 512;
    options.outputFilename = "multifab-render.png";

    amrVolumeRenderer::api::Render(data, options);
  }
  amrex::Finalize();
  return 0;
}
