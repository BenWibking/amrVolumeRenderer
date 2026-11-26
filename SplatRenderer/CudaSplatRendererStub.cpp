// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Used when the library is built without CUDA.

#include <stdexcept>

#include "SplatRenderer.hpp"

namespace splatrender {

struct CudaSplatRenderer::Impl {};

bool CudaBackendAvailable() { return false; }

CudaSplatRenderer::CudaSplatRenderer(int /*device*/) {
  throw std::runtime_error("splat renderer was built without CUDA support");
}

CudaSplatRenderer::~CudaSplatRenderer() = default;

void CudaSplatRenderer::SetRecords(const SplatRecord*, std::size_t) {}

RenderStats CudaSplatRenderer::Render(const FrameParams&, Image&) { return {}; }

const void* CudaSplatRenderer::RenderToDevice(const FrameParams&,
                                              RenderStats*) {
  return nullptr;
}

std::string CudaSplatRenderer::DeviceName() const { return {}; }

}  // namespace splatrender
