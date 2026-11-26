// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Tile-based Gaussian splat renderers. Both backends run the same pipeline:
//   1. select (LOD, phase), classify (transfer function) and project every
//      record, counting the 16x16 screen tiles each splat overlaps;
//   2. emit one (tile, depth) key per overlapped tile and sort the keys;
//   3. for every tile, blend its splats front to back until opaque.
// The CPU backend is a reference implementation; the CUDA backend is the one
// meant for interactive use.

#ifndef SPLAT_RENDERER_SPLAT_RENDERER_HPP
#define SPLAT_RENDERER_SPLAT_RENDERER_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "SplatScene.hpp"
#include "SplatTypes.hpp"

namespace splatrender {

struct RenderStats {
  std::uint64_t records = 0;
  std::uint64_t visibleSplats = 0;
  std::uint64_t tileInstances = 0;
  double preprocessMs = 0.0;
  double sortMs = 0.0;
  double rasterMs = 0.0;
  double totalMs = 0.0;
};

class CpuSplatRenderer {
 public:
  /// The records must stay alive while the renderer uses them.
  void SetRecords(const SplatRecord* records, std::size_t count);
  RenderStats Render(const FrameParams& frame, Image& out);

 private:
  const SplatRecord* records_ = nullptr;
  std::size_t count_ = 0;
};

/// True if the library was built with the CUDA backend and a device exists.
bool CudaBackendAvailable();

class CudaSplatRenderer {
 public:
  explicit CudaSplatRenderer(int device = 0);
  ~CudaSplatRenderer();
  CudaSplatRenderer(const CudaSplatRenderer&) = delete;
  CudaSplatRenderer& operator=(const CudaSplatRenderer&) = delete;

  /// Copy the records to device memory.
  void SetRecords(const SplatRecord* records, std::size_t count);

  /// Render and copy the image to the host.
  RenderStats Render(const FrameParams& frame, Image& out);

  /// Render into device memory and return a pointer to width*height float4
  /// RGBA pixels (valid until the next call). Intended for graphics interop.
  const void* RenderToDevice(const FrameParams& frame, RenderStats* stats);

  std::string DeviceName() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace splatrender

#endif  // SPLAT_RENDERER_SPLAT_RENDERER_HPP
