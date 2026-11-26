// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// CUDA backend of the splat renderer. Same pipeline and per-splat math as
// CpuSplatRenderer.cpp; see SplatRenderer.hpp.

#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cub/cub.cuh>
#include <stdexcept>
#include <string>

#include "SplatKernels.hpp"
#include "SplatRenderer.hpp"

namespace splatrender {

namespace {

__constant__ FrameParams c_frame;

constexpr int kThreadsPerBlock = 256;
constexpr int kTilePixels = kTileSize * kTileSize;

void Check(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " +
                             cudaGetErrorString(status));
  }
}

#define SR_CUDA_CHECK(call) Check((call), #call)

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  ~DeviceBuffer() {
    if (ptr_ != nullptr) {
      cudaFree(ptr_);
    }
  }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  /// Ensure room for `count` elements; contents are not preserved.
  void Reserve(std::size_t count, bool exact = false) {
    if (count <= capacity_) {
      return;
    }
    if (ptr_ != nullptr) {
      SR_CUDA_CHECK(cudaFree(ptr_));
      ptr_ = nullptr;
      capacity_ = 0;
    }
    const std::size_t newCapacity =
        exact ? count : std::max(count, capacity_ + capacity_ / 2);
    SR_CUDA_CHECK(
        cudaMalloc(reinterpret_cast<void**>(&ptr_),
                   std::max<std::size_t>(newCapacity, 1) * sizeof(T)));
    capacity_ = newCapacity;
  }

  T* get() const { return ptr_; }

 private:
  T* ptr_ = nullptr;
  std::size_t capacity_ = 0;
};

__global__ void PreprocessKernel(
    const SplatRecord* __restrict__ records,
    unsigned long long count,
    ProjectedSplat* __restrict__ projected,
    unsigned long long* __restrict__ tileCounts,
    unsigned long long* __restrict__ visibleCount) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  bool visible = false;
  if (i < count) {
    ProjectedSplat p;
    unsigned long long tiles = 0;
    if (ProjectSplat(records[i], c_frame, p)) {
      visible = true;
      tiles = static_cast<unsigned long long>(p.tileMaxX - p.tileMinX) *
              static_cast<unsigned long long>(p.tileMaxY - p.tileMinY);
      projected[i] = p;
    }
    tileCounts[i] = tiles;
  }
  // Warp-aggregated count of visible splats (all lanes are still active).
  const unsigned int ballot = __ballot_sync(0xffffffffu, visible);
  if ((threadIdx.x & 31u) == 0u && ballot != 0u) {
    atomicAdd(visibleCount, static_cast<unsigned long long>(__popc(ballot)));
  }
}

__global__ void DuplicateKernel(const ProjectedSplat* __restrict__ projected,
                                const unsigned long long* __restrict__ offsets,
                                unsigned long long count,
                                int tilesX,
                                unsigned long long* __restrict__ keys,
                                unsigned int* __restrict__ values) {
  const unsigned long long i =
      static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const unsigned long long end = offsets[i];
  const unsigned long long begin = i > 0 ? offsets[i - 1] : 0ull;
  if (begin == end) {
    return;
  }
  const ProjectedSplat p = projected[i];
  unsigned long long k = begin;
  for (int ty = p.tileMinY; ty < p.tileMaxY; ++ty) {
    for (int tx = p.tileMinX; tx < p.tileMaxX; ++tx) {
      keys[k] =
          TileDepthKey(static_cast<unsigned int>(ty * tilesX + tx), p.depth);
      values[k] = static_cast<unsigned int>(i);
      ++k;
    }
  }
}

__global__ void RangesKernel(const unsigned long long* __restrict__ keys,
                             unsigned int count,
                             uint2* __restrict__ ranges) {
  const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= count) {
    return;
  }
  const unsigned int tile = static_cast<unsigned int>(keys[idx] >> 32);
  if (idx == 0) {
    ranges[tile].x = 0;
  } else {
    const unsigned int previous =
        static_cast<unsigned int>(keys[idx - 1] >> 32);
    if (tile != previous) {
      ranges[previous].y = idx;
      ranges[tile].x = idx;
    }
  }
  if (idx == count - 1) {
    ranges[tile].y = count;
  }
}

__global__ void __launch_bounds__(kTilePixels)
    RenderKernel(const uint2* __restrict__ ranges,
                 const unsigned int* __restrict__ values,
                 const ProjectedSplat* __restrict__ projected,
                 int width,
                 int height,
                 int tilesX,
                 float4* __restrict__ out) {
  const int px = blockIdx.x * kTileSize + threadIdx.x;
  const int py = blockIdx.y * kTileSize + threadIdx.y;
  const int tid = threadIdx.y * kTileSize + threadIdx.x;
  const bool inside = px < width && py < height;
  const uint2 range = ranges[blockIdx.y * tilesX + blockIdx.x];
  const float x = static_cast<float>(px) + 0.5f;
  const float y = static_cast<float>(py) + 0.5f;

  __shared__ float2 sPosition[kTilePixels];
  __shared__ float4 sConic[kTilePixels];  // conic A, B, C and tau0
  __shared__ float3 sColor[kTilePixels];

  float transmittance = 1.0f;
  float r = 0.0f, g = 0.0f, b = 0.0f;
  bool done = !inside;
  const int total = static_cast<int>(range.y - range.x);
  for (int base = 0; base < total; base += kTilePixels) {
    // Stop once every pixel of the tile is opaque (uniform across the block).
    if (__syncthreads_count(done) == kTilePixels) {
      break;
    }
    const int idx = base + tid;
    if (idx < total) {
      const ProjectedSplat& s = projected[values[range.x + idx]];
      sPosition[tid] = make_float2(s.px, s.py);
      sConic[tid] = make_float4(s.conicA, s.conicB, s.conicC, s.tau0);
      sColor[tid] = make_float3(s.r, s.g, s.b);
    }
    __syncthreads();
    const int batch = min(kTilePixels, total - base);
    for (int j = 0; !done && j < batch; ++j) {
      const float4 conic = sConic[j];
      const float2 position = sPosition[j];
      const float alpha = SplatAlpha(
          conic.w, conic.x, conic.y, conic.z, x - position.x, y - position.y);
      if (alpha < kMinAlpha) {
        continue;
      }
      const float weight = alpha * transmittance;
      const float3 color = sColor[j];
      r += weight * color.x;
      g += weight * color.y;
      b += weight * color.z;
      transmittance *= 1.0f - alpha;
      if (transmittance < kMinTransmittance) {
        done = true;
      }
    }
    __syncthreads();
  }
  if (inside) {
    out[static_cast<std::size_t>(py) * width + px] =
        make_float4(r + transmittance * c_frame.background[0],
                    g + transmittance * c_frame.background[1],
                    b + transmittance * c_frame.background[2],
                    1.0f - transmittance);
  }
}

int BitsFor(unsigned long long value) {
  int bits = 0;
  while (bits < 32 && (1ull << bits) <= value) {
    ++bits;
  }
  return std::max(bits, 1);
}

unsigned int BlocksFor(unsigned long long count) {
  return static_cast<unsigned int>((count + kThreadsPerBlock - 1) /
                                   kThreadsPerBlock);
}

}  // namespace

struct CudaSplatRenderer::Impl {
  int device = 0;
  cudaStream_t stream = nullptr;
  cudaEvent_t events[4] = {nullptr, nullptr, nullptr, nullptr};

  DeviceBuffer<SplatRecord> records;
  std::size_t count = 0;
  DeviceBuffer<ProjectedSplat> projected;
  DeviceBuffer<unsigned long long> tileCounts;
  DeviceBuffer<unsigned long long> offsets;
  DeviceBuffer<unsigned long long> visibleCount;
  DeviceBuffer<unsigned long long> keys;
  DeviceBuffer<unsigned long long> sortedKeys;
  DeviceBuffer<unsigned int> values;
  DeviceBuffer<unsigned int> sortedValues;
  DeviceBuffer<uint2> ranges;
  DeviceBuffer<float4> image;
  DeviceBuffer<unsigned char> scratch;

  ~Impl() {
    for (cudaEvent_t& event : events) {
      if (event != nullptr) {
        cudaEventDestroy(event);
      }
    }
    if (stream != nullptr) {
      cudaStreamDestroy(stream);
    }
  }

  float Elapsed(int from, int to) const {
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, events[from], events[to]);
    return ms;
  }

  void Run(const FrameParams& frame, RenderStats& stats);
};

void CudaSplatRenderer::Impl::Run(const FrameParams& frame,
                                  RenderStats& stats) {
  SR_CUDA_CHECK(cudaSetDevice(device));
  const ScreenCamera& cam = frame.camera;
  const int tilesX = TilesX(cam);
  const int tilesY = TilesY(cam);
  const unsigned long long numTiles = static_cast<unsigned long long>(tilesX) *
                                      static_cast<unsigned long long>(tilesY);
  const std::size_t pixels = static_cast<std::size_t>(cam.width) * cam.height;
  image.Reserve(pixels);
  ranges.Reserve(numTiles);
  visibleCount.Reserve(1);
  stats = RenderStats{};
  stats.records = count;

  SR_CUDA_CHECK(cudaMemcpyToSymbolAsync(
      c_frame, &frame, sizeof(FrameParams), 0, cudaMemcpyHostToDevice, stream));
  SR_CUDA_CHECK(
      cudaMemsetAsync(ranges.get(), 0, numTiles * sizeof(uint2), stream));
  SR_CUDA_CHECK(cudaMemsetAsync(
      visibleCount.get(), 0, sizeof(unsigned long long), stream));
  SR_CUDA_CHECK(cudaEventRecord(events[0], stream));

  unsigned long long instances = 0;
  if (count > 0) {
    projected.Reserve(count);
    tileCounts.Reserve(count);
    offsets.Reserve(count);
    PreprocessKernel<<<BlocksFor(count), kThreadsPerBlock, 0, stream>>>(
        records.get(),
        count,
        projected.get(),
        tileCounts.get(),
        visibleCount.get());
    SR_CUDA_CHECK(cudaGetLastError());

    std::size_t scratchBytes = 0;
    SR_CUDA_CHECK(cub::DeviceScan::InclusiveSum(
        nullptr, scratchBytes, tileCounts.get(), offsets.get(), count, stream));
    scratch.Reserve(scratchBytes);
    SR_CUDA_CHECK(cub::DeviceScan::InclusiveSum(scratch.get(),
                                                scratchBytes,
                                                tileCounts.get(),
                                                offsets.get(),
                                                count,
                                                stream));
    SR_CUDA_CHECK(cudaMemcpyAsync(&instances,
                                  offsets.get() + (count - 1),
                                  sizeof(instances),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    unsigned long long visible = 0;
    SR_CUDA_CHECK(cudaMemcpyAsync(&visible,
                                  visibleCount.get(),
                                  sizeof(visible),
                                  cudaMemcpyDeviceToHost,
                                  stream));
    SR_CUDA_CHECK(cudaStreamSynchronize(stream));
    stats.visibleSplats = visible;
  }
  SR_CUDA_CHECK(cudaEventRecord(events[1], stream));
  stats.tileInstances = instances;

  if (instances > 0) {
    if (instances > static_cast<unsigned long long>(INT_MAX)) {
      throw std::runtime_error("Too many splat-tile instances in one frame (" +
                               std::to_string(instances) + ")");
    }
    keys.Reserve(instances);
    values.Reserve(instances);
    sortedKeys.Reserve(instances);
    sortedValues.Reserve(instances);
    DuplicateKernel<<<BlocksFor(count), kThreadsPerBlock, 0, stream>>>(
        projected.get(),
        offsets.get(),
        count,
        tilesX,
        keys.get(),
        values.get());
    SR_CUDA_CHECK(cudaGetLastError());

    const int endBit = 32 + BitsFor(numTiles);
    const int items = static_cast<int>(instances);
    std::size_t scratchBytes = 0;
    SR_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr,
                                                  scratchBytes,
                                                  keys.get(),
                                                  sortedKeys.get(),
                                                  values.get(),
                                                  sortedValues.get(),
                                                  items,
                                                  0,
                                                  endBit,
                                                  stream));
    scratch.Reserve(scratchBytes);
    SR_CUDA_CHECK(cub::DeviceRadixSort::SortPairs(scratch.get(),
                                                  scratchBytes,
                                                  keys.get(),
                                                  sortedKeys.get(),
                                                  values.get(),
                                                  sortedValues.get(),
                                                  items,
                                                  0,
                                                  endBit,
                                                  stream));

    RangesKernel<<<BlocksFor(instances), kThreadsPerBlock, 0, stream>>>(
        sortedKeys.get(), static_cast<unsigned int>(instances), ranges.get());
    SR_CUDA_CHECK(cudaGetLastError());
  }
  SR_CUDA_CHECK(cudaEventRecord(events[2], stream));

  const dim3 grid(static_cast<unsigned int>(tilesX),
                  static_cast<unsigned int>(tilesY));
  const dim3 block(kTileSize, kTileSize);
  RenderKernel<<<grid, block, 0, stream>>>(ranges.get(),
                                           sortedValues.get(),
                                           projected.get(),
                                           cam.width,
                                           cam.height,
                                           tilesX,
                                           image.get());
  SR_CUDA_CHECK(cudaGetLastError());
  SR_CUDA_CHECK(cudaEventRecord(events[3], stream));
  SR_CUDA_CHECK(cudaEventSynchronize(events[3]));

  stats.preprocessMs = Elapsed(0, 1);
  stats.sortMs = Elapsed(1, 2);
  stats.rasterMs = Elapsed(2, 3);
  stats.totalMs = Elapsed(0, 3);
}

bool CudaBackendAvailable() {
  int devices = 0;
  return cudaGetDeviceCount(&devices) == cudaSuccess && devices > 0;
}

CudaSplatRenderer::CudaSplatRenderer(int device)
    : impl_(std::make_unique<Impl>()) {
  impl_->device = device;
  SR_CUDA_CHECK(cudaSetDevice(device));
  SR_CUDA_CHECK(
      cudaStreamCreateWithFlags(&impl_->stream, cudaStreamNonBlocking));
  for (cudaEvent_t& event : impl_->events) {
    SR_CUDA_CHECK(cudaEventCreate(&event));
  }
}

CudaSplatRenderer::~CudaSplatRenderer() = default;

void CudaSplatRenderer::SetRecords(const SplatRecord* records,
                                   std::size_t count) {
  if (count > static_cast<std::size_t>(UINT_MAX)) {
    throw std::runtime_error("At most 2^32 - 1 splat records are supported");
  }
  SR_CUDA_CHECK(cudaSetDevice(impl_->device));
  impl_->records.Reserve(count, true);
  if (count > 0) {
    SR_CUDA_CHECK(cudaMemcpy(impl_->records.get(),
                             records,
                             count * sizeof(SplatRecord),
                             cudaMemcpyHostToDevice));
  }
  impl_->count = count;
}

RenderStats CudaSplatRenderer::Render(const FrameParams& frame, Image& out) {
  RenderStats stats;
  impl_->Run(frame, stats);
  out.Resize(frame.camera.width, frame.camera.height);
  SR_CUDA_CHECK(cudaMemcpy(out.rgba.data(),
                           impl_->image.get(),
                           out.rgba.size() * sizeof(float),
                           cudaMemcpyDeviceToHost));
  return stats;
}

const void* CudaSplatRenderer::RenderToDevice(const FrameParams& frame,
                                              RenderStats* stats) {
  RenderStats local;
  impl_->Run(frame, local);
  if (stats != nullptr) {
    *stats = local;
  }
  return impl_->image.get();
}

std::string CudaSplatRenderer::DeviceName() const {
  cudaDeviceProp properties{};
  SR_CUDA_CHECK(cudaGetDeviceProperties(&properties, impl_->device));
  return properties.name;
}

}  // namespace splatrender
