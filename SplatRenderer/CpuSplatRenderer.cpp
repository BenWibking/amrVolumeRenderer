// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <utility>
#include <vector>

#include "SplatKernels.hpp"
#include "SplatRenderer.hpp"

namespace splatrender {

namespace {

using Clock = std::chrono::steady_clock;

double MillisecondsSince(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}

int WorkerCount() { return std::max(1u, std::thread::hardware_concurrency()); }

template <typename Body>
void ParallelFor(std::size_t count, Body&& body) {
  const int workers = WorkerCount();
  std::atomic<std::size_t> next{0};
  auto run = [&] {
    for (std::size_t i = next++; i < count; i = next++) {
      body(i);
    }
  };
  std::vector<std::thread> threads;
  for (int t = 1; t < workers; ++t) {
    threads.emplace_back(run);
  }
  run();
  for (auto& thread : threads) {
    thread.join();
  }
}

}  // namespace

void CpuSplatRenderer::SetRecords(const SplatRecord* records,
                                  std::size_t count) {
  records_ = records;
  count_ = count;
}

RenderStats CpuSplatRenderer::Render(const FrameParams& frame, Image& out) {
  RenderStats stats;
  stats.records = count_;
  const auto start = Clock::now();
  const ScreenCamera& cam = frame.camera;
  const int tilesX = TilesX(cam);
  const int tilesY = TilesY(cam);

  // 1. Project in contiguous chunks so the merged list keeps record order,
  //    which makes depth ties resolve exactly as in the CUDA backend.
  const std::size_t chunkSize = 1 << 16;
  const std::size_t chunks = (count_ + chunkSize - 1) / chunkSize;
  std::vector<std::vector<ProjectedSplat>> perChunk(chunks);
  ParallelFor(chunks, [&](std::size_t c) {
    const std::size_t begin = c * chunkSize;
    const std::size_t end = std::min(count_, begin + chunkSize);
    ProjectedSplat projected;
    for (std::size_t i = begin; i < end; ++i) {
      if (ProjectSplat(records_[i], frame, projected)) {
        perChunk[c].push_back(projected);
      }
    }
  });
  std::vector<ProjectedSplat> splats;
  for (auto& chunk : perChunk) {
    splats.insert(splats.end(), chunk.begin(), chunk.end());
    std::vector<ProjectedSplat>().swap(chunk);
  }
  stats.visibleSplats = splats.size();
  stats.preprocessMs = MillisecondsSince(start);

  // 2. One key per overlapped tile, sorted by (tile, depth, splat index).
  const auto sortStart = Clock::now();
  std::vector<std::pair<unsigned long long, std::uint32_t>> keys;
  for (std::size_t i = 0; i < splats.size(); ++i) {
    const ProjectedSplat& s = splats[i];
    for (int ty = s.tileMinY; ty < s.tileMaxY; ++ty) {
      for (int tx = s.tileMinX; tx < s.tileMaxX; ++tx) {
        keys.emplace_back(
            TileDepthKey(static_cast<unsigned int>(ty * tilesX + tx), s.depth),
            static_cast<std::uint32_t>(i));
      }
    }
  }
  std::sort(keys.begin(), keys.end());
  stats.tileInstances = keys.size();
  std::vector<std::pair<std::size_t, std::size_t>> ranges(
      static_cast<std::size_t>(tilesX) * tilesY, {0, 0});
  for (std::size_t k = 0; k < keys.size();) {
    const std::size_t tile = static_cast<std::size_t>(keys[k].first >> 32);
    std::size_t end = k;
    while (end < keys.size() && (keys[end].first >> 32) == tile) {
      ++end;
    }
    ranges[tile] = {k, end};
    k = end;
  }
  stats.sortMs = MillisecondsSince(sortStart);

  // 3. Blend each tile front to back.
  const auto rasterStart = Clock::now();
  out.Resize(cam.width, cam.height);
  ParallelFor(ranges.size(), [&](std::size_t tile) {
    const int tx = static_cast<int>(tile % tilesX);
    const int ty = static_cast<int>(tile / tilesX);
    const auto range = ranges[tile];
    for (int py = ty * kTileSize;
         py < std::min(cam.height, (ty + 1) * kTileSize);
         ++py) {
      for (int px = tx * kTileSize;
           px < std::min(cam.width, (tx + 1) * kTileSize);
           ++px) {
        const float x = static_cast<float>(px) + 0.5f;
        const float y = static_cast<float>(py) + 0.5f;
        float transmittance = 1.0f;
        float r = 0.0f, g = 0.0f, b = 0.0f;
        for (std::size_t k = range.first; k < range.second; ++k) {
          const ProjectedSplat& s = splats[keys[k].second];
          const float alpha = SplatAlpha(
              s.tau0, s.conicA, s.conicB, s.conicC, x - s.px, y - s.py);
          if (alpha < kMinAlpha) {
            continue;
          }
          const float weight = alpha * transmittance;
          r += weight * s.r;
          g += weight * s.g;
          b += weight * s.b;
          transmittance *= 1.0f - alpha;
          if (transmittance < kMinTransmittance) {
            break;
          }
        }
        float* pixel = out.Pixel(px, py);
        pixel[0] = r + transmittance * frame.background[0];
        pixel[1] = g + transmittance * frame.background[1];
        pixel[2] = b + transmittance * frame.background[2];
        pixel[3] = 1.0f - transmittance;
      }
    }
  });
  stats.rasterMs = MillisecondsSince(rasterStart);
  stats.totalMs = MillisecondsSince(start);
  return stats;
}

}  // namespace splatrender
