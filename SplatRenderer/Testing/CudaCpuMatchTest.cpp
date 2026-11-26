// amrVolumeRenderer is distributed under the OSI-approved BSD 3-clause License.
// See LICENSE.txt for details.
//
// Renders a random scene with both backends and requires the images to
// agree. Built only with the CUDA backend.

#include <SplatRenderer/SplatRenderer.hpp>
#include <SplatRenderer/SplatScene.hpp>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace splatrender;

int main() {
  if (!CudaBackendAvailable()) {
    std::cout << "No CUDA device; skipping" << std::endl;
    return 0;
  }
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
  std::vector<SplatRecord> records(50000);
  for (SplatRecord& r : records) {
    r = SplatRecord{};
    for (float& p : r.position) {
      p = 2.0f * uniform(rng) - 1.0f;
    }
    // Random anisotropic covariance: A A^T with small entries.
    float a[3][3];
    for (auto& row : a) {
      for (float& v : row) {
        v = 0.03f * (uniform(rng) - 0.5f);
      }
    }
    auto cov = [&](int i, int j) {
      return a[i][0] * a[j][0] + a[i][1] * a[j][1] + a[i][2] * a[j][2] +
             (i == j ? 1e-5f : 0.0f);
    };
    r.covariance[0] = cov(0, 0);
    r.covariance[1] = cov(0, 1);
    r.covariance[2] = cov(0, 2);
    r.covariance[3] = cov(1, 1);
    r.covariance[4] = cov(1, 2);
    r.covariance[5] = cov(2, 2);
    r.volume = 1e-5f;
    r.logDensity = -2.0f * uniform(rng);
    r.logTemperature = 4.0f + 3.0f * uniform(rng);
    r.phase = static_cast<std::uint8_t>(uniform(rng) * 3.0f);
    r.flags = kRecordLeaf;
  }

  TransferSettings settings;
  settings.logDensityMin = -2.0f;
  settings.logDensityMax = 0.0f;
  settings.extinctionAtMax = 300.0f;
  settings.logTemperatureMin = 4.0f;
  settings.logTemperatureMax = 7.0f;
  FrameParams frame{};
  frame.transfer = MakeTransferFunction(settings);
  frame.camera = MakeLookAtCamera({0.4f, 0.3f, 3.0f},
                                  {0.0f, 0.0f, 0.0f},
                                  {0.0f, 1.0f, 0.0f},
                                  50.0f,
                                  640,
                                  480,
                                  1e-3f);
  frame.lod.enabled = 0;

  CpuSplatRenderer cpu;
  cpu.SetRecords(records.data(), records.size());
  CudaSplatRenderer cuda(0);
  cuda.SetRecords(records.data(), records.size());
  Image a, b;
  const RenderStats cpuStats = cpu.Render(frame, a);
  const RenderStats cudaStats = cuda.Render(frame, b);
  if (cpuStats.visibleSplats != cudaStats.visibleSplats ||
      cpuStats.tileInstances != cudaStats.tileInstances) {
    std::cerr << "Splat counts differ: cpu " << cpuStats.visibleSplats << '/'
              << cpuStats.tileInstances << ", cuda " << cudaStats.visibleSplats
              << '/' << cudaStats.tileInstances << std::endl;
    return 1;
  }
  double maxDiff = 0.0;
  for (std::size_t i = 0; i < a.rgba.size(); ++i) {
    maxDiff = std::max(maxDiff,
                       std::fabs(static_cast<double>(a.rgba[i]) - b.rgba[i]));
  }
  std::cout << "visible " << cudaStats.visibleSplats << ", instances "
            << cudaStats.tileInstances << ", max |cpu - cuda| " << maxDiff
            << std::endl;
  if (!(maxDiff < 2e-3)) {
    std::cerr << "CudaCpuMatchTest failed: images differ" << std::endl;
    return 1;
  }
  std::cout << "CudaCpuMatchTest passed" << std::endl;
  return 0;
}
