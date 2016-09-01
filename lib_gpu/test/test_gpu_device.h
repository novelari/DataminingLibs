#pragma once
#include "lib_gpu.h"

namespace lib_gpu {
auto &gpu_interface = GpuInterface::GetInstance();
TEST(lib_gpu, gpu_device) {
  auto &device = gpu_interface.GetGpuDevice();
  ASSERT_TRUE(device.SupportedDevice());
}
}