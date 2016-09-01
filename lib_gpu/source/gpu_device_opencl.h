#pragma once
#include "gpu_device.h"

namespace lib_gpu {
class GpuDeviceOpenCl : public GpuDevice {
 public:
  bool SupportedDevice() override;
  void SynchronizeDevice(int dev_id) override;
};
}