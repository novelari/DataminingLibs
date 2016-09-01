#pragma once
#include "gpu_device.h"

namespace lib_gpu {
class DLLExport GpuInterface {
 public:
  static GpuInterface& GetInstance();

  enum PreferredApi { kCuda, kOpenCl };
  GpuDevice& GetGpuDevice(int dev_id = 0, PreferredApi pref_api = kCuda);

 private:
  GpuDevice* cuda_device_;
  GpuDevice* opencl_device_;

  GpuInterface();
  ~GpuInterface();
};
}