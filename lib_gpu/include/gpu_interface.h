#pragma once
#include "gpu_device.h"

namespace lib_gpu {
class DLLExport GpuInterface {
 public:
  static GpuInterface& GetInstance();

  enum PreferredApi { kCuda, kOpenCl };
  sp<GpuDevice> CreateGpuDevice(int dev_id = 0, PreferredApi pref_api = kCuda);

 private:
  GpuInterface();
  ~GpuInterface();
  bool init_cuda_;
  bool cuda_support_;
};
}