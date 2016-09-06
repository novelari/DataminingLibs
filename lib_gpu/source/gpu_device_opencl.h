#pragma once
#include "gpu_device.h"

namespace lib_gpu {
class GpuDeviceOpenCl : public GpuDevice {
 public:
  GpuDeviceOpenCl(int dev_id) : GpuDevice(dev_id) {}

  void PushContextOnThread() override;
  void SynchronizeDevice(int stream = -1) override;
  void DeallocateMemory(void *dev_ptr) override;
  void DeallocateHostMemory(void *host_ptr) override;
  void AllocateManagedMemory(void **dev_ptr, size_t size) override;
  void CopyToDevice(void *host_ptr, void *dev_ptr, size_t size,
                    int stream = 0) override;
  void CopyToHost(void *host_ptr, void *dev_ptr, size_t size,
                  int stream = 0) override;
  void AllocateMemory(void **dev_ptr, size_t size) override;
  void AllocateHostMemory(void **dev_ptr, size_t size) override;
};
}