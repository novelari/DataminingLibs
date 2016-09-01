#pragma once
#include "../../lib_core/include/core_interface.h"
#include "../../lib_gpu/include/gpu_device.h"

namespace lib_gpu {
class GpuDeviceCuda : public GpuDevice {
 public:
  bool SupportedDevice() override;
  void SynchronizeDevice(int dev_id) override;

  template <typename T>
  static void DeallocateMemory(T **dev_ptr) {
    if (*dev_ptr) {
      cudaError_t error = cudaFree(*dev_ptr);
      if (error != cudaSuccess)
        lib_core::CoreInterface::GetInstance().ThrowException(
            "Cuda deallocation failed.");
      *dev_ptr = nullptr;
    }
  }
  template <typename T>
  static void AllocateManagedMemory(T **dev_ptr, size_t size) {
    cudaError_t error = cudaMallocManaged(dev_ptr, size);
    if (error != cudaSuccess)
      lib_core::CoreInterface::GetInstance().ThrowException(
          "Cuda managed allocation failed.");
  }
  template <typename T>
  static void AllocateMemory(T **dev_ptr, size_t size) {
    cudaError_t error = cudaMalloc(dev_ptr, size);
    if (error != cudaSuccess)
      lib_core::CoreInterface::GetInstance().ThrowException(
          "Cuda allocation failed.");
  }
};
}