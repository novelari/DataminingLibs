#include "precomp.h"

#include <cuda_runtime_api.h>
#include "core_interface.h"
#include "gpu_device_cuda.h"

namespace lib_gpu {
bool GpuDeviceCuda::SupportedDevice() {
  auto &core_interface = lib_core::CoreInterface::GetInstance();
  cudaError_t err = cudaGetDeviceCount(&device_count_);
  if (err != cudaSuccess) {
    core_interface.DebugMessage(cudaGetErrorString(err));
    return false;
  }

  for (int i = 0; i < device_count_; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    std::stringstream device_msg;
    device_msg << "Device Number: %d\n" << i << "\n";
    device_msg << "  Device name: %s\n" << prop.name << "\n";
    device_msg << "  Memory Clock Rate (KHz): %d\n"
               << prop.memoryClockRate << "\n";
    device_msg << "  Memory Bus Width (bits): %d\n"
               << prop.memoryBusWidth << "\n";
    device_msg << "  Peak Memory Bandwidth (GB/s): %f\n\n"
               << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6
               << "\n";

    core_interface.DebugMessage(device_msg.str().c_str());
  }

  return true;
}
void GpuDeviceCuda::SynchronizeDevice() {
  CheckCudaError(cudaDeviceSynchronize());
}
void GpuDeviceCuda::SetDeviceForThread(int dev_id) {
  cudaError_t error = cudaSetDevice(dev_id);
}
void GpuDeviceCuda::CheckCudaError(cudaError_t error) {
  auto &core_interface = lib_core::CoreInterface::GetInstance();
  if (error != cudaSuccess)
    core_interface.ThrowException("Cuda error found: " +
                                  std::string(cudaGetErrorString(error)));
}
}