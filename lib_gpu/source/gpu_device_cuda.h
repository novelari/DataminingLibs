#pragma once
#ifndef CUDA_API_PER_THREAD_DEFAULT_STREAM
#define CUDA_API_PER_THREAD_DEFAULT_STREAM
#endif
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "../../lib_core/include/core_interface.h"
#include "../../lib_gpu/include/gpu_device.h"

namespace lib_gpu {
class GpuDeviceCuda : public GpuDevice {
 public:
  GpuDeviceCuda(int dev_id);

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

 private:
  class CudaDeviceContext {
   public:
    CudaDeviceContext(int dev_id);
    ~CudaDeviceContext();

    CUcontext context_;
    col_array<CUstream> streams_;
  };

  void CheckCudaError(cudaError_t error);
  bool CheckCudaError(CUresult error);

  sp<CudaDeviceContext> cuda_context_;
};
}
