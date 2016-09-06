#include "precomp.h"

#include "core_interface.h"
#include "gpu_device_cuda.h"

namespace lib_gpu {
GpuDeviceCuda::GpuDeviceCuda(int dev_id) : GpuDevice(dev_id) {
  int nr_of_gpus = 0;
  if (!CheckCudaError(cuDeviceGetCount(&nr_of_gpus))) return;
  device_count_ = nr_of_gpus;
  if (device_count_ == 0) return;

  cuda_context_ = std::make_shared<CudaDeviceContext>(dev_id);
}

void GpuDeviceCuda::PushContextOnThread() {
  cuCtxPushCurrent(cuda_context_->context_);
}

void GpuDeviceCuda::SynchronizeDevice(int stream) {
  if (stream == -1)
    cuCtxSynchronize();
  else
    cuStreamSynchronize(cuda_context_->streams_[stream]);
}
void GpuDeviceCuda::DeallocateMemory(void *dev_ptr) {
  CUresult error = cuMemFree(reinterpret_cast<CUdeviceptr>(dev_ptr));
  if (error != CUDA_SUCCESS)
    lib_core::CoreInterface::GetInstance().ThrowException(
        "Cuda deallocation failed.");
}
void GpuDeviceCuda::DeallocateHostMemory(void *host_ptr) {
  CUresult error = cuMemFreeHost(host_ptr);
  if (error != CUDA_SUCCESS)
    lib_core::CoreInterface::GetInstance().ThrowException(
        "Cuda deallocation failed.");
}
void GpuDeviceCuda::AllocateManagedMemory(void **dev_ptr, size_t size) {
  CUresult error = cuMemAllocManaged(reinterpret_cast<CUdeviceptr *>(dev_ptr),
                                     size, CU_MEM_ATTACH_GLOBAL);
  if (error != CUDA_SUCCESS)
    lib_core::CoreInterface::GetInstance().ThrowException(
        "Cuda managed allocation failed.");
}
void GpuDeviceCuda::CopyToDevice(void *host_ptr, void *dev_ptr, size_t size,
                                 int stream) {
  cuMemcpyHtoDAsync(reinterpret_cast<CUdeviceptr>(dev_ptr), host_ptr, size,
                    cuda_context_->streams_[stream]);
}
void GpuDeviceCuda::CopyToHost(void *host_ptr, void *dev_ptr, size_t size,
                               int stream) {
  cuMemcpyDtoHAsync(host_ptr, reinterpret_cast<CUdeviceptr>(dev_ptr), size,
                    cuda_context_->streams_[stream]);
}
void GpuDeviceCuda::AllocateMemory(void **dev_ptr, size_t size) {
  CUresult error = cuMemAlloc(reinterpret_cast<CUdeviceptr *>(dev_ptr), size);
  if (error != CUDA_SUCCESS)
    lib_core::CoreInterface::GetInstance().ThrowException(
        "Cuda allocation failed.");
}
void GpuDeviceCuda::AllocateHostMemory(void **dev_ptr, size_t size) {
  CUresult error = cuMemAllocHost(dev_ptr, size);
  if (error != CUDA_SUCCESS)
    lib_core::CoreInterface::GetInstance().ThrowException(
        "Cuda allocation failed.");
}
void GpuDeviceCuda::CheckCudaError(cudaError_t error) {
  auto &core_interface = lib_core::CoreInterface::GetInstance();
  if (error != cudaSuccess)
    core_interface.ThrowException("Cuda error found: " +
                                  std::string(cudaGetErrorString(error)));
}
bool GpuDeviceCuda::CheckCudaError(CUresult error) {
  if (error != CUDA_SUCCESS) return false;
  return true;
}
GpuDeviceCuda::CudaDeviceContext::CudaDeviceContext(int dev_id) {
  CUdevice device;
  cuDeviceGet(&device, dev_id);
  cuCtxCreate(&context_, 0, device);

  for (int i = 0; i < 3; ++i) {
    streams_.emplace_back(CUstream());
    cuStreamCreate(&streams_.back(), CU_STREAM_NON_BLOCKING);
  }

  cuCtxPushCurrent(context_);
}
GpuDeviceCuda::CudaDeviceContext::~CudaDeviceContext() {
  for (int i = 0; i < streams_.size(); ++i) cuStreamDestroy(streams_[i]);
  cuCtxDestroy(context_);
}
}