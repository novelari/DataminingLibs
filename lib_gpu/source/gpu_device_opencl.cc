#include "precomp.h"

#include "gpu_device_opencl.h"

namespace lib_gpu {
void GpuDeviceOpenCl::PushContextOnThread() {}
void GpuDeviceOpenCl::SynchronizeDevice(int stream) {}
void GpuDeviceOpenCl::DeallocateMemory(void* dev_ptr) {}
void GpuDeviceOpenCl::DeallocateHostMemory(void* host_ptr) {}
void GpuDeviceOpenCl::AllocateManagedMemory(void** dev_ptr, size_t size) {}
void GpuDeviceOpenCl::CopyToDevice(void* host_ptr, void* dev_ptr, size_t size,
                                   int stream) {}
void GpuDeviceOpenCl::CopyToHost(void* host_ptr, void* dev_ptr, size_t size,
                                 int stream) {}
void GpuDeviceOpenCl::AllocateMemory(void** dev_ptr, size_t size) {}
void GpuDeviceOpenCl::AllocateHostMemory(void** dev_ptr, size_t size) {}
}