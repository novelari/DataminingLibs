#include "precomp.h"

#include "gpu_device_opencl.h"

namespace lib_gpu {
bool GpuDeviceOpenCl::SupportedDevice() { return false; }
void GpuDeviceOpenCl::SynchronizeDevice() {}
void GpuDeviceOpenCl::SetDeviceForThread(int dev_id) {}
}