#include "precomp.h"

#include "gpu_device_cuda.h"
#include "gpu_device_opencl.h"
#include "gpu_interface.h"

namespace lib_gpu {
GpuInterface& GpuInterface::GetInstance() {
  static GpuInterface instance;
  return instance;
}

GpuDevice& GpuInterface::GetGpuDevice(int dev_id, PreferredApi pref_api) {
  switch (pref_api) {
    case kCuda:
      if (cuda_device_->SupportedDevice()) return *cuda_device_;
    case kOpenCl:
      if (opencl_device_->SupportedDevice()) return *opencl_device_;
    default:
      return *opencl_device_;
  }
}

GpuInterface::GpuInterface()
    : cuda_device_(new GpuDeviceCuda()),
      opencl_device_(new GpuDeviceOpenCl()) {}

GpuInterface::~GpuInterface() {
  delete cuda_device_;
  delete opencl_device_;
}
}