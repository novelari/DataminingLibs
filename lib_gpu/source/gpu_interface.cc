#include "precomp.h"

#include "gpu_device_cuda.h"
#include "gpu_device_opencl.h"
#include "gpu_interface.h"

namespace lib_gpu {
GpuInterface& GpuInterface::GetInstance() {
  static GpuInterface instance;
  return instance;
}

sp<GpuDevice> GpuInterface::CreateGpuDevice(int dev_id, PreferredApi pref_api) {
  sp<GpuDevice> res;
  switch (pref_api) {
    case kCuda:
      if (!cuda_support_) {
        res = std::make_shared<GpuDeviceOpenCl>(dev_id);
        break;
      }
      if (init_cuda_) {
        init_cuda_ = false;
        CUresult error = cuInit(0);
        if (error != CUDA_SUCCESS) {
          res = std::make_shared<GpuDeviceOpenCl>(dev_id);
          cuda_support_ = false;
          break;
        }
      }
      res = std::make_shared<GpuDeviceCuda>(dev_id);
      break;
    case kOpenCl:
      res = std::make_shared<GpuDeviceOpenCl>(dev_id);
      break;
    default:
      res = std::make_shared<GpuDeviceOpenCl>(dev_id);
      break;
  }
  return res;
}

GpuInterface::GpuInterface() : init_cuda_(true), cuda_support_(true) {}

GpuInterface::~GpuInterface() {}
}