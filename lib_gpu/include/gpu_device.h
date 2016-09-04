#pragma once

namespace lib_gpu {
class DLLExport GpuDevice {
 public:
  virtual ~GpuDevice() {}

  virtual bool SupportedDevice() = 0;
  virtual void SynchronizeDevice() = 0;
  virtual void SetDeviceForThread(int dev_id) = 0;

  int GetDeviceCount();

 protected:
  int device_count_;
};
}