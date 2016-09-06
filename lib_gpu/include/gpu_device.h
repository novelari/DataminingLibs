#pragma once

namespace lib_gpu {
class DLLExport GpuDevice {
 public:
  GpuDevice(int dev_id) : dev_id_(dev_id) {}
  virtual ~GpuDevice() {}

  virtual void PushContextOnThread() = 0;
  virtual void SynchronizeDevice(int stream = -1) = 0;
  virtual void DeallocateMemory(void *dev_ptr) = 0;
  virtual void DeallocateHostMemory(void *host_ptr) = 0;
  virtual void AllocateManagedMemory(void **dev_ptr, size_t size) = 0;
  virtual void CopyToDevice(void *host_ptr, void *dev_ptr, size_t size,
                            int stream = 0) = 0;
  virtual void CopyToHost(void *host_ptr, void *dev_ptr, size_t size,
                          int stream = 0) = 0;
  virtual void AllocateMemory(void **dev_ptr, size_t size) = 0;
  virtual void AllocateHostMemory(void **dev_ptr, size_t size) = 0;

  int GetDeviceCount();

 protected:
  int device_count_;
  int dev_id_;
};
}