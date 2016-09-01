#pragma once

namespace lib_core {
class DLLExport Barrier {
 public:
  virtual ~Barrier() {}

  virtual void Wait() = 0;
  virtual void Signal() = 0;
};
}