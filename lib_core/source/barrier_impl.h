#pragma once
#include "barrier.h"

namespace lib_core {
class BarrierImpl : public Barrier {
 public:
  BarrierImpl(int nr_threads);
  ~BarrierImpl();

  void Wait() override;
  void Signal() override;

 private:
  up<condition_var> condition_;
  up<mutex> mutex_;

  int nr_threads_;
  int original_thread_count_;
};
}