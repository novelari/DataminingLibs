#include "precomp.h"

#include "barrier_impl.h"
#include "core_interface.h"

namespace lib_core {
static col_array<string> exception_stack_;
static col_array<string> debug_stack_;

CoreInterface& CoreInterface::GetInstance() {
  static CoreInterface instance;
  return instance;
}
void CoreInterface::ThrowException(string msg) {
  exception_stack_.emplace_back(msg);
  throw std::exception(msg.c_str());
}
void CoreInterface::DebugMessage(string msg) { debug_stack_.emplace_back(msg); }
sp<Barrier> CoreInterface::CreateBarrier(int nr_threads) {
  return std::make_shared<BarrierImpl>(nr_threads);
}
CoreInterface::CoreInterface() {}
CoreInterface::~CoreInterface() {}
}