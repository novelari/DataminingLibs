#include "precomp.h"

#include "barrier_impl.h"

namespace lib_core {
BarrierImpl::BarrierImpl(int nr_threads)
    : nr_threads_(nr_threads), original_thread_count_(nr_threads) {
  condition_ = std::make_unique<condition_var>();
  mutex_ = std::make_unique<mutex>();
}

BarrierImpl::~BarrierImpl() {}

void BarrierImpl::Wait() {
  mutex_lock lk(*mutex_);
  --nr_threads_;
  if (nr_threads_ <= 0) {
    nr_threads_ = original_thread_count_;
    lk.unlock();
    condition_->notify_all();
  } else {
    condition_->wait(lk);
  }
}

void BarrierImpl::Signal() {
  mutex_lock lk(*mutex_);
  --nr_threads_;
  if (nr_threads_ <= 0) {
    nr_threads_ = original_thread_count_;
    lk.unlock();
    condition_->notify_all();
  }
}
}
