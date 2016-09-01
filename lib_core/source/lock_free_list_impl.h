#pragma once
#include "lock_free_list.h"

namespace lib_core {
template <typename T>
class LockFreeListImpl : public LockFreeList<T> {
  struct Node {
    T t;
    sp<Node> next;
  };
  sp<Node> head_;
  sp<Node> tail_;

  LockFreeListImpl(LockFreeListImpl&) = delete;
  void operator=(LockFreeListImpl&) = delete;

 public:
  LockFreeListImpl() = default;
  virtual ~LockFreeListImpl() = default;

  class reference_impl : public reference {
    sp<Node> p_;

   public:
    reference_impl(sp<Node> p) : p_{p} {}
    T& operator*() { return p_->t; }
    T* operator->() { return &p_->t; }
    bool empty() { return !p_; }
  };

  reference find(T t) const override;
  void push_front(T t) override;
  reference pop_front() override;
};
}