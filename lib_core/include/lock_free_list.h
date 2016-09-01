#pragma once

namespace lib_core {
template <typename T>
class LockFreeList {
  LockFreeList() = default;

 public:
  virtual ~LockFreeList() = default;

  class reference {
   public:
    virtual T& operator*() = 0;
    virtual T* operator->() = 0;
    virtual bool empty() = 0;
  };

  virtual reference find(T t) const = 0;
  virtual void push_front(T t) = 0;
  virtual reference pop_front() = 0;
};
}