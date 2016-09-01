#pragma once

namespace sutil {
template <typename T>
class LockFreeList {
  struct Node {
    T t;
    sp<Node> next;
  };
  sp<Node> head_;

  LockFreeList(LockFreeList&) = delete;
  void operator=(LockFreeList&) = delete;

 public:
  LockFreeList() = default;
  ~LockFreeList() = default;

  class reference {
    sp<Node> p_;

   public:
    reference(sp<Node> p) : p_{p} {}
    T& operator*() { return p_->t; }
    T* operator->() { return &p_->t; }
	bool empty() { return !p_; }
  };

  reference find(T t) const {
    auto p = std::atomic_load(&head_);
    while (p && p->t != t) p = p->next;
    return reference(std::move(p));
  }

  void push_front(T t) {
    auto p = std::make_shared<Node>();
    p->t = t;
    p->next = std::atomic_load(&head_);
    while (!std::atomic_compare_exchange_weak(&head_,&p->next, p)) {
    }
  }

  reference pop_front() {
    auto p = std::atomic_load(&head_);
    while (p && !std::atomic_compare_exchange_weak(&head_,&p, p->next)) {
    }
	return reference(std::move(p));
  }
};
}