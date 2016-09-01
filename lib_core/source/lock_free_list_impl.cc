#include "precomp.h"

#include "lock_free_list_impl.h"

namespace lib_core {
template <typename T>
LockFreeList<T>::reference LockFreeListImpl<T>::find(T t) const {
  auto p = std::atomic_load(&head_);
  while (p && p->t != t) p = p->next;
  return reference_impl(std::move(p));
}

template <typename T>
void LockFreeListImpl<T>::push_front(T t) {
  auto p = std::make_shared<Node>();
  p->t = t;
  p->next = std::atomic_load(&head_);
  while (!std::atomic_compare_exchange_weak(&head_, &p->next, p)) {
  }
}

template <typename T>
LockFreeList<T>::reference LockFreeListImpl<T>::pop_front() {
  auto p = std::atomic_load(&head_);
  while (p && !std::atomic_compare_exchange_weak(&head_, &p, p->next)) {
  }
  return reference_impl(std::move(p));
}
}