#pragma once
#include <functional>
#include <string>
#include <thread>
#include "../../lib_core/include/barrier.h"

namespace lib_core {
class DLLExport CoreInterface {
 public:
  static CoreInterface& GetInstance();

  void ThrowException(string msg);
  void DebugMessage(string msg);
  sp<Barrier> CreateBarrier(int nr_threads);

  template <typename F>
  sp<std::thread> LaunchThread(F func) {
    sp<std::thread> thread = std::make_shared<std::thread>(func);
    return thread;
  }

  template <typename I, typename F>
  void ParallelFor(I start, I end, F func) {
    // Estimate number of threads in the pool
    const static unsigned nb_threads_hint = std::thread::hardware_concurrency();
    const static unsigned nb_threads =
        (nb_threads_hint == 0u ? 8u : nb_threads_hint);

    // Size of a slice for the range functions
    I n = end - start + 1;
    I slice = (I)std::round(n / static_cast<double>(nb_threads));
    slice = slice > I(1) ? slice : I(1);

    // [Helper] Inner loop
    auto launch_range = [&func](int k1, int k2) {
      for (I k = k1; k < k2; k++) {
        func(k);
      }
    };

    // Create pool and launch jobs
    col_array<std::thread> pool;
    pool.reserve(nb_threads);
    I i1 = start;
    I i2 = start + slice < end ? start + slice : end;
    for (unsigned i = 0; i + 1 < nb_threads && i1 < end; ++i) {
      pool.emplace_back(launch_range, i1, i2);
      i1 = i2;
      i2 = i2 + slice < end ? i2 + slice : end;
    }
    if (i1 < end) {
      pool.emplace_back(launch_range, i1, end);
    }

    // Wait for jobs to finish
    for (std::thread& t : pool) {
      if (t.joinable()) {
        t.join();
      }
    }
  }

 private:
  CoreInterface();
  ~CoreInterface();
};
}