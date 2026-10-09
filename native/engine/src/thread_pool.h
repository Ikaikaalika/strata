// Persistent fork-join pool for row-parallel CPU kernels.
#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace lokahi {

class ThreadPool {
 public:
  explicit ThreadPool(unsigned threads = 0);
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  unsigned size() const { return static_cast<unsigned>(workers_.size()) + 1; }

  // Calls fn(begin, end) over disjoint chunks of [0, count). The calling
  // thread participates; returns after every chunk finishes.
  void parallel_for(size_t count, const std::function<void(size_t, size_t)>& fn,
                    size_t min_chunk = 1);

 private:
  void worker_loop(unsigned index);

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable start_cv_;
  std::condition_variable done_cv_;
  const std::function<void(size_t, size_t)>* job_ = nullptr;
  size_t job_count_ = 0;
  size_t chunk_ = 0;
  unsigned pending_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

}  // namespace lokahi
