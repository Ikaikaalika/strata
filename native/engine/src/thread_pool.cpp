#include "thread_pool.h"

#include <algorithm>

namespace lokahi {

ThreadPool::ThreadPool(unsigned threads) {
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());
  for (unsigned i = 1; i < threads; ++i) {
    workers_.emplace_back([this, i] { worker_loop(i); });
  }
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_ = true;
  }
  start_cv_.notify_all();
  for (auto& worker : workers_) worker.join();
}

void ThreadPool::worker_loop(unsigned index) {
  uint64_t seen = 0;
  while (true) {
    const std::function<void(size_t, size_t)>* job;
    size_t count, chunk;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      start_cv_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) return;
      seen = generation_;
      job = job_;
      count = job_count_;
      chunk = chunk_;
    }
    size_t begin = index * chunk;
    if (begin < count) (*job)(begin, std::min(count, begin + chunk));
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (--pending_ == 0) done_cv_.notify_one();
    }
  }
}

void ThreadPool::parallel_for(size_t count, const std::function<void(size_t, size_t)>& fn,
                              size_t min_chunk) {
  if (count == 0) return;
  size_t parts = std::min<size_t>(size(), (count + min_chunk - 1) / min_chunk);
  if (parts <= 1 || workers_.empty()) {
    fn(0, count);
    return;
  }
  size_t chunk = (count + parts - 1) / parts;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    job_ = &fn;
    job_count_ = count;
    chunk_ = chunk;
    pending_ = static_cast<unsigned>(workers_.size());
    ++generation_;
  }
  start_cv_.notify_all();
  fn(0, std::min(count, chunk));
  std::unique_lock<std::mutex> lock(mutex_);
  done_cv_.wait(lock, [&] { return pending_ == 0; });
  job_ = nullptr;
}

}  // namespace lokahi
