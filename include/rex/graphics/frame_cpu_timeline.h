// Opt-in exclusive CPU timeline for the graphics command processor thread.
#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>

namespace rex::graphics {

class FrameCpuTimeline {
 public:
  using Clock = std::chrono::steady_clock;
  using Time = Clock::time_point;
  enum class Bucket : size_t {
    kOther, kRingIdle, kWaitMem, kWaitReg, kPm4, kDraw, kCopy, kSwap,
    kExecute, kSubmit, kFence, kInterrupt, kSwapPacket, kCoherent, kPendingFns,
    kEndCommandBuffer, kResetFence, kQueueAcquire, kCount
  };
  static constexpr size_t kCount = static_cast<size_t>(Bucket::kCount);

  void Start(Time now) {
    durations_.fill(std::chrono::nanoseconds::zero());
    wait_polls_ = {}; wait_counts_ = {}; overflow_ = 0;
    queue_acquire_count_ = 0;
    last_ = now;
    running_ = true;
  }
  void Stop(Time now) {
    if (running_) { Charge(now); running_ = false; }
  }
  void Switch(Bucket next) {
    if (running_) Charge(Clock::now());
    current_ = next;
  }
  void ObserveWait(bool memory, uint32_t address, uint64_t polls) {
    if (!running_) return;
    ++wait_counts_[memory ? 0 : 1];
    uint64_t key = (uint64_t(memory) << 32) | address;
    for (auto& entry : wait_polls_) {
      if (entry.polls == 0 || entry.key == key) {
        entry.key = key; entry.polls += polls; return;
      }
    }
    ++overflow_;
  }
  struct PollEntry { uint64_t key = 0, polls = 0; };
  const auto& wait_polls() const { return wait_polls_; }
  const auto& wait_counts() const { return wait_counts_; }
  uint64_t wait_overflow() const { return overflow_; }
  Bucket current() const { return current_; }
  const auto& durations() const { return durations_; }
  void NoteQueueAcquire() { if (running_) ++queue_acquire_count_; }
  uint64_t queue_acquire_count() const { return queue_acquire_count_; }

  class Scope {
   public:
    Scope(FrameCpuTimeline& timeline, Bucket bucket)
        : timeline_(timeline), previous_(timeline.current()) { timeline_.Switch(bucket); }
    ~Scope() { timeline_.Switch(previous_); }
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
   private:
    FrameCpuTimeline& timeline_;
    Bucket previous_;
  };
 private:
  void Charge(Time now) {
    durations_[static_cast<size_t>(current_)] +=
        std::chrono::duration_cast<std::chrono::nanoseconds>(now - last_);
    last_ = now;
  }
  std::array<std::chrono::nanoseconds, kCount> durations_{};
  std::array<PollEntry, 32> wait_polls_{};
  std::array<uint64_t, 2> wait_counts_{};
  uint64_t overflow_ = 0;
  uint64_t queue_acquire_count_ = 0;
  Time last_{};
  Bucket current_ = Bucket::kOther;
  bool running_ = false;
};

FrameCpuTimeline& GetFrameCpuTimeline();

}  // namespace rex::graphics
