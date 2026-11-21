// Copyright 2026 ReXGlue contributors. SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace rex::input {

// Canonical controller state, independent of host bindings and guest byte order.
struct InputTraceState {
  uint32_t result = 0;
  uint32_t packet = 0;
  uint16_t buttons = 0;
  uint8_t lt = 0, rt = 0;
  int16_t lx = 0, ly = 0, rx = 0, ry = 0;
  bool operator==(const InputTraceState&) const = default;
  bool neutral() const { return !buttons && !lt && !rt && !lx && !ly && !rx && !ry; }
  bool released() const {
    // XInput's conventional radial rest zones, used only at an explicitly
    // requested route end. Never quantize the recorded gamepad samples.
    return !buttons && !lt && !rt &&
           int64_t(lx) * lx + int64_t(ly) * ly <= int64_t(7849) * 7849 &&
           int64_t(rx) * rx + int64_t(ry) * ry <= int64_t(8689) * 8689;
  }
};

struct InputTraceEvent {
  uint64_t micros = 0, poll = 0;
  InputTraceState state;
};

// Called under InputSystem's trace mutex. Time is supplied by the consumer so
// recording/replay can be tested without sleeps or a running window/game.
class InputTrace {
 public:
  static constexpr uint64_t kMaxMicros = 900000000;
  static constexpr uint64_t kMaxPolls = 1000000;
  static constexpr size_t kMaxEvents = 65536;
  ~InputTrace();
  InputTrace() = default;
  InputTrace(const InputTrace&) = delete;
  InputTrace& operator=(const InputTrace&) = delete;
  bool OpenRecord(const std::filesystem::path& path);
  bool OpenReplay(const std::filesystem::path& path, bool by_poll = false);
  bool Record(uint64_t micros, const InputTraceState& state, bool finish = false);
  InputTraceState Replay(uint64_t micros);
  bool recording() const { return file_ != nullptr; }
  bool replaying() const { return replay_; }
  bool complete() const { return complete_; }
  const std::string& error() const { return error_; }
  void RejectKeystroke() { Fail("guest keystroke events are not supported by controller-state recording"); }

 private:
  bool Fail(std::string message);
  FILE* file_ = nullptr;
  std::vector<InputTraceEvent> events_;
  InputTraceState last_{};
  uint64_t polls_ = 0, end_micros_ = 0, end_poll_ = 0, last_micros_ = 0;
  size_t cursor_ = 0;
  bool have_last_ = false, replay_ = false, by_poll_ = false, complete_ = false;
  std::string error_;
};
}  // namespace rex::input
