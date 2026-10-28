// Copyright 2026 ReXGlue contributors. SPDX-License-Identifier: BSD-3-Clause
#include <rex/input/input_trace.h>

#include <fstream>
#include <sstream>
#include <utility>

namespace rex::input {
InputTrace::~InputTrace() {
  if (file_) std::fclose(file_);
}

bool InputTrace::Fail(std::string message) {
  error_ = std::move(message);
  if (file_) std::fclose(file_);
  file_ = nullptr;
  return false;
}

bool InputTrace::OpenRecord(const std::filesystem::path& path) {
  if (file_ || replay_ || complete_) return Fail("trace already configured");
  // Exclusive create: never replace a user's route or follow an existing link.
  file_ = std::fopen(path.string().c_str(), "wx");
  if (!file_) return Fail("cannot exclusively create input trace");
  if (std::fputs("COSMIC_INPUT_TRACE 1\n", file_) < 0 || std::fflush(file_))
    return Fail("cannot write trace header");
  return true;
}

bool InputTrace::OpenReplay(const std::filesystem::path& path, bool by_poll) {
  if (file_ || replay_ || complete_) return Fail("trace already configured");
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec || size > 16 * 1024 * 1024) return Fail("missing or oversized input trace");
  std::ifstream input(path);
  std::string line;
  if (!std::getline(input, line) || line != "COSMIC_INPUT_TRACE 1")
    return Fail("unsupported input trace header");
  std::vector<InputTraceEvent> parsed;
  bool ended = false;
  uint64_t end_time = 0, end_poll = 0;
  while (std::getline(input, line)) {
    if (ended || line.size() > 512) return Fail("data after end or oversized trace row");
    std::istringstream row(line);
    std::string kind, extra;
    row >> kind;
    if (kind == "END") {
      if (!(row >> end_time >> end_poll) || (row >> extra) || parsed.empty() ||
          end_time < parsed.back().micros || end_time > kMaxMicros ||
          end_poll < parsed.back().poll || end_poll > kMaxPolls ||
          !parsed.back().state.neutral())
        return Fail("invalid end marker (release all controls before finishing)");
      ended = true;
      continue;
    }
    InputTraceEvent event;
    uint64_t result, packet, buttons, lt, rt;
    int64_t lx, ly, rx, ry;
    if (kind != "S" ||
        !(row >> event.micros >> event.poll >> result >> packet >> buttons >> lt >> rt >>
          lx >> ly >> rx >> ry) || (row >> extra) ||
        event.micros > kMaxMicros || !event.poll || event.poll > kMaxPolls ||
        (result != 0 && result != 1167) || packet > UINT32_MAX || buttons > UINT16_MAX ||
        lt > UINT8_MAX || rt > UINT8_MAX ||
        lx < INT16_MIN || lx > INT16_MAX || ly < INT16_MIN || ly > INT16_MAX ||
        rx < INT16_MIN || rx > INT16_MAX || ry < INT16_MIN || ry > INT16_MAX ||
        parsed.size() >= kMaxEvents || (parsed.empty() && (event.micros != 0 || event.poll != 1)) ||
        (!parsed.empty() && (event.micros < parsed.back().micros || event.poll <= parsed.back().poll)))
      return Fail("invalid or out-of-order input trace row");
    event.state = {uint32_t(result), uint32_t(packet), uint16_t(buttons), uint8_t(lt),
                   uint8_t(rt), int16_t(lx), int16_t(ly), int16_t(rx), int16_t(ry)};
    parsed.push_back(event);
  }
  if (!ended || input.bad()) return Fail("unfinished input recording");
  events_ = std::move(parsed);
  end_micros_ = end_time;
  end_poll_ = end_poll;
  by_poll_ = by_poll;
  replay_ = true;
  return true;
}

bool InputTrace::Record(uint64_t micros, const InputTraceState& state, bool finish) {
  if (!file_) return false;
  ++polls_;
  if (micros > kMaxMicros || polls_ > kMaxPolls)
    return Fail("input recording exceeded duration/poll bound");
  // Packet counters can change on every poll without a control transition.
  auto compare = state;
  compare.packet = last_.packet;
  if (!have_last_ || compare != last_) {
    if (++cursor_ > kMaxEvents) return Fail("input recording exceeded event bound");
    if (std::fprintf(file_, "S %llu %llu %u %u %u %u %u %d %d %d %d\n",
                     static_cast<unsigned long long>(micros),
                     static_cast<unsigned long long>(polls_), state.result, state.packet,
                     unsigned(state.buttons), unsigned(state.lt), unsigned(state.rt),
                     int(state.lx), int(state.ly), int(state.rx), int(state.ry)) < 0 ||
        std::fflush(file_)) return Fail("cannot persist input trace row");
    last_ = state;
    have_last_ = true;
  }
  // A completed route must end released, never replay a stuck movement key.
  if (finish && state.released()) {
    if (!state.neutral()) {
      // The user explicitly ended the route. Preserve the raw resting sample
      // above, then append a terminal release so existing version-1 readers
      // also see a fully neutral endpoint. This is an end-of-route action,
      // not dead-zone filtering of recorded gameplay input.
      if (++polls_ > kMaxPolls || ++cursor_ > kMaxEvents)
        return Fail("no room for terminal input release");
      if (std::fprintf(file_, "S %llu %llu %u %u 0 0 0 0 0 0 0\n",
                       static_cast<unsigned long long>(micros),
                       static_cast<unsigned long long>(polls_), state.result,
                       state.packet + 1) < 0 || std::fflush(file_))
        return Fail("cannot persist terminal input release");
    }
    if (std::fprintf(file_, "END %llu %llu\n", static_cast<unsigned long long>(micros),
                     static_cast<unsigned long long>(polls_)) < 0 || std::fflush(file_))
      return Fail("cannot finish input trace");
    const int result = std::fclose(file_);
    file_ = nullptr;
    if (result) return Fail("cannot close input trace");
    complete_ = true;
  }
  return true;
}

InputTraceState InputTrace::Replay(uint64_t micros) {
  ++polls_;
  // At most one transition per guest poll. Even if a slow replay misses both
  // timestamps of a tap, its press and release are each delivered once.
  if (cursor_ < events_.size() &&
      (by_poll_ ? polls_ >= events_[cursor_].poll : micros >= events_[cursor_].micros))
    last_ = events_[cursor_++].state;
  if (!complete_ && cursor_ == events_.size() &&
      (by_poll_ ? polls_ >= end_poll_ : micros >= end_micros_)) {
    complete_ = true;
    // Once the route has ended, keep no residual stick drift held forever.
    // All samples before the end marker remain byte-for-byte authentic.
    const uint32_t packet = last_.packet + (last_.neutral() ? 0 : 1);
    const uint32_t result = last_.result;
    last_ = {};
    last_.packet = packet;
    last_.result = result;
  }
  return last_;
}
}  // namespace rex::input
