#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <tuple>
#include <utility>
#include <vector>

namespace rex::graphics::vulkan {
// Diagnostic resolve-extent provenance only. Never consulted by cache decisions.
// Extents may include padding; zero writers mean unobserved, not unchanged.
class ResolveChainAudit {
 public:
  struct Sample {
    bool valid = false, previous = false, same_observed_writers = false;
    uint64_t unknown_bytes = 0;
    std::map<uint64_t, uint64_t> producer_bytes;
  };
  void ObserveFrame(uint64_t frame) {
    if (have_frame_ && (frame < frame_ || frame > frame_ + 1)) {
      ranges_.clear();
      loads_.clear();
    }
    have_frame_ = true;
    frame_ = frame;
  }
  uint64_t Write(uint32_t start, uint32_t length) {
    if (!Valid(start, length))
      return 0;
    const uint32_t end = start + length;
    Split(start);
    Split(end);
    ranges_.erase(ranges_.lower_bound(start), ranges_.lower_bound(end));
    const uint64_t id = ++producer_id_;
    ranges_.emplace(start, Region{end, id});
    return id;
  }
  Sample Load(uint64_t texture, bool mip, uint32_t start, uint32_t length) {
    Sample result;
    const auto key = std::make_pair(texture, mip);
    if (!Valid(start, length)) {
      loads_.erase(key);
      return result;
    }
    result.valid = true;
    Snapshot current{start, length, {}};
    uint32_t cursor = start, end = start + length;
    auto it = ranges_.upper_bound(start);
    if (it != ranges_.begin())
      --it;
    while (cursor < end) {
      while (it != ranges_.end() && it->second.end <= cursor)
        ++it;
      uint32_t stop = end;
      uint64_t id = 0;
      if (it != ranges_.end()) {
        if (it->first > cursor)
          stop = std::min(end, it->first);
        else {
          stop = std::min(end, it->second.end);
          id = it->second.producer;
        }
      }
      current.versions.emplace_back(cursor - start, stop - cursor, id);
      if (id)
        result.producer_bytes[id] += stop - cursor;
      else
        result.unknown_bytes += stop - cursor;
      cursor = stop;
    }
    const auto previous = loads_.find(key);
    result.previous = previous != loads_.end();
    result.same_observed_writers = result.previous && previous->second.start == start &&
                                   previous->second.length == length &&
                                   previous->second.versions == current.versions;
    loads_[key] = std::move(current);
    return result;
  }
  void ForgetTexture(uint64_t id) {
    loads_.erase({id, false});
    loads_.erase({id, true});
  }

 private:
  static bool Valid(uint32_t start, uint32_t length) {
    return length && uint64_t(start) + length <= 0x20000000u;
  }
  struct Region {
    uint32_t end;
    uint64_t producer;
  };
  void Split(uint32_t point) {
    auto it = ranges_.upper_bound(point);
    if (it == ranges_.begin())
      return;
    --it;
    if (it->first == point || it->second.end <= point)
      return;
    Region tail = it->second;
    it->second.end = point;
    ranges_.emplace(point, tail);
  }
  struct Snapshot {
    uint32_t start, length;
    std::vector<std::tuple<uint32_t, uint32_t, uint64_t>> versions;
  };
  std::map<uint32_t, Region> ranges_;
  std::map<std::pair<uint64_t, bool>, Snapshot> loads_;
  uint64_t producer_id_ = 0, frame_ = 0;
  bool have_frame_ = false;
};
}  // namespace rex::graphics::vulkan
