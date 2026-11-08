// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <memory>
#include <vector>

namespace rex::graphics::vulkan {
// Command-thread owned registry. Callers serialize memory-watch notifications
// before lookup. A lease owns the immutable image, including after invalidation;
// submission retirement must retain the lease until GPU completion.
template <typename Key, typename Image>
class ResolvedImageOwnership {
 public:
  using Lease = std::shared_ptr<const Image>;
  explicit ResolvedImageOwnership(size_t capacity = 64, uint64_t byte_budget = UINT64_MAX)
      : capacity_(capacity), byte_budget_(byte_budget) {}
  uint64_t owned_bytes() const {
    uint64_t bytes = 0;
    for (const auto& entry : entries_) bytes += entry.bytes;
    return bytes;
  }
  void Invalidate(uint32_t address, uint32_t length) {
    if (!length) return;
    const uint64_t begin = address & 0x1FFFFFFFu;
    const uint64_t end = std::min(begin + length, uint64_t(0x20000000));
    entries_.erase(std::remove_if(entries_.begin(), entries_.end(),
        [=](const Entry& e) { return begin < e.end && e.begin < end; }), entries_.end());
  }
  // Publish only after conversion and visibility barriers have been recorded.
  // Partial writes invalidate older images even if the replacement is ineligible.
  bool Publish(uint32_t address, uint32_t length, const Key& key, Lease image, uint64_t bytes = 0) {
    Invalidate(address, length);
    const uint64_t begin = address & 0x1FFFFFFFu;
    if (!length || !image || begin + length > 0x20000000ull) return false;
    if (!capacity_ || bytes > byte_budget_) return false;
    // Eviction only drops address ownership. Submission and consumer leases
    // independently retain immutable allocations still referenced by the GPU.
    while (!entries_.empty() &&
           (entries_.size() >= capacity_ || owned_bytes() > byte_budget_ - bytes)) {
      entries_.erase(entries_.begin());
    }
    entries_.push_back({begin, begin + length, key, std::move(image), bytes});
    return true;
  }
  Lease Find(uint32_t address, uint32_t length, const Key& key) const {
    const uint64_t begin = address & 0x1FFFFFFFu;
    if (!length || begin + length > 0x20000000ull) return {};
    for (const auto& e : entries_) {
      // Subregions require a proven layout transform; exact extents only.
      if (e.begin == begin && e.end == begin + length && e.key == key) return e.image;
    }
    return {};
  }
  // Retain each sampled lease independently of address ownership. Invalidation
  // must never free resources referenced by an already recorded command buffer.
  void RetainForSubmission(Lease image, uint64_t submission) {
    if (image) pending_.push_back({submission, std::move(image)});
  }
  void CompletedSubmissionUpdated(uint64_t completed) {
    pending_.erase(std::remove_if(pending_.begin(), pending_.end(),
        [=](const Pending& p) { return p.submission <= completed; }), pending_.end());
  }
  void Clear() { entries_.clear(); }
 private:
  struct Entry { uint64_t begin, end; Key key; Lease image; uint64_t bytes; };
  struct Pending { uint64_t submission; Lease image; };
  size_t capacity_;
  uint64_t byte_budget_;
  std::vector<Entry> entries_;
  std::vector<Pending> pending_;
};
}  // namespace rex::graphics::vulkan
