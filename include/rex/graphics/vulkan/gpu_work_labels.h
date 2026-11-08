// Diagnostic identities only. These labels never represent measured GPU time.
#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <iterator>

namespace rex::graphics::vulkan {
struct ResolveWorkMetadata {
  bool valid = false;
  bool is_depth = false;
  // 0=legacy, 1=direct fused, 2=direct attempted then fallback, 3=parity candidate.
  uint32_t path = 0;
  uint32_t shader = 0;
  uint32_t msaa = 0;
  uint32_t sample_select = 0;
  uint32_t source_format = 0;
  uint32_t dest_format = 0;
  uint32_t dest_base = 0;
  uint32_t dest_start = 0;
  uint32_t dest_length = 0;
  char fallback_reason[40]{};
};

// Widest window one bounded label run may cover. A guest replay advances by wall
// clock while input is delivered per poll, so equal frame numbers are not equal
// simulation state across runs; a window wide enough to contain a whole scripted
// segment keeps a comparison possible without predicting the drift. Stays modest
// because every labelled frame emits one encoder-membership line per work item.
constexpr uint64_t kMaxGpuWorkLabelFrames = 384;
constexpr bool IsGpuWorkLabelFrame(bool enabled, uint64_t frame, uint64_t start, uint64_t count) {
  return enabled && start > 0 && count > 0 && count <= kMaxGpuWorkLabelFrames &&
         frame >= start && frame-start < count;
}
inline std::array<char,384> FormatGpuWorkLabel(uint64_t frame, uint64_t submission,
                                              bool prologue, uint64_t ordinal,
                                              const char* kind, uint64_t pipeline,
                                              uint32_t x=0, uint32_t y=0, uint32_t z=0,
                                              const ResolveWorkMetadata* resolve = nullptr) {
  std::array<char,384> result{};
  if (resolve && resolve->valid) {
    static constexpr const char* kPaths[] = {"legacy", "fused", "fallback", "parity_candidate"};
    const char* path = resolve->path < std::size(kPaths) ? kPaths[resolve->path] : "unknown";
    std::snprintf(result.data(),result.size(),
                  "CRGPU f=%llu sub=%llu stream=%s op=%llu kind=resolve_copy path=%s "
                  "class=%s shader=%u msaa=%u select=%u srcfmt=%u dstfmt=%u "
                  "base=%08X start=%08X len=%08X reason=%s pipe=%016llX x=%u y=%u z=%u",
                  static_cast<unsigned long long>(frame), static_cast<unsigned long long>(submission),
                  prologue ? "prologue" : "main", static_cast<unsigned long long>(ordinal), path,
                  resolve->is_depth ? "depth" : "color", resolve->shader, resolve->msaa,
                  resolve->sample_select, resolve->source_format, resolve->dest_format,
                  resolve->dest_base, resolve->dest_start, resolve->dest_length,
                  resolve->fallback_reason[0] ? resolve->fallback_reason : "none",
                  static_cast<unsigned long long>(pipeline), x,y,z);
  } else {
    std::snprintf(result.data(),result.size(),
                  "CRGPU f=%llu sub=%llu stream=%s op=%llu kind=%s pipe=%016llX x=%u y=%u z=%u",
                  static_cast<unsigned long long>(frame), static_cast<unsigned long long>(submission),
                  prologue ? "prologue" : "main", static_cast<unsigned long long>(ordinal), kind,
                  static_cast<unsigned long long>(pipeline), x,y,z);
  }
  return result;
}
} // namespace rex::graphics::vulkan
