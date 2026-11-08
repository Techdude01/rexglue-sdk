#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <map>
#include <tuple>

namespace rex::graphics::vulkan {

// Diagnostic command-stream placement, not measured Metal encoder counts.
enum class WorkPurpose : uint32_t {
  kUnknownDispatch,
  kTextureGuest,
  kTextureScaledResolve,
  kTextureConvert,
  kResolveCopy,
  kResolveClearDepth,
  kResolveClearColor,
  kHostDepthStore,
  kRtSnapshotColor,
  kRtSnapshotDepth,
  kRtStore,
  kSwap,
  kFxaa,
  kDownscale,
  kBufferCopy,
  kImageCopy,
  kSharedUpload,
  kQueryCopy,
  kClearImage,
  kCount
};
inline constexpr const char* kWorkPurposeNames[] = {"unknown_dispatch",
                                                    "texture_guest",
                                                    "texture_scaled_resolve",
                                                    "texture_convert",
                                                    "resolve_copy",
                                                    "resolve_clear_depth",
                                                    "resolve_clear_color",
                                                    "host_depth_store",
                                                    "rt_snapshot_color",
                                                    "rt_snapshot_depth",
                                                    "rt_store",
                                                    "swap",
                                                    "fxaa",
                                                    "downscale",
                                                    "buffer_copy",
                                                    "image_copy",
                                                    "shared_upload",
                                                    "query_copy",
                                                    "clear_image"};
static_assert(std::size(kWorkPurposeNames) == size_t(WorkPurpose::kCount));

class WorkPurposeAudit {
 public:
  struct Key {
    WorkPurpose purpose;
    uint32_t x, y, z;
    uint64_t bytes;
    bool operator<(const Key& b) const {
      return std::tie(purpose, x, y, z, bytes) < std::tie(b.purpose, b.x, b.y, b.z, b.bytes);
    }
  };
  struct Counts {
    uint64_t count = 0, prologue = 0, before_render = 0, between = 0, tail = 0;
  };
  void BeginSubmission() {
    seen_render_ = false;
    pending_.clear();
  }
  void Record(Key key, bool prologue = false) {
    auto& row = rows[key];
    ++row.count;
    if (prologue)
      ++row.prologue;
    else if (!seen_render_)
      ++row.before_render;
    else
      ++pending_[key];
  }
  void BeginRender() {
    ++render_scopes;
    if (!pending_.empty()) {
      ++interrupted_gaps;
      std::array<bool, size_t(WorkPurpose::kCount)> causes{};
      for (const auto& [key, count] : pending_) {
        rows[key].between += count;
        causes[size_t(key.purpose)] = true;
      }
      for (size_t i = 0; i < causes.size(); ++i)
        if (causes[i])
          ++gaps_by_cause[i];
      pending_.clear();
    } else if (seen_render_)
      ++unattributed_gaps;
    seen_render_ = true;
  }
  void EndSubmission() {
    for (const auto& [key, count] : pending_)
      rows[key].tail += count;
    pending_.clear();
    seen_render_ = false;
  }
  void Reset() { *this = WorkPurposeAudit{}; }
  std::map<Key, Counts> rows;
  std::array<uint64_t, size_t(WorkPurpose::kCount)> gaps_by_cause{};
  uint64_t render_scopes = 0, interrupted_gaps = 0, unattributed_gaps = 0;

 private:
  bool seen_render_ = false;
  std::map<Key, uint64_t> pending_;
};
}  // namespace rex::graphics::vulkan
