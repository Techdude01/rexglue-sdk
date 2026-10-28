// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <charconv>
#include <limits>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string_view>

namespace rex::graphics::vulkan::diagnostics {

inline constexpr bool FragmentDiagnosticCountersValid(
    uint32_t entry_count, uint32_t entry_roi_count, uint32_t eligible_count,
    uint32_t eligible_roi_count, bool entry_counters_recorded = true) {
  if (eligible_roi_count > eligible_count) return false;
  if (!entry_counters_recorded) return entry_count == 0 && entry_roi_count == 0;
  return entry_roi_count <= entry_count && eligible_count <= entry_count;
}

inline constexpr char kFragmentDiagnosticEnablePath[] =
    "/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/tf14-fp32-probe/enable.txt";
inline constexpr char kFragmentDiagnosticStatusPath[] =
    "/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/tf14-fp32-probe/status.log";
inline constexpr uint64_t kDefaultFragmentDiagnosticFrame = 8680;
inline constexpr uint64_t kFragmentDiagnosticPixelHash = UINT64_C(0x79A3448304FA0D22);
inline constexpr uint64_t kFragmentDiagnosticGBufferPixelHash = UINT64_C(0xED58716070696DA4);
inline constexpr uint64_t kFragmentDiagnosticDeferredPixelHash = UINT64_C(0x3CC2A52F0A5CB063);
enum class FragmentDiagnosticShaderTarget : uint8_t { kHead, kGBuffer, kDeferred, kProducerConsumer };

inline FragmentDiagnosticShaderTarget ParseFragmentDiagnosticShaderTarget(std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line == "shader=gbuffer") return FragmentDiagnosticShaderTarget::kGBuffer;
      if (line == "shader=deferred") return FragmentDiagnosticShaderTarget::kDeferred;
      if (line == "shader=producer-consumer")
        return FragmentDiagnosticShaderTarget::kProducerConsumer;
      if (line == "shader=head") return FragmentDiagnosticShaderTarget::kHead;
      if (line.starts_with("shader=")) return FragmentDiagnosticShaderTarget::kHead;
    }
    if (newline == std::string_view::npos) break;
    content.remove_prefix(newline + 1);
  }
  return FragmentDiagnosticShaderTarget::kHead;
}

inline constexpr uint64_t GetFragmentDiagnosticPixelHash(FragmentDiagnosticShaderTarget target) {
  switch (target) {
    case FragmentDiagnosticShaderTarget::kGBuffer: return kFragmentDiagnosticGBufferPixelHash;
    case FragmentDiagnosticShaderTarget::kDeferred: return kFragmentDiagnosticDeferredPixelHash;
    default: return kFragmentDiagnosticPixelHash;
  }
}

inline constexpr bool IsFragmentDiagnosticShaderTarget(uint64_t pixel_hash,
                                                       FragmentDiagnosticShaderTarget target) {
  return pixel_hash == GetFragmentDiagnosticPixelHash(target) ||
         (target == FragmentDiagnosticShaderTarget::kProducerConsumer &&
          pixel_hash == kFragmentDiagnosticDeferredPixelHash);
}

// Frame-6000 replay chest/jacket patch at 1x, selected from the Ben body mesh
// coverage in the lower-strip head/body draw. Coordinates are output-space.
inline constexpr uint32_t kFragmentDiagnosticNoseMinX = 628;
inline constexpr uint32_t kFragmentDiagnosticNoseMaxX = 643;
inline constexpr uint32_t kFragmentDiagnosticNoseMinY = 445;
inline constexpr uint32_t kFragmentDiagnosticNoseMaxY = 460;
// Frame 8680 resolve-map evidence places the lower strip 384 guest rows after
// the first strip. Face diagnostics now run at 1x by default, so the
// output-space Y offset is 384 pixels. Keep candidate sampling opt-in; this
// maps the configured output nose ROI into lower-strip local FragCoord space
// without changing default rendering.
inline constexpr uint32_t kFragmentDiagnosticLowerStripGuestRows = 384;
inline constexpr uint32_t kFragmentDiagnosticCaptureScaleY = 1;
inline constexpr uint32_t kFragmentDiagnosticLowerStripOutputOffsetY =
    kFragmentDiagnosticLowerStripGuestRows * kFragmentDiagnosticCaptureScaleY;
static_assert(kFragmentDiagnosticNoseMinY >= kFragmentDiagnosticLowerStripOutputOffsetY);
inline constexpr uint32_t kFragmentDiagnosticLocalCandidateMinX = kFragmentDiagnosticNoseMinX;
// Include the right-hand derivative lane for the retained (614,12) sample.
inline constexpr uint32_t kFragmentDiagnosticLocalCandidateMaxX = kFragmentDiagnosticNoseMaxX + 1;
inline constexpr uint32_t kFragmentDiagnosticLocalCandidateMinY =
    kFragmentDiagnosticNoseMinY - kFragmentDiagnosticLowerStripOutputOffsetY;
inline constexpr uint32_t kFragmentDiagnosticLocalCandidateMaxY =
    kFragmentDiagnosticNoseMaxY - kFragmentDiagnosticLowerStripOutputOffsetY;

inline constexpr bool IsFragmentDiagnosticInstrumentedShader(uint64_t hash) {
  return hash == kFragmentDiagnosticPixelHash || hash == UINT64_C(0x2D4C010D3D75DEA5) ||
         hash == UINT64_C(0x3EB44C3A6D2D267B) ||
         hash == kFragmentDiagnosticGBufferPixelHash ||
         hash == kFragmentDiagnosticDeferredPixelHash;
}
inline constexpr int GetFragmentDiagnosticFetchSampleIndex(uint64_t shader_hash,
                                                            uint32_t fetch_slot) {
  if (shader_hash == kFragmentDiagnosticGBufferPixelHash) return fetch_slot == 4 ? 0 : -1;
  if (shader_hash == kFragmentDiagnosticDeferredPixelHash) {
    return fetch_slot == 11 ? 0 : fetch_slot == 13 ? 1 : -1;
  }
  return fetch_slot == 0 ? 0 : fetch_slot == 2 ? 1 : fetch_slot == 14 ? 2 : -1;
}
inline constexpr std::array<uint32_t, 3> GetFragmentDiagnosticRecordFetchSlots(
    uint64_t shader_hash) {
  if (shader_hash == kFragmentDiagnosticGBufferPixelHash) return {4, UINT32_MAX, UINT32_MAX};
  if (shader_hash == kFragmentDiagnosticDeferredPixelHash) return {11, 13, UINT32_MAX};
  return {0, 2, 14};
}
inline constexpr uint32_t GetFragmentDiagnosticCandidateMinY(uint64_t shader_hash) {
  return IsFragmentDiagnosticInstrumentedShader(shader_hash) &&
                 shader_hash != kFragmentDiagnosticPixelHash &&
                 shader_hash != UINT64_C(0x2D4C010D3D75DEA5) &&
                 shader_hash != UINT64_C(0x3EB44C3A6D2D267B)
             ? kFragmentDiagnosticNoseMinY
             : kFragmentDiagnosticLocalCandidateMinY;
}
inline constexpr uint32_t GetFragmentDiagnosticCandidateMaxY(uint64_t shader_hash) {
  return IsFragmentDiagnosticInstrumentedShader(shader_hash) &&
                 shader_hash != kFragmentDiagnosticPixelHash &&
                 shader_hash != UINT64_C(0x2D4C010D3D75DEA5) &&
                 shader_hash != UINT64_C(0x3EB44C3A6D2D267B)
             ? kFragmentDiagnosticNoseMaxY
             : kFragmentDiagnosticLocalCandidateMaxY;
}
// Frame 6000 of the replay route emits the lower-strip head draw after more
// than 128 candidate draws, so the multi-draw ceiling must cover it.
inline constexpr uint32_t kFragmentDiagnosticMaxCensusDraws = 4096;
inline constexpr uint32_t kFragmentDiagnosticMaxDrawStateLogEntries = 4096;
// Each eligible local sample stores tf0, tf2, and tf14 records. The bounded
// frame-8680 ROI run observed 8,368 triples (25,104 records) before overflow.
inline constexpr uint32_t kFragmentDiagnosticMaxRecords = 32768;
static_assert(kFragmentDiagnosticMaxRecords >= 3 * 8368);

// Combined vertex-output captures also need pixel instrumentation: the vertex
// dump selector chooses the draw, but it must not suppress the fragment side
// of that same diagnostic draw.
inline constexpr uint32_t GetPixelDiagnosticStorageVersion(
    bool enabled, bool vertex_input_dump, bool combined_vertex_output,
    bool local_overwrite_census, bool candidate_shader,
    bool local_candidate_output) {
  if (!enabled || (vertex_input_dump && !combined_vertex_output) ||
      (!local_overwrite_census && !candidate_shader)) {
    return 0;
  }
  return local_overwrite_census ? 9 : (local_candidate_output ? 10 : 8);
}

inline constexpr bool IsFragmentDiagnosticNoseCoordinate(uint32_t x, uint32_t y) {
  return x >= kFragmentDiagnosticNoseMinX && x <= kFragmentDiagnosticNoseMaxX &&
         y >= kFragmentDiagnosticNoseMinY && y <= kFragmentDiagnosticNoseMaxY;
}

inline constexpr bool IsFragmentDiagnosticLocalCandidateCoordinate(uint32_t x, uint32_t y) {
  return x >= kFragmentDiagnosticLocalCandidateMinX &&
         x <= kFragmentDiagnosticLocalCandidateMaxX &&
         y >= kFragmentDiagnosticLocalCandidateMinY && y <= kFragmentDiagnosticLocalCandidateMaxY;
}

// Existing 'enabled' content keeps the historical frame. Only a complete,
// positive frame=N line overrides it; reserve two frames for near-miss logging.
inline uint64_t ParseFragmentDiagnosticFrame(std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line.starts_with("frame=")) {
        line.remove_prefix(6);
        uint64_t frame = 0;
        const auto result = std::from_chars(line.data(), line.data() + line.size(), frame);
        if (result.ec == std::errc{} && result.ptr == line.data() + line.size() && frame >= 1 &&
            frame <= std::numeric_limits<uint64_t>::max() - 2) {
          return frame;
        }
        return kDefaultFragmentDiagnosticFrame;
      }
    }
    if (newline == std::string_view::npos)
      break;
    content.remove_prefix(newline + 1);
  }
  return kDefaultFragmentDiagnosticFrame;
}

inline uint32_t ParseFragmentDiagnosticDrawOrdinal(std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line.starts_with("draw=")) {
        line.remove_prefix(5);
        uint64_t draw = 0;
        const auto result = std::from_chars(line.data(), line.data() + line.size(), draw);
        if (result.ec == std::errc{} && result.ptr == line.data() + line.size() &&
            draw <= UINT32_MAX) {
          return uint32_t(draw);
        }
        return 0;
      }
    }
    if (newline == std::string_view::npos) break;
    content.remove_prefix(newline + 1);
  }
  return 0;
}

inline bool ParseFragmentDiagnosticLocalCandidateOutputMode(std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line == "mode=local-candidate-output")
        return true;
      if (line.starts_with("mode="))
        return false;
    }
    if (newline == std::string_view::npos)
      break;
    content.remove_prefix(newline + 1);
  }
  return false;
}

inline bool ParseFragmentDiagnosticLocalCandidateTriplesMode(std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line == "mode=local-candidate-triples")
        return true;
      if (line.starts_with("mode="))
        return false;
    }
    if (newline == std::string_view::npos)
      break;
    content.remove_prefix(newline + 1);
  }
  return false;
}

inline bool ParseFragmentDiagnosticOverwriteCensusMode(std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line == "mode=overwrite-census")
        return true;
      if (line.starts_with("mode="))
        return false;
    }
    if (newline == std::string_view::npos)
      break;
    content.remove_prefix(newline + 1);
  }
  return false;
}

inline bool ParseFragmentDiagnosticLocalOverwriteCensusMode(std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line == "mode=local-overwrite-census")
        return true;
      if (line.starts_with("mode="))
        return false;
    }
    if (newline == std::string_view::npos)
      break;
    content.remove_prefix(newline + 1);
  }
  return false;
}

inline bool ParseFragmentDiagnosticCensusMode(std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line == "mode=census")
        return true;
      if (line.starts_with("mode="))
        return false;
    }
    if (newline == std::string_view::npos)
      break;
    content.remove_prefix(newline + 1);
  }
  return false;
}

enum class FragmentDiagnosticTf14FilterMode : uint8_t {
  kDefault,
  kPoint,
  kLinear,
};

inline FragmentDiagnosticTf14FilterMode ParseFragmentDiagnosticTf14FilterMode(
    std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line.starts_with("sampler=")) {
        if (line == "sampler=tf14-point")
          return FragmentDiagnosticTf14FilterMode::kPoint;
        if (line == "sampler=tf14-linear")
          return FragmentDiagnosticTf14FilterMode::kLinear;
        return FragmentDiagnosticTf14FilterMode::kDefault;
      }
    }
    if (newline == std::string_view::npos)
      break;
    content.remove_prefix(newline + 1);
  }
  return FragmentDiagnosticTf14FilterMode::kDefault;
}

inline constexpr std::string_view FragmentDiagnosticTf14FilterModeName(
    FragmentDiagnosticTf14FilterMode mode) {
  switch (mode) {
    case FragmentDiagnosticTf14FilterMode::kPoint:
      return "tf14-point";
    case FragmentDiagnosticTf14FilterMode::kLinear:
      return "tf14-linear";
    default:
      return "default";
  }
}

inline bool IsFragmentDiagnosticTarget(uint64_t frame, uint64_t pixel_hash,
                                       uint64_t capture_frame,
                                       FragmentDiagnosticShaderTarget target =
                                           FragmentDiagnosticShaderTarget::kHead) {
  return frame == capture_frame && IsFragmentDiagnosticShaderTarget(pixel_hash, target);
}

inline bool IsFragmentDiagnosticTf14FilterTarget(FragmentDiagnosticTf14FilterMode mode,
                                                 uint64_t frame, uint64_t pixel_hash,
                                                 uint32_t fetch_constant,
                                                 uint64_t capture_frame) {
  return mode != FragmentDiagnosticTf14FilterMode::kDefault && frame == capture_frame &&
         pixel_hash == kFragmentDiagnosticPixelHash && fetch_constant == 14;
}

// Opt-in tf0 mip-filter A/B. The guest requests point mip filtering; this gate
// changes only the host mip mode for fetch constant 0 of the verified head PS,
// leaving minification, magnification and anisotropy at their guest values.
enum class FragmentDiagnosticTf0MipFilterMode : uint8_t {
  kDefault,
  kPoint,
  kLinear,
};

inline FragmentDiagnosticTf0MipFilterMode ParseFragmentDiagnosticTf0MipFilterMode(
    std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line == "tf0_mip=point")
        return FragmentDiagnosticTf0MipFilterMode::kPoint;
      if (line == "tf0_mip=linear")
        return FragmentDiagnosticTf0MipFilterMode::kLinear;
      if (line.starts_with("tf0_mip="))
        return FragmentDiagnosticTf0MipFilterMode::kDefault;
    }
    if (newline == std::string_view::npos)
      break;
    content.remove_prefix(newline + 1);
  }
  return FragmentDiagnosticTf0MipFilterMode::kDefault;
}

inline constexpr std::string_view FragmentDiagnosticTf0MipFilterModeName(
    FragmentDiagnosticTf0MipFilterMode mode) {
  switch (mode) {
    case FragmentDiagnosticTf0MipFilterMode::kPoint:
      return "tf0-mip-point";
    case FragmentDiagnosticTf0MipFilterMode::kLinear:
      return "tf0-mip-linear";
    default:
      return "default";
  }
}

inline bool IsFragmentDiagnosticTf0MipFilterTarget(FragmentDiagnosticTf0MipFilterMode mode,
                                                   uint64_t frame, uint64_t pixel_hash,
                                                   uint32_t fetch_constant,
                                                   uint64_t capture_frame) {
  return mode != FragmentDiagnosticTf0MipFilterMode::kDefault && frame == capture_frame &&
         pixel_hash == kFragmentDiagnosticPixelHash && fetch_constant == 0;
}

// The rotating application log is not an evidence store. Append and flush each
// bounded status event separately; callers must report any persistence failure.
inline bool AppendFragmentDiagnosticStatus(const std::filesystem::path& path,
                                           std::string_view message) {
  std::error_code error;
  std::filesystem::create_directories(path.parent_path(), error);
  if (error)
    return false;
  std::ofstream file(path, std::ios::app);
  file << message << '\n';
  file.flush();
  return file.good();
}

// First eight raw draw candidates per frame, plus the first expected PS in
// ordinary mode. Comprehensive mode records up to a separate draw-state cap,
// reserving a slot for the expected PS. This observes (but never consumes) the
// capture latch.
class FragmentDiagnosticDrawLog {
 public:
  struct FrameSummary {
    bool available = false;
    uint64_t frame = 0;
    uint32_t observed_draws = 0;
    uint32_t logged_entries = 0;
    bool truncated = false;
  };

  explicit FragmentDiagnosticDrawLog(uint64_t capture_frame = kDefaultFragmentDiagnosticFrame,
                                      bool comprehensive = false,
                                      uint64_t target_shader_hash = kFragmentDiagnosticPixelHash)
      : capture_frame_(capture_frame),
        comprehensive_(comprehensive),
        target_shader_hash_(target_shader_hash) {}

  bool Observe(uint64_t frame, uint64_t pixel_hash, FrameSummary* completed_frame = nullptr) {
    if (completed_frame)
      *completed_frame = {};
    if (frame != frame_) {
      if (completed_frame && comprehensive_ && frame_ == capture_frame_) {
        completed_frame->available = true;
        completed_frame->frame = frame_;
        completed_frame->observed_draws = ordinal_;
        completed_frame->logged_entries = emitted_count_;
        completed_frame->truncated = truncated_;
      }
      frame_ = frame;
      ordinal_ = 0;
      target_seen_ = false;
      emitted_count_ = 0;
      truncated_ = false;
    }
    if (frame < capture_frame_ - 1 || frame > capture_frame_ + 2)
      return false;

    ++ordinal_;
    const bool first_target = pixel_hash == target_shader_hash_ && !target_seen_;
    target_seen_ |= pixel_hash == target_shader_hash_;
    if (comprehensive_) {
      if (frame != capture_frame_)
        return false;
      if (first_target && emitted_count_ < kFragmentDiagnosticMaxDrawStateLogEntries) {
        ++emitted_count_;
        return true;
      }
      const uint32_t ordinary_limit =
          target_seen_ ? kFragmentDiagnosticMaxDrawStateLogEntries
                       : kFragmentDiagnosticMaxDrawStateLogEntries - 1;
      if (emitted_count_ < ordinary_limit) {
        ++emitted_count_;
        return true;
      }
      truncated_ = true;
      return false;
    }
    return ordinal_ <= 8 || first_target;
  }

  uint32_t ordinal() const { return ordinal_; }

 private:
  uint64_t capture_frame_;
  uint64_t frame_ = UINT64_MAX;
  uint32_t ordinal_ = 0;
  bool target_seen_ = false;
  bool comprehensive_ = false;
  uint64_t target_shader_hash_ = kFragmentDiagnosticPixelHash;
  uint32_t emitted_count_ = 0;
  bool truncated_ = false;
};

// Exact keys, not hashes: once per fetch/base-page/format/dimension/sign, with a
// hard process-context budget so even continually changing textures cannot spam.
class FragmentDiagnosticDescriptorLog {
 public:
  struct Key {
    uint32_t fetch, base_page, format, dimension;
    bool is_signed, has_binding;
    bool operator==(const Key&) const = default;
  };
  static constexpr uint32_t kMaxEntries = 32;
  bool ShouldLog(bool enabled, const Key& key) {
    if (!enabled || count_ == kMaxEntries)
      return false;
    for (uint32_t i = 0; i < count_; ++i) {
      if (keys_[i] == key)
        return false;
    }
    keys_[count_++] = key;
    return true;
  }

 private:
  std::array<Key, kMaxEntries> keys_{};
  uint32_t count_ = 0;
};

}  // namespace rex::graphics::vulkan::diagnostics
