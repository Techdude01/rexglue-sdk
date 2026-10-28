// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <limits>
#include <cstring>
#include <span>
#include <string_view>

namespace rex::graphics::vulkan::diagnostics {

inline constexpr char kRenderTargetSnapshotEnablePath[] =
    "/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/render-target-snapshot.txt";
inline constexpr char kRenderTargetSnapshotOutputPath[] =
    "/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/frame-6000-render-target-snapshot.bin";
inline constexpr char kBenOcclusionSnapshotOutputPath[] =
    "/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/frame-complete-route-ben-color-snapshot.bin";
inline constexpr char kBenPixelHistorySnapshotOutputPath[] =
    "/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/frame-complete-route-ben-pixel-history.bin";

inline constexpr uint32_t kRenderTargetSnapshotMagic = 0x53525458;  // XTRS
// Version 1 used multisample SPIR-V image types even for single-sample images.
// Its host_sample_count == 1 attachment reads are not valid Vulkan evidence.
inline constexpr uint32_t kRenderTargetSnapshotVersion = 2;
inline constexpr uint32_t kRenderTargetSnapshotMaxBoundaries = 768;
inline constexpr uint32_t kRenderTargetSnapshotWordsPerSample = 8;
inline constexpr uint32_t kBenOcclusionSnapshotMagic = 0x53434F42;  // BOCS
inline constexpr uint32_t kBenOcclusionSnapshotVersion = 1;
inline constexpr uint32_t kBenPixelHistorySnapshotVersion = 2;
inline constexpr uint64_t kBenOcclusionSnapshotMaxBytes = 256ull * 1024 * 1024;
inline constexpr uint32_t kBenOcclusionSnapshotMaxBoundaries = kRenderTargetSnapshotMaxBoundaries;
inline constexpr uint32_t kBenOcclusionSnapshotWordsPerSample = 2;
inline constexpr uint32_t kBenOcclusionCandidateCuff = 1;
inline constexpr uint32_t kBenOcclusionCandidateDial = 2;
inline constexpr uint32_t kBenOcclusionCandidateGlow = 3;
inline constexpr uint32_t kBenOcclusionCandidateAllDraw = 4;
inline constexpr uint32_t GetBenOcclusionCandidateKind(uint64_t vertex_hash,
                                                        uint64_t pixel_hash,
                                                        bool is_hud_glow = false) {
  if (vertex_hash == UINT64_C(0x046523BA1EB2717E) &&
      pixel_hash == UINT64_C(0x2D4C010D3D75DEA5)) {
    return kBenOcclusionCandidateCuff;
  }
  if (vertex_hash == UINT64_C(0x8A449D0796A656D7) &&
      pixel_hash == UINT64_C(0x6E6F06115807B602)) {
    return kBenOcclusionCandidateDial;
  }
  if (vertex_hash == UINT64_C(0x7A0841365E75A823) &&
      pixel_hash == UINT64_C(0x7ABDFEFBCADB4BF9) && !is_hud_glow) {
    return kBenOcclusionCandidateGlow;
  }
  return 0;
}

inline constexpr uint32_t GetBenOcclusionCandidateKind(uint64_t vertex_hash,
                                                        uint64_t pixel_hash,
                                                        [[maybe_unused]] uint32_t index_count,
                                                        [[maybe_unused]] uint32_t width,
                                                        uint32_t height) {
  return GetBenOcclusionCandidateKind(vertex_hash, pixel_hash, height != 336);
}
inline constexpr uint64_t kRenderTargetSnapshotHeadVertexHash = UINT64_C(0x4197798218650D43);
inline constexpr uint64_t kRenderTargetSnapshotHeadPixelHash = UINT64_C(0x79A3448304FA0D22);
inline constexpr uint64_t kRenderTargetSnapshotGBufferVertexHash = UINT64_C(0x41B062C33EE4A804);
inline constexpr uint64_t kRenderTargetSnapshotGBufferPixelHash = UINT64_C(0xED58716070696DA4);
inline constexpr uint64_t kRenderTargetSnapshotDeferredVertexHash = UINT64_C(0x2FD14D3D03C603F3);
inline constexpr uint64_t kRenderTargetSnapshotDeferredPixelHash = UINT64_C(0x3CC2A52F0A5CB063);

enum class RenderTargetSnapshotPoint : uint32_t {
  kBeforeHeadDraw = 0,
  kAfterHeadDraw = 1,
  kAfterSubsequentDraw = 2,
};

enum class RenderTargetSnapshotTarget : uint32_t {
  kHeadShader = 0,
  kGBufferHeadShader = 1,
  kDeferredLighting = 2,
  kQuadPartner = 3,
  kBenCuff = 4,
};

struct RenderTargetSnapshotBoundary {
  uint64_t frame;
  uint64_t color_image;
  uint64_t depth_image;
  uint64_t vertex_shader_hash;
  uint64_t pixel_shader_hash;
  uint32_t draw_ordinal;
  uint32_t point;
  uint32_t color_format;
  uint32_t depth_format;
  uint32_t host_sample_count;
  // These are two logical capture slots. For a single-sample image both map
  // to host sample 0 and do not constitute two independent MSAA samples.
  uint32_t guest_sample_0_host_index;
  uint32_t guest_sample_1_host_index;
  uint32_t depth_control;
  uint32_t color_control;
  uint32_t stencil_ref_mask;
  uint32_t blend_control_0;
  uint32_t color_info_0;
  uint32_t depth_info;
  uint32_t color_write_mask;
  uint32_t coverage_control;
  uint32_t alpha_ref;
  uint32_t scissor_x;
  uint32_t scissor_y;
  uint32_t scissor_width;
  uint32_t scissor_height;
  uint32_t capture_flags;
  uint32_t width;
  uint32_t height;
  uint32_t payload_word_offset;
};

struct RenderTargetSnapshotSample {
  uint32_t color[4];  // Raw UINT transfer-view bits; exact for supported host paths.
  uint32_t depth_bits;  // IEEE-754 bits returned by the depth sampled-image view.
  uint32_t stencil;
  uint32_t host_sample_index;
  uint32_t guest_sample_index;
};

struct RenderTargetSnapshotFileHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t frame;
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
  uint32_t boundary_count;
  uint32_t sample_count_per_pixel;
  uint32_t sample_record_size;
  uint32_t overflow;
};

struct BenOcclusionSnapshotFileHeader {
  uint32_t magic;
  uint32_t version;
  uint64_t frame;
  uint32_t x;
  uint32_t y;
  uint32_t width;
  uint32_t height;
  uint32_t boundary_count;
  uint32_t logical_samples_per_pixel;
  uint32_t sample_record_size;
  uint32_t overflow;
  uint32_t color_encoding;  // RGBA8 packed as UNORM4x8.
  uint32_t depth_encoding;  // 0 when omitted, 1 for IEEE-754 float bits.
  uint32_t boundary_metadata_size;
};

struct BenOcclusionSnapshotBoundaryMetadata {
  float viewport_x;
  float viewport_y;
  uint32_t viewport_width;
  uint32_t viewport_height;
  float viewport_min_depth;
  float viewport_max_depth;
  uint32_t index_count;
  uint32_t candidate_kind;
  uint32_t capture_origin_y;
  uint32_t output_y_offset;
  uint32_t host_color_vk_format;
  uint32_t renderpass_attachment_mask;
  uint32_t eyeball_depth_bias_active;
  uint32_t depth_attachment_bound;
};

struct BenOcclusionSnapshotView {
  BenOcclusionSnapshotFileHeader header{};
  std::span<const uint8_t> boundary_records;
  std::span<const uint8_t> boundary_metadata_records;
  std::span<const uint8_t> sample_records;
};

static_assert(sizeof(RenderTargetSnapshotSample) == 32);
static_assert(sizeof(BenOcclusionSnapshotFileHeader) == 64);
static_assert(sizeof(BenOcclusionSnapshotBoundaryMetadata) == 56);

inline bool ParseBenOcclusionSnapshot(std::span<const uint8_t> bytes,
                                      BenOcclusionSnapshotView& view) {
  if (bytes.size() < sizeof(BenOcclusionSnapshotFileHeader)) return false;
  BenOcclusionSnapshotFileHeader header{};
  std::memcpy(&header, bytes.data(), sizeof(header));
  const bool pixel_history = header.version == kBenPixelHistorySnapshotVersion;
  const bool supported_shape = pixel_history
      ? header.x == 512 && header.y == 0 && header.width == 256 && header.height == 128
      : header.x == 512 && header.y == 0 && header.width == 256 && header.height == 720;
  const uint32_t expected_metadata_size = pixel_history ? 56u : 32u;
  if (header.magic != kBenOcclusionSnapshotMagic ||
      (header.version != kBenOcclusionSnapshotVersion && !pixel_history) || header.frame == 0 ||
      !supported_shape ||
      header.logical_samples_per_pixel != 2 || header.sample_record_size != 8 ||
      header.boundary_count == 0 ||
      header.boundary_count > kBenOcclusionSnapshotMaxBoundaries ||
      header.color_encoding != 1 || header.depth_encoding != 0 ||
      header.boundary_metadata_size != expected_metadata_size) {
    return false;
  }
  const uint64_t boundary_bytes =
      uint64_t(header.boundary_count) * sizeof(RenderTargetSnapshotBoundary);
  const uint64_t metadata_bytes =
      uint64_t(header.boundary_count) * header.boundary_metadata_size;
  const uint64_t payload_bytes = uint64_t(header.width) * header.height *
                                 header.logical_samples_per_pixel * header.sample_record_size *
                                 header.boundary_count;
  const uint64_t total_bytes = sizeof(header) + boundary_bytes + metadata_bytes + payload_bytes;
  if (payload_bytes > kBenOcclusionSnapshotMaxBytes || total_bytes != bytes.size()) return false;
  view.header = header;
  view.boundary_records = bytes.subspan(sizeof(header), size_t(boundary_bytes));
  view.boundary_metadata_records = bytes.subspan(sizeof(header) + size_t(boundary_bytes),
                                                  size_t(metadata_bytes));
  view.sample_records = bytes.subspan(sizeof(header) + size_t(boundary_bytes) + size_t(metadata_bytes),
                                     size_t(payload_bytes));
  return true;
}

inline bool ReadBenOcclusionSnapshotBoundary(const BenOcclusionSnapshotView& view,
                                              uint32_t index,
                                              RenderTargetSnapshotBoundary& boundary) {
  if (index >= view.header.boundary_count ||
      view.boundary_records.size() !=
          size_t(view.header.boundary_count) * sizeof(RenderTargetSnapshotBoundary)) {
    return false;
  }
  std::memcpy(&boundary,
              view.boundary_records.data() + size_t(index) * sizeof(boundary),
              sizeof(boundary));
  const uint32_t candidate_kind = boundary.capture_flags >> 16;
  return (candidate_kind == kBenOcclusionCandidateCuff ||
          candidate_kind == kBenOcclusionCandidateDial ||
          candidate_kind == kBenOcclusionCandidateGlow ||
          (view.header.version == kBenPixelHistorySnapshotVersion &&
           candidate_kind == kBenOcclusionCandidateAllDraw)) &&
         (boundary.host_sample_count == 1 || boundary.host_sample_count == 2) &&
         (boundary.point == uint32_t(RenderTargetSnapshotPoint::kBeforeHeadDraw) ||
          boundary.point == uint32_t(RenderTargetSnapshotPoint::kAfterHeadDraw));
}

inline constexpr bool IsBenOcclusionLogicalSampleValid(
    const RenderTargetSnapshotBoundary& boundary, uint32_t logical_sample) {
  return logical_sample < 2 && logical_sample < boundary.host_sample_count &&
         !(logical_sample == 1 && (boundary.capture_flags & 16u));
}

inline bool ReadBenOcclusionSnapshotBoundaryMetadata(
    const BenOcclusionSnapshotView& view, uint32_t index,
    BenOcclusionSnapshotBoundaryMetadata& metadata) {
  if (index >= view.header.boundary_count ||
      view.boundary_metadata_records.size() !=
          size_t(view.header.boundary_count) * view.header.boundary_metadata_size) {
    return false;
  }
  metadata = {};
  std::memcpy(&metadata,
              view.boundary_metadata_records.data() +
                  size_t(index) * view.header.boundary_metadata_size,
              view.header.boundary_metadata_size);
  return metadata.viewport_width != 0 && metadata.viewport_height != 0 &&
         (metadata.candidate_kind == kBenOcclusionCandidateCuff ||
          metadata.candidate_kind == kBenOcclusionCandidateDial ||
          metadata.candidate_kind == kBenOcclusionCandidateGlow ||
          (view.header.version == kBenPixelHistorySnapshotVersion &&
           metadata.candidate_kind == kBenOcclusionCandidateAllDraw));
}

inline constexpr std::array<uint32_t, 2> GetGuest2xHostSampleIndices(bool host_2x_supported) {
  return host_2x_supported ? std::array<uint32_t, 2>{0, 1}
                          : std::array<uint32_t, 2>{0, 3};
}

// Configuration contract for a future attachment snapshot consumer. Keeping
// this parser independent of Vulkan makes the default-off and allocation bounds
// testable without creating images, ending passes, or allocating GPU memory.
struct RenderTargetSnapshotRequest {
  static constexpr uint32_t kMaxRoiDimension = 512;
  static constexpr uint64_t kMaxSnapshotBytes = 1 * 1024 * 1024;

  bool enabled = false;
  uint64_t frame = 0;
  uint32_t draw_ordinal = 0;
  uint32_t color_target = 0;
  uint32_t x = 0;
  uint32_t y = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  bool color = false;
  bool depth = false;
  bool after_each_draw = false;
  bool compact_ben = false;
  bool ben_pixel_history = false;
  RenderTargetSnapshotTarget target = RenderTargetSnapshotTarget::kHeadShader;

  constexpr bool valid() const {
    if (compact_ben) {
      const bool supported_shape = ben_pixel_history
          ? x == 512 && y == 0 && width == 256 && height == 128
          : x == 512 && y == 0 && width == 256 && height == 720;
      return enabled && frame && color && !depth && target == RenderTargetSnapshotTarget::kBenCuff &&
             !draw_ordinal && supported_shape &&
             uint64_t(width) * uint64_t(height) * 2u * 8u <= kBenOcclusionSnapshotMaxBytes;
    }
    if (!enabled || !frame || !width || !height || (!color && !depth) ||
        width > kMaxRoiDimension || height > kMaxRoiDimension || color_target > 3 ||
        (after_each_draw && !color && !depth) ||
        uint64_t(x) + width > UINT32_MAX || uint64_t(y) + height > UINT32_MAX) {
      return false;
    }
    const uint64_t bytes_per_pixel = sizeof(RenderTargetSnapshotSample) * 2u;
    return uint64_t(width) * uint64_t(height) * bytes_per_pixel <= kMaxSnapshotBytes;
  }

  constexpr bool matches(uint64_t current_frame, uint32_t current_draw) const {
    return valid() && frame == current_frame &&
           (!draw_ordinal || draw_ordinal == current_draw);
  }

  constexpr uint64_t payload_bytes() const {
    return compact_ben ? uint64_t(width) * uint64_t(height) * 2u * 8u
                       : uint64_t(width) * uint64_t(height) * sizeof(RenderTargetSnapshotSample) * 2u;
  }
};

inline bool ParseRenderTargetSnapshotUnsigned(std::string_view value, uint64_t& out) {
  if (value.empty()) return false;
  uint64_t parsed = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) return false;
  out = parsed;
  return true;
}

inline bool ParseRenderTargetSnapshotMode(std::string_view content) {
  while (!content.empty()) {
    const size_t newline = content.find('\n');
    auto line = content.substr(0, newline);
    const size_t first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line == "mode=render-target-snapshot" || line == "mode=ben-occlusion-snapshot" ||
          line == "mode=ben-pixel-history") return true;
      if (line.starts_with("mode=")) return false;
    }
    if (newline == std::string_view::npos) break;
    content.remove_prefix(newline + 1);
  }
  return false;
}

inline RenderTargetSnapshotRequest ParseRenderTargetSnapshotRequest(std::string_view content) {
  RenderTargetSnapshotRequest request;
  bool have_mode = false;
  bool have_frame = false, have_draw = false, have_x = false, have_y = false;
  bool have_width = false, have_height = false, have_color = false, have_depth = false;
  while (!content.empty()) {
    const size_t newline = content.find('\n');
    auto line = content.substr(0, newline);
    const size_t first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      const size_t last = line.find_last_not_of(" \t\r");
      line = line.substr(0, last + 1);
      const size_t equal = line.find('=');
      if (equal != std::string_view::npos) {
        const auto key = line.substr(0, equal);
        const auto value = line.substr(equal + 1);
        uint64_t number = 0;
        if (key == "mode") {
          have_mode = value == "render-target-snapshot" || value == "ben-occlusion-snapshot" ||
                      value == "ben-pixel-history";
          request.compact_ben = value == "ben-occlusion-snapshot" || value == "ben-pixel-history";
          request.ben_pixel_history = value == "ben-pixel-history";
          if (!have_mode) return {};
        } else if (key == "enabled") {
          request.enabled = value == "1";
        } else if (key == "color") {
          request.color = value == "1";
          have_color = true;
        } else if (key == "depth") {
          request.depth = value == "1";
          have_depth = true;
        } else if (key == "after_each_draw") {
          request.after_each_draw = value == "1";
        } else if (key == "target") {
          if (value == "quad-partner") {
            request.target = RenderTargetSnapshotTarget::kQuadPartner;
          } else if (value == "head") {
            request.target = RenderTargetSnapshotTarget::kHeadShader;
          } else if (value == "gbuffer") {
            request.target = RenderTargetSnapshotTarget::kGBufferHeadShader;
          } else if (value == "deferred") {
            request.target = RenderTargetSnapshotTarget::kDeferredLighting;
          } else if (value == "ben-cuff") {
            request.target = RenderTargetSnapshotTarget::kBenCuff;
          } else if (value == "ben-pixel-history") {
            request.target = RenderTargetSnapshotTarget::kBenCuff;
          } else {
            return {};
          }
        } else if (!ParseRenderTargetSnapshotUnsigned(value, number)) {
          return {};
        } else if (key == "frame" && number > 0) {
          request.frame = number;
          have_frame = true;
        } else if (key == "draw" && number <= UINT32_MAX) {
          request.draw_ordinal = uint32_t(number);
          have_draw = true;
        } else if (key == "color_target" && number <= 3) {
          request.color_target = uint32_t(number);
        } else if (key == "x" && number <= UINT32_MAX) {
          request.x = uint32_t(number);
          have_x = true;
        } else if (key == "y" && number <= UINT32_MAX) {
          request.y = uint32_t(number);
          have_y = true;
        } else if (key == "width" && number <= UINT32_MAX) {
          request.width = uint32_t(number);
          have_width = true;
        } else if (key == "height" && number <= UINT32_MAX) {
          request.height = uint32_t(number);
          have_height = true;
        } else {
          return {};
        }
      }
    }
    if (newline == std::string_view::npos) break;
    content.remove_prefix(newline + 1);
  }
  if (!(have_mode && request.enabled && have_frame && have_draw && have_x && have_y && have_width &&
        have_height && have_color && have_depth && request.valid())) {
    return {};
  }
  return request;
}

}  // namespace rex::graphics::vulkan::diagnostics
