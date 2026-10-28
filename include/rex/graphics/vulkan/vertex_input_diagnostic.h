// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <string_view>

namespace rex::graphics::vulkan::diagnostics {

inline constexpr uint64_t kVertexInputDiagnosticFirstFrame = 8680;
inline constexpr uint64_t kVertexInputDiagnosticSecondFrame = 8690;
inline constexpr uint64_t kVertexInputDiagnosticSnapshotFrame = 6000;
inline constexpr uint64_t kVertexInputDiagnosticVertexHash = UINT64_C(0x4197798218650D43);
inline constexpr uint64_t kVertexInputDiagnosticPixelHash = UINT64_C(0x79A3448304FA0D22);
inline constexpr uint64_t kGBufferVertexDiagnosticVertexHash = UINT64_C(0x41B062C33EE4A804);
inline constexpr uint64_t kGBufferVertexDiagnosticPixelHash = UINT64_C(0xED58716070696DA4);
inline constexpr uint64_t kBenCuffVertexDiagnosticVertexHash = UINT64_C(0x046523BA1EB2717E);
inline constexpr uint64_t kBenCuffVertexDiagnosticPixelHash = UINT64_C(0x2D4C010D3D75DEA5);
inline constexpr uint64_t kBenDialVertexDiagnosticVertexHash = UINT64_C(0x8A449D0796A656D7);
inline constexpr uint64_t kBenDialVertexDiagnosticPixelHash = UINT64_C(0x6E6F06115807B602);
inline constexpr uint64_t kBenGlowVertexDiagnosticVertexHash = UINT64_C(0x7A0841365E75A823);
inline constexpr uint64_t kBenGlowVertexDiagnosticPixelHash = UINT64_C(0x7ABDFEFBCADB4BF9);
// Zero selects the first matching lower-strip head-shader draw in the captured
// frame. The saved controller trace can vary the draw sequence between replays.
inline constexpr uint32_t kVertexInputDiagnosticTargetDrawOrdinal = 0;
inline constexpr uint32_t kVertexInputDiagnosticTargetScissorHeight = 336;
inline constexpr uint32_t kVertexInputDiagnosticMaxBytes = 16 * 1024 * 1024;
inline constexpr uint32_t kVertexOutputDiagnosticStageMarker = 2;

inline constexpr bool IsVertexOutputRecordInBounds(uint32_t record_index,
                                                   uint32_t capacity) {
  return capacity != 0 && record_index < capacity;
}

inline bool ParseVertexInputDiagnosticMode(std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line == "mode=vertex-input-dump")
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

inline bool ParseBenOcclusionVertexOutputMode(std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line == "mode=ben-occlusion-output") return true;
      if (line.starts_with("mode=")) return false;
    }
    if (newline == std::string_view::npos) break;
    content.remove_prefix(newline + 1);
  }
  return false;
}

inline bool ParseCombinedLocalCandidateVertexOutputMode(std::string_view content) {
  while (!content.empty()) {
    const auto newline = content.find('\n');
    auto line = content.substr(0, newline);
    const auto first = line.find_first_not_of(" \t\r");
    if (first != std::string_view::npos) {
      line.remove_prefix(first);
      line = line.substr(0, line.find_last_not_of(" \t\r") + 1);
      if (line == "vertex_output=1") return true;
      if (line.starts_with("vertex_output=")) return false;
    }
    if (newline == std::string_view::npos) break;
    content.remove_prefix(newline + 1);
  }
  return false;
}

inline bool IsVertexInputDiagnosticFrame(uint64_t frame) {
  return frame == kVertexInputDiagnosticFirstFrame || frame == kVertexInputDiagnosticSecondFrame ||
         frame == kVertexInputDiagnosticSnapshotFrame;
}

inline uint32_t GetVertexInputDiagnosticFrameSlot(uint64_t frame) {
  return frame == kVertexInputDiagnosticFirstFrame ? 0
         : frame == kVertexInputDiagnosticSecondFrame ? 1
                                                     : 2;
}

inline bool IsVertexInputDiagnosticHeadDraw(uint64_t frame, uint64_t draw_ordinal,
                                            uint32_t scissor_height, uint64_t vertex_hash,
                                            uint64_t pixel_hash) {
  const bool old_head_target = scissor_height == kVertexInputDiagnosticTargetScissorHeight &&
                               vertex_hash == kVertexInputDiagnosticVertexHash &&
                               pixel_hash == kVertexInputDiagnosticPixelHash;
  const bool gbuffer_head_target = scissor_height == 720 &&
                                   vertex_hash == kGBufferVertexDiagnosticVertexHash &&
                                   pixel_hash == kGBufferVertexDiagnosticPixelHash;
  return frame == kVertexInputDiagnosticSnapshotFrame &&
         (!kVertexInputDiagnosticTargetDrawOrdinal ||
          draw_ordinal == kVertexInputDiagnosticTargetDrawOrdinal) &&
         (old_head_target || gbuffer_head_target);
}

inline uint32_t GetBenOcclusionVertexOutputDiagnosticTarget(
    uint64_t frame, uint32_t scissor_height, uint64_t vertex_hash, uint64_t pixel_hash,
    uint32_t index_count, uint32_t depth_control) {
  if (frame != kVertexInputDiagnosticSnapshotFrame || scissor_height != 336) return 0;
  if (vertex_hash == kBenCuffVertexDiagnosticVertexHash &&
      pixel_hash == kBenCuffVertexDiagnosticPixelHash && index_count == 1836 &&
      depth_control == UINT32_C(0xC8700536)) return 1;
  if (vertex_hash == kBenDialVertexDiagnosticVertexHash &&
      pixel_hash == kBenDialVertexDiagnosticPixelHash && index_count == 144 &&
      depth_control == UINT32_C(0xC8700532)) return 2;
  if (vertex_hash == kBenGlowVertexDiagnosticVertexHash &&
      pixel_hash == kBenGlowVertexDiagnosticPixelHash && index_count == 40 &&
      depth_control == UINT32_C(0xC8700532)) return 4;
  if (vertex_hash == kBenGlowVertexDiagnosticVertexHash &&
      pixel_hash == kBenGlowVertexDiagnosticPixelHash && index_count == 60 &&
      depth_control == UINT32_C(0xC8700532)) return 4;
  return 0;
}

inline bool IsVertexInputDiagnosticRangeValid(uint32_t physical_address, uint64_t byte_count) {
  return byte_count && byte_count <= kVertexInputDiagnosticMaxBytes &&
         byte_count - 1 <= UINT32_MAX - uint64_t(physical_address);
}

}  // namespace rex::graphics::vulkan::diagnostics
