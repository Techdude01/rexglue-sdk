/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <fstream>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <mutex>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <SPIRV/GlslangToSpv.h>
#include <glslang/Public/ShaderLang.h>
#include <rex/assert.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/pipeline/shader/shader.h>
#include <rex/graphics/pipeline/shader/spirv_translator.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/registers.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/frame_capture_schedule.h>
#include <rex/graphics/vulkan/scope_profile_timing.h>
#include <rex/graphics/vulkan/transfer_scope_reuse.h>
#include <rex/graphics/vulkan/vertex_input_diagnostic.h>
#include <rex/graphics/vulkan/quad_partner_marker.h>
#include <rex/graphics/vulkan/standalone_derivative_probe.h>
#include <rex/graphics/vulkan/pipeline_cache.h>
#include <rex/graphics/vulkan/render_area.h>
#include <rex/graphics/vulkan/render_target_cache.h>
#include <rex/graphics/vulkan/shader.h>
#include <rex/graphics/vulkan/shared_memory.h>
#include <rex/graphics/xenos.h>
#include <rex/kernel/xboxkrnl/video.h>
#include <rex/types.h>
#include <rex/memory/utils.h>
#include <rex/ui/flags.h>
#include <rex/ui/vulkan/presenter.h>
#include <rex/ui/vulkan/util.h>

#ifndef REX_UNLIKELY
#define REX_UNLIKELY(x) __builtin_expect(!!(x), 0)
#endif

REXCVAR_DECLARE(bool, gpu_wait_correlation);

// Record only candidate destinations covering the diagnostic address.
static void TraceWaitDestination(const char* kind, uint64_t frame, uint32_t address, uint32_t length) {
  constexpr uint64_t target = 0x1fca3004;
  if (!REXCVAR_GET(gpu_wait_correlation) || uint64_t(address) > target ||
      uint64_t(address) + length <= target) return;
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  REXGPU_INFO("[WAIT_CORRELATION] event={} ts_ns={} frame={} address={} length={}",
              kind, ns, frame, address, length);
}

// Legacy backend compatibility aliases for shared readback controls.
REXCVAR_DEFINE_BOOL(gpu_frame_timeline_enable, false, "GPU/Vulkan",
                    "Enable exclusive CPU timeline within frame audit windows");

REXCVAR_DEFINE_BOOL(vulkan_readback_resolve, false, "GPU/Vulkan",
                    "Read render-to-texture results on the CPU")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(vulkan_readback_memexport, false, "GPU/Vulkan",
                    "Read data written by memory export in shaders on the CPU")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(vulkan_async_skip_incomplete_frames, true, "GPU/Vulkan",
                    "When async shader compilation is enabled, skip presenting frames that "
                    "used placeholder pipelines to avoid visible flashing")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(vulkan_submit_on_primary_buffer_end, true, "GPU/Vulkan",
                    "Submit command buffer when PM4 primary buffer ends")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(vulkan_dynamic_rendering, true, "GPU/Vulkan",
                    "Use VK_KHR_dynamic_rendering for Vulkan GPU emulation when supported by the "
                    "device (falls back to render passes otherwise)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(vulkan_scissor_render_area, false, "GPU/Vulkan",
                    "Bound load-only guest dynamic rendering to the draw scissor; "
                    "preserve full clears and restart scopes when coverage expands")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_hoist_shared_memory_uploads, false, "GPU/Vulkan",
    "Collect and hoist shared-memory buffer uploads into a submission prologue before "
    "render passes, avoiding render pass splits.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_prune_unused_attachment_load_store, false, "GPU/Vulkan",
    "Prune redundant load and store memory operations (DONT_CARE) for unneeded or "
    "transient attachments that are cleared before load and never resolved or sampled.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_elide_dead_edram_transfers, false, "GPU/Vulkan",
    "Elide redundant EDRAM transfers whose destination attachment is overwritten or "
    "cleared before being read.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_elide_intertile_transfers, false, "GPU/Vulkan",
    "Elide redundant intermediate EDRAM depth transfers between predicated scene tiles.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_merge_matching_attachment_passes, false, "GPU/Vulkan",
    "Coalesce consecutive draws with matching framebuffer and attachment configurations into "
    "the same render pass, avoiding pass restart stalls on dynamic scissor changes.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_prune_transfer_stencil, false, "GPU/Vulkan",
    "Use DONT_CARE for stencil load/store in depth transfer passes to eliminate VRAM traffic.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_alias_4x_depth_to_1x, false, "GPU/Vulkan",
    "Alias depth-only 4x MSAA passes to 1x host render target to avoid EDRAM transfer thrashing.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_elide_intermediate_depth_transfers, false, "GPU/Vulkan",
    "Elide intermediate depth transfers on base-0 (shadow-cascade thrash). Default OFF: user visual "
    "reject 2026-11-02 on catacomb — missing/dim lights and sticky Ben ground shadow/light vs real "
    "game; Xenos A/B fixed with this false. Do not re-enable without a matched-level visual pass.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_promote_viewport_resolve_clears, false, "GPU/Vulkan",
    "Promote full-viewport resolve clears to attachment loadOp=CLEAR on subsequent guest passes.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_coalesce_adjacent_clears, false, "GPU/Vulkan",
    "Coalesce adjacent depth/color resolve clear passes with identical attachments and hoist shared upload.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_early_z_opaque, false, "GPU/Vulkan",
    "Restore hardware EarlyFragmentTests / TBDR HSR on eligible opaque and depth read-only draws.")
    .lifecycle(rex::cvar::Lifecycle::kRequiresRestart);

REXCVAR_DEFINE_DOUBLE(
    vulkan_ablate_viewport_scale, 1.0, "GPU/Vulkan",
    "Diagnostic ablation: scale viewport dimensions to isolate fillrate from memory bandwidth.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_ablate_skip_transfers, false, "GPU/Vulkan",
    "Diagnostic ablation: completely skip all EDRAM transfers to establish upper bound of transfer tax.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    vulkan_ablate_skip_depth_transfers, false, "GPU/Vulkan",
    "Diagnostic ablation: skip only depth EDRAM transfers.")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(
    vulkan_ablate_depth_transfer_mask, 0, "GPU/Vulkan",
    "Diagnostic ablation: bitmask of depth transfer pass indices to elide (1 << pass_idx).")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(
    vulkan_ablate_skip_depth_transfer_width, 0, "GPU/Vulkan",
    "Diagnostic ablation: host width of depth transfer surfaces to elide (e.g. 2080).")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(
    gpu_provenance_frame, 0, "GPU/Vulkan",
    "Trace this guest frame through draws, resolve, slot-0 swap, and output capture; "
    "0 disables tracing")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(
    gpu_provenance_frame_window, 48, "GPU/Vulkan",
    "Trace consecutive frames [-N, +N] around gpu_provenance_frame; 0 disables window sampling")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    gpu_provenance_capture_raw, false, "GPU/Vulkan",
    "Save a pre-watermark PPM sidecar for the selected GPU provenance frame")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    gpu_profile_cpu, true, "GPU/Vulkan",
    "Profile and log CPU draw recording breakdown periodically (every 120 frames) or on provenance frames")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    gpu_profile_transfer_timestamps, false, "GPU/Vulkan",
    "Diagnostic transfer batch timestamps; may be coalesced by Metal and add overhead")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    gpu_profile_scope_timestamps, false, "GPU/Vulkan",
    "Diagnostic GPU timestamps around each render-pass scope, transfer scope, resolve and compute dispatch")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    gpu_frame_audit_enable, false, "GPU/Vulkan",
    "Enable machine-readable frame-time audit logging and CSV export for matched A/B measurement")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_BOOL(
    gpu_raster_variant_audit, false, "GPU/Vulkan",
    "Count two selected pixel shader variants and early-Z gates within a finite frame audit window");

REXCVAR_DEFINE_UINT64(
    gpu_frame_audit_window_start, 0, "GPU/Vulkan",
    "First guest frame to include in detailed frame-time audit (0 = from start)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_UINT64(
    gpu_frame_audit_window_end, 0, "GPU/Vulkan",
    "Last guest frame to include in detailed frame-time audit (0 = unbounded)")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_STRING(
    gpu_frame_audit_csv_path, "", "GPU/Vulkan",
    "File path to append machine-readable frame-time audit CSV records")
    .lifecycle(rex::cvar::Lifecycle::kHotReload);

REXCVAR_DEFINE_INT32(gpu_provenance_capture_interval, 0, "GPU/Vulkan",
                    "Capture FPS diagnostic images every N frames through gpu_provenance_frame; 0 disables")
    .range(0, 1000000);
REXCVAR_DEFINE_BOOL(vulkan_reuse_transfer_scope_for_guest, false, "GPU/Vulkan",
                    "Continue compatible single-attachment transfers into guest rendering");
REXCVAR_DEFINE_BOOL(vulkan_defer_non_conflicting_barriers, true, "GPU/Vulkan",
                    "Do not break active dynamic rendering pass for non-conflicting buffer and texture barriers");
REXCVAR_DEFINE_STRING(gpu_provenance_capture_directory, "", "GPU/Vulkan",
                     "Existing directory for diagnostic frame images; empty uses /tmp");

namespace rex::graphics::vulkan {

namespace {

bool RasterVariantAuditActive(uint64_t frame) {
  if (!REXCVAR_GET(gpu_raster_variant_audit)) return false;
  const uint64_t start = REXCVAR_GET(gpu_frame_audit_window_start);
  const uint64_t end = REXCVAR_GET(gpu_frame_audit_window_end);
  const bool audit_enabled = REXCVAR_GET(gpu_frame_audit_enable);
  if (!audit_enabled || start == 0 || end < start || end - start >= 1024) {
    static bool warned = false;
    if (!warned) {
      REXGPU_ERROR("[RASTER_VARIANT_DISABLED] requires enabled frame audit, start>0, "
                   "end>=start, and at most 1024 frames; start={} end={}", start, end);
      warned = true;
    }
    return false;
  }
  return IsRasterVariantWindow(true, audit_enabled, frame, start, end);
}

bool IsGpuProvenanceTargetFrame(uint64_t frame) {
  int32_t selected_frame = REXCVAR_GET(gpu_provenance_frame);
  return selected_frame > 0 && frame == uint64_t(selected_frame);
}

bool IsGpuProvenanceFrame(uint64_t frame) {
  if (IsGpuProvenanceTargetFrame(frame)) {
    return true;
  }
  if (IsPeriodicCaptureFrame(
          frame, uint32_t(std::max(0, REXCVAR_GET(gpu_provenance_capture_interval))),
          uint64_t(std::max(0, REXCVAR_GET(gpu_provenance_frame))))) {
    return true;
  }
  if (REXCVAR_GET(gpu_frame_audit_enable) &&
      (frame == 10200 || frame == 10500 || frame == 10800)) {
    return true;
  }
  int32_t selected_frame = REXCVAR_GET(gpu_provenance_frame);
  if (selected_frame <= 0) {
    return false;
  }
  int32_t window = std::max(0, REXCVAR_GET(gpu_provenance_frame_window));
  if (!window) {
    return false;
  }
  uint64_t selected = uint64_t(selected_frame);
  uint64_t lower = selected > uint64_t(window) ? selected - uint64_t(window) : 1;
  uint64_t upper = selected + uint64_t(window);
  // [flicker-diag] Every frame in the window, not every fifth frame: the UI-loss
  // glitch hits single frames, so a strided sample cannot catch it.
  return frame >= lower && frame <= upper;
}

namespace {

// [flicker-diag] Pillar 7: per-guest-frame counters for the single-frame UI-loss
// defect, written one CSV row per closing frame so the row for exactly the frame
// the readback flags as an outlier can be read directly. Off unless
// /private/tmp/cosmic-flicker/enable.txt exists (checked once per process).
bool FlickerDiagEnabled() {
  static const bool enabled = [] {
    std::ifstream file("/private/tmp/cosmic-flicker/enable.txt");
    return file.good();
  }();
  return enabled;
}

uint64_t flicker_frame_pixel_texture_slots = 0;
uint64_t flicker_frame_null_pixel_textures = 0;

struct FlickerFrameCsv {
  uint64_t copy_count = 0;
  uint64_t resolve_ok = 0;
  uint64_t resolve_fail = 0;
  uint32_t resolve_targets[16] = {};
  uint32_t resolve_target_count = 0;
  uint32_t resolve_length = 0;

  void NoteCopy(bool resolved, uint32_t address, uint32_t length) {
    ++copy_count;
    if (resolved) {
      ++resolve_ok;
    } else {
      ++resolve_fail;
    }
    resolve_length = length;
    for (uint32_t i = 0; i < resolve_target_count; ++i) {
      if (resolve_targets[i] == address) {
        return;
      }
    }
    if (resolve_target_count < uint32_t(sizeof(resolve_targets) / sizeof(resolve_targets[0]))) {
      resolve_targets[resolve_target_count++] = address;
    }
  }

  void Flush(uint64_t frame_number, uint64_t draw_count) {
    std::ofstream file("/private/tmp/cosmic-flicker/frames.csv", std::ios::app);
    if (file.tellp() == std::streampos(0)) {
      file << "frame,draws,copies,resolve_ok,resolve_fail,distinct_targets,target_bases,"
              "resolve_length,pixel_tex_slots,null_pixel_tex\n";
    }
    file << frame_number << ',' << draw_count << ',' << copy_count << ',' << resolve_ok << ','
         << resolve_fail << ',' << resolve_target_count << ',';
    for (uint32_t i = 0; i < resolve_target_count; ++i) {
      if (i) {
        file << '|';
      }
      file << "0x" << std::hex << resolve_targets[i] << std::dec;
    }
    file << ',' << resolve_length << ',' << flicker_frame_pixel_texture_slots << ','
         << flicker_frame_null_pixel_textures << '\n';
    // The run ends by timeout, so every row must reach the file immediately.
    file.flush();
    *this = FlickerFrameCsv{};
  }
};
FlickerFrameCsv flicker_frame_csv;

void DrawGlyph(uint8_t* rgba, uint32_t width, uint32_t height, uint32_t stride,
               int x, int y, char c, int scale) {
  static const uint8_t glyphs[128][8] = {
      [' '] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
      ['('] = {0x0C, 0x18, 0x30, 0x30, 0x30, 0x18, 0x0C, 0x00},
      [')'] = {0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x18, 0x30, 0x00},
      ['-'] = {0x00, 0x00, 0x00, 0x7E, 0x00, 0x00, 0x00, 0x00},
      ['.'] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x18, 0x18, 0x00},
      [':'] = {0x00, 0x18, 0x18, 0x00, 0x18, 0x18, 0x00, 0x00},
      ['|'] = {0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x18, 0x00},
      ['0'] = {0x3C, 0x66, 0x6E, 0x76, 0x66, 0x66, 0x3C, 0x00},
      ['1'] = {0x18, 0x38, 0x18, 0x18, 0x18, 0x18, 0x7E, 0x00},
      ['2'] = {0x3C, 0x66, 0x06, 0x1C, 0x30, 0x66, 0x7E, 0x00},
      ['3'] = {0x3C, 0x66, 0x06, 0x1C, 0x06, 0x66, 0x3C, 0x00},
      ['4'] = {0x0C, 0x1C, 0x34, 0x64, 0x7E, 0x04, 0x0E, 0x00},
      ['5'] = {0x7E, 0x60, 0x7C, 0x06, 0x06, 0x66, 0x3C, 0x00},
      ['6'] = {0x1C, 0x30, 0x60, 0x7C, 0x66, 0x66, 0x3C, 0x00},
      ['7'] = {0x7E, 0x66, 0x06, 0x0C, 0x18, 0x18, 0x18, 0x00},
      ['8'] = {0x3C, 0x66, 0x66, 0x3C, 0x66, 0x66, 0x3C, 0x00},
      ['9'] = {0x3C, 0x66, 0x66, 0x3E, 0x06, 0x0C, 0x38, 0x00},
      ['F'] = {0x7E, 0x60, 0x60, 0x7C, 0x60, 0x60, 0x60, 0x00},
      ['P'] = {0x7C, 0x66, 0x66, 0x7C, 0x60, 0x60, 0x60, 0x00},
      ['S'] = {0x3C, 0x66, 0x60, 0x3C, 0x06, 0x66, 0x3C, 0x00},
      ['r'] = {0x00, 0x00, 0x36, 0x3E, 0x30, 0x30, 0x30, 0x00},
      ['a'] = {0x00, 0x00, 0x3C, 0x06, 0x3E, 0x66, 0x3E, 0x00},
      ['e'] = {0x00, 0x00, 0x3C, 0x66, 0x7E, 0x60, 0x3C, 0x00},
      ['m'] = {0x00, 0x00, 0x7C, 0x54, 0x54, 0x54, 0x54, 0x00},
      ['s'] = {0x00, 0x00, 0x3C, 0x60, 0x3C, 0x06, 0x7C, 0x00},
  };
  const uint8_t* rows = glyphs[static_cast<uint8_t>(c) < 128 ? static_cast<uint8_t>(c) : 0];
  for (int row = 0; row < 8; ++row) {
    uint8_t bits = rows[row];
    for (int col = 0; col < 8; ++col) {
      if ((bits << col) & 0x80) {
        for (int dy = 0; dy < scale; ++dy) {
          int py = y + row * scale + dy;
          if (py < 0 || py >= int(height)) continue;
          for (int dx = 0; dx < scale; ++dx) {
            int px = x + col * scale + dx;
            if (px < 0 || px >= int(width)) continue;
            uint8_t* p = rgba + size_t(py) * stride + size_t(px) * 4;
            p[0] = 255;
            p[1] = 255;
            p[2] = 255;
          }
        }
      }
    }
  }
}

void StampFpsOverlay(ui::RawImage& image, uint64_t frame, double fps, double frame_time_ms) {
  if (image.width < 320 || image.height < 100 || !image.data.data()) {
    return;
  }
  char text[64];
  if (fps > 0.0) {
    std::snprintf(text, sizeof(text), "Frame %llu | %.1f FPS (%.2f ms)",
                  (unsigned long long)frame, fps, frame_time_ms);
  } else {
    std::snprintf(text, sizeof(text), "Frame %llu", (unsigned long long)frame);
  }

  const int scale = 2;
  const int char_width = 8 * scale;
  const int char_height = 8 * scale;
  const int text_len = int(std::strlen(text));
  const int pad_x = 10;
  const int pad_y = 6;
  const int badge_w = text_len * char_width + pad_x * 2;
  const int badge_h = char_height + pad_y * 2;
  const int start_x = int(image.width) - badge_w - 20;
  const int start_y = 16;

  // Dark translucent background box
  for (int y = start_y; y < start_y + badge_h && y < int(image.height); ++y) {
    uint8_t* row = image.data.data() + size_t(y) * image.stride;
    for (int x = start_x; x < start_x + badge_w && x < int(image.width); ++x) {
      uint8_t* p = row + size_t(x) * 4;
      p[0] = p[0] / 4;
      p[1] = p[1] / 4;
      p[2] = p[2] / 4;
    }
  }

  // Draw white text
  for (int i = 0; i < text_len; ++i) {
    DrawGlyph(image.data.data(), image.width, image.height, uint32_t(image.stride),
              start_x + pad_x + i * char_width, start_y + pad_y, text[i], scale);
  }
}

}  // namespace

void LogGpuProvenanceOutput(uint64_t frame, ui::RawImage& image, double fps, double frame_time_ms) {
  if (!image.width || !image.height || image.stride < size_t(image.width) * 4 ||
      image.data.size() < image.stride * image.height) {
    REXGPU_WARN("GPU provenance frame {} output capture has invalid dimensions/storage", frame);
    return;
  }
  uint64_t nonblack_pixels = 0;
  uint8_t minimum[3] = {UINT8_MAX, UINT8_MAX, UINT8_MAX};
  uint8_t maximum[3] = {0, 0, 0};
  for (uint32_t y = 0; y < image.height; ++y) {
    const uint8_t* row = image.data.data() + size_t(y) * image.stride;
    for (uint32_t x = 0; x < image.width; ++x) {
      const uint8_t* pixel = row + size_t(x) * 4;
      bool nonblack = false;
      for (size_t channel = 0; channel < 3; ++channel) {
        minimum[channel] = std::min(minimum[channel], pixel[channel]);
        maximum[channel] = std::max(maximum[channel], pixel[channel]);
        nonblack |= pixel[channel] != 0;
      }
      nonblack_pixels += nonblack;
    }
  }
  REXGPU_INFO(
      "GPU provenance frame {} captured guest output {}x{} nonblack_pixels={}/{} "
      "rgb_min={},{},{} rgb_max={},{},{} FPS={:.1f} ({:.2f} ms)",
      frame, image.width, image.height, nonblack_pixels, uint64_t(image.width) * image.height,
      minimum[0], minimum[1], minimum[2], maximum[0], maximum[1], maximum[2],
      fps, frame_time_ms);

  if (IsGpuProvenanceTargetFrame(frame) && REXCVAR_GET(gpu_provenance_capture_raw)) {
    if (const char* capture_path = std::getenv("COSMIC_GPU_PROVENANCE_CAPTURE_PATH")) {
      char raw_path[1024];
      int path_length = std::snprintf(raw_path, sizeof(raw_path), "%s.raw", capture_path);
      if (path_length > 0 && size_t(path_length) < sizeof(raw_path)) {
        if (FILE* f = std::fopen(raw_path, "wb")) {
          std::fprintf(f, "P6\n%u %u\n255\n", image.width, image.height);
          for (uint32_t y = 0; y < image.height; ++y) {
            const uint8_t* row = image.data.data() + size_t(y) * image.stride;
            for (uint32_t x = 0; x < image.width; ++x) {
              const uint8_t* pixel = row + size_t(x) * 4;
              std::fputc(pixel[0], f);
              std::fputc(pixel[1], f);
              std::fputc(pixel[2], f);
            }
          }
          std::fclose(f);
        }
      }
    }
  }

  // Stamp on-screen FPS and frame watermark for headless verification.
  StampFpsOverlay(image, frame, fps, frame_time_ms);

  std::string capture_directory = REXCVAR_GET(gpu_provenance_capture_directory);
  if (capture_directory.empty()) capture_directory = "/tmp";
  std::string auto_ppm_path = capture_directory + "/frame_" + std::to_string(frame) + ".ppm";
  if (FILE* f = std::fopen(auto_ppm_path.c_str(), "wb")) {
    std::fprintf(f, "P6\n%u %u\n255\n", image.width, image.height);
    for (uint32_t y = 0; y < image.height; ++y) {
      const uint8_t* row = image.data.data() + size_t(y) * image.stride;
      for (uint32_t x = 0; x < image.width; ++x) {
        const uint8_t* pixel = row + size_t(x) * 4;
        std::fputc(pixel[0], f);
        std::fputc(pixel[1], f);
        std::fputc(pixel[2], f);
      }
    }
    std::fclose(f);
  }

  if (IsGpuProvenanceTargetFrame(frame)) {
    if (const char* capture_path = std::getenv("COSMIC_GPU_PROVENANCE_CAPTURE_PATH")) {
      if (FILE* f = std::fopen(capture_path, "wb")) {
        std::fprintf(f, "P6\n%u %u\n255\n", image.width, image.height);
        for (uint32_t y = 0; y < image.height; ++y) {
          const uint8_t* row = image.data.data() + size_t(y) * image.stride;
          for (uint32_t x = 0; x < image.width; ++x) {
            const uint8_t* pixel = row + size_t(x) * 4;
            std::fputc(pixel[0], f);
            std::fputc(pixel[1], f);
            std::fputc(pixel[2], f);
          }
        }
        std::fclose(f);
      }
    }
  }
}

// glslang default built-in resource limits.
constexpr TBuiltInResource kGlslangDefaultTBuiltInResource = {
    /* .maxLights = */ 32,
    /* .maxClipPlanes = */ 6,
    /* .maxTextureUnits = */ 32,
    /* .maxTextureCoords = */ 32,
    /* .maxVertexAttribs = */ 64,
    /* .maxVertexUniformComponents = */ 4096,
    /* .maxVaryingFloats = */ 64,
    /* .maxVertexTextureImageUnits = */ 32,
    /* .maxCombinedTextureImageUnits = */ 80,
    /* .maxTextureImageUnits = */ 32,
    /* .maxFragmentUniformComponents = */ 4096,
    /* .maxDrawBuffers = */ 32,
    /* .maxVertexUniformVectors = */ 128,
    /* .maxVaryingVectors = */ 8,
    /* .maxFragmentUniformVectors = */ 16,
    /* .maxVertexOutputVectors = */ 16,
    /* .maxFragmentInputVectors = */ 15,
    /* .minProgramTexelOffset = */ -8,
    /* .maxProgramTexelOffset = */ 7,
    /* .maxClipDistances = */ 8,
    /* .maxComputeWorkGroupCountX = */ 65535,
    /* .maxComputeWorkGroupCountY = */ 65535,
    /* .maxComputeWorkGroupCountZ = */ 65535,
    /* .maxComputeWorkGroupSizeX = */ 1024,
    /* .maxComputeWorkGroupSizeY = */ 1024,
    /* .maxComputeWorkGroupSizeZ = */ 64,
    /* .maxComputeUniformComponents = */ 1024,
    /* .maxComputeTextureImageUnits = */ 16,
    /* .maxComputeImageUniforms = */ 8,
    /* .maxComputeAtomicCounters = */ 8,
    /* .maxComputeAtomicCounterBuffers = */ 1,
    /* .maxVaryingComponents = */ 60,
    /* .maxVertexOutputComponents = */ 64,
    /* .maxGeometryInputComponents = */ 64,
    /* .maxGeometryOutputComponents = */ 128,
    /* .maxFragmentInputComponents = */ 128,
    /* .maxImageUnits = */ 8,
    /* .maxCombinedImageUnitsAndFragmentOutputs = */ 8,
    /* .maxCombinedShaderOutputResources = */ 8,
    /* .maxImageSamples = */ 0,
    /* .maxVertexImageUniforms = */ 0,
    /* .maxTessControlImageUniforms = */ 0,
    /* .maxTessEvaluationImageUniforms = */ 0,
    /* .maxGeometryImageUniforms = */ 0,
    /* .maxFragmentImageUniforms = */ 8,
    /* .maxCombinedImageUniforms = */ 8,
    /* .maxGeometryTextureImageUnits = */ 16,
    /* .maxGeometryOutputVertices = */ 256,
    /* .maxGeometryTotalOutputComponents = */ 1024,
    /* .maxGeometryUniformComponents = */ 1024,
    /* .maxGeometryVaryingComponents = */ 64,
    /* .maxTessControlInputComponents = */ 128,
    /* .maxTessControlOutputComponents = */ 128,
    /* .maxTessControlTextureImageUnits = */ 16,
    /* .maxTessControlUniformComponents = */ 1024,
    /* .maxTessControlTotalOutputComponents = */ 4096,
    /* .maxTessEvaluationInputComponents = */ 128,
    /* .maxTessEvaluationOutputComponents = */ 128,
    /* .maxTessEvaluationTextureImageUnits = */ 16,
    /* .maxTessEvaluationUniformComponents = */ 1024,
    /* .maxTessPatchComponents = */ 120,
    /* .maxPatchVertices = */ 32,
    /* .maxTessGenLevel = */ 64,
    /* .maxViewports = */ 16,
    /* .maxVertexAtomicCounters = */ 0,
    /* .maxTessControlAtomicCounters = */ 0,
    /* .maxTessEvaluationAtomicCounters = */ 0,
    /* .maxGeometryAtomicCounters = */ 0,
    /* .maxFragmentAtomicCounters = */ 8,
    /* .maxCombinedAtomicCounters = */ 8,
    /* .maxAtomicCounterBindings = */ 1,
    /* .maxVertexAtomicCounterBuffers = */ 0,
    /* .maxTessControlAtomicCounterBuffers = */ 0,
    /* .maxTessEvaluationAtomicCounterBuffers = */ 0,
    /* .maxGeometryAtomicCounterBuffers = */ 0,
    /* .maxFragmentAtomicCounterBuffers = */ 1,
    /* .maxCombinedAtomicCounterBuffers = */ 1,
    /* .maxAtomicCounterBufferSize = */ 16384,
    /* .maxTransformFeedbackBuffers = */ 4,
    /* .maxTransformFeedbackInterleavedComponents = */ 64,
    /* .maxCullDistances = */ 8,
    /* .maxCombinedClipAndCullDistances = */ 8,
    /* .maxSamples = */ 4,
    /* .maxMeshOutputVerticesNV = */ 256,
    /* .maxMeshOutputPrimitivesNV = */ 512,
    /* .maxMeshWorkGroupSizeX_NV = */ 32,
    /* .maxMeshWorkGroupSizeY_NV = */ 1,
    /* .maxMeshWorkGroupSizeZ_NV = */ 1,
    /* .maxTaskWorkGroupSizeX_NV = */ 32,
    /* .maxTaskWorkGroupSizeY_NV = */ 1,
    /* .maxTaskWorkGroupSizeZ_NV = */ 1,
    /* .maxMeshViewCountNV = */ 4,
    /* .maxDualSourceDrawBuffersEXT = */ 1,
    /* .limits = */
    {
        /* .nonInductiveForLoops = */ 1,
        /* .whileLoops = */ 1,
        /* .doWhileLoops = */ 1,
        /* .generalUniformIndexing = */ 1,
        /* .generalAttributeMatrixVectorIndexing = */ 1,
        /* .generalVaryingIndexing = */ 1,
        /* .generalSamplerIndexing = */ 1,
        /* .generalVariableIndexing = */ 1,
        /* .generalConstantMatrixVectorIndexing = */ 1,
    },
};

const char* GetSwapFxaaComputeSource(bool extreme_quality) {
  return extreme_quality ? R"(#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform XeApplyGammaRampConstants {
  uvec2 xe_fxaa_size;
  vec2 xe_fxaa_size_inv;
};

layout(set = 0, binding = 0) uniform sampler2D xe_fxaa_source;
layout(set = 1, binding = 0, rgb10_a2) writeonly uniform image2D xe_fxaa_dest;

const vec3 kLumaWeights = vec3(0.299, 0.587, 0.114);
const float kEdgeThreshold = 0.063;
const float kEdgeThresholdMin = 0.0312;
const float kSpanMax = 12.0;
const float kDirReduceMul = 0.125;
const float kDirReduceMin = 1.0 / 128.0;

float SampleLuma(vec2 uv) {
  return textureLod(xe_fxaa_source, uv, 0.0).a;
}

void main() {
  uvec2 pixel = gl_GlobalInvocationID.xy;
  if (any(greaterThanEqual(pixel, xe_fxaa_size))) {
    return;
  }

  vec2 uv = (vec2(pixel) + vec2(0.5)) * xe_fxaa_size_inv;
  vec2 texel = xe_fxaa_size_inv;
  vec4 rgbm = textureLod(xe_fxaa_source, uv, 0.0);

  float lumaM = rgbm.a;
  float lumaNW = SampleLuma(uv + vec2(-texel.x, -texel.y));
  float lumaNE = SampleLuma(uv + vec2( texel.x, -texel.y));
  float lumaSW = SampleLuma(uv + vec2(-texel.x,  texel.y));
  float lumaSE = SampleLuma(uv + vec2( texel.x,  texel.y));

  float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
  float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
  float lumaRange = lumaMax - lumaMin;

  vec3 result = rgbm.rgb;
  if (lumaRange >= max(kEdgeThresholdMin, lumaMax * kEdgeThreshold)) {
    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max(
        (lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * kDirReduceMul),
        kDirReduceMin);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2(-kSpanMax), vec2(kSpanMax)) * texel;

    vec3 rgbA =
        0.5 *
        (textureLod(xe_fxaa_source, uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb +
         textureLod(xe_fxaa_source, uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb);
    vec3 rgbB =
        rgbA * 0.5 +
        0.25 *
            (textureLod(xe_fxaa_source, uv + dir * -0.5, 0.0).rgb +
             textureLod(xe_fxaa_source, uv + dir * 0.5, 0.0).rgb);
    float lumaB = dot(rgbB, kLumaWeights);
    result = (lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;
  }

  imageStore(xe_fxaa_dest, ivec2(pixel), vec4(result, 1.0));
}
)"
                         : R"(#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform XeApplyGammaRampConstants {
  uvec2 xe_fxaa_size;
  vec2 xe_fxaa_size_inv;
};

layout(set = 0, binding = 0) uniform sampler2D xe_fxaa_source;
layout(set = 1, binding = 0, rgb10_a2) writeonly uniform image2D xe_fxaa_dest;

const vec3 kLumaWeights = vec3(0.299, 0.587, 0.114);
const float kEdgeThreshold = 0.166;
const float kEdgeThresholdMin = 0.0833;
const float kSpanMax = 8.0;
const float kDirReduceMul = 0.125;
const float kDirReduceMin = 1.0 / 128.0;

float SampleLuma(vec2 uv) {
  return textureLod(xe_fxaa_source, uv, 0.0).a;
}

void main() {
  uvec2 pixel = gl_GlobalInvocationID.xy;
  if (any(greaterThanEqual(pixel, xe_fxaa_size))) {
    return;
  }

  vec2 uv = (vec2(pixel) + vec2(0.5)) * xe_fxaa_size_inv;
  vec2 texel = xe_fxaa_size_inv;
  vec4 rgbm = textureLod(xe_fxaa_source, uv, 0.0);

  float lumaM = rgbm.a;
  float lumaNW = SampleLuma(uv + vec2(-texel.x, -texel.y));
  float lumaNE = SampleLuma(uv + vec2( texel.x, -texel.y));
  float lumaSW = SampleLuma(uv + vec2(-texel.x,  texel.y));
  float lumaSE = SampleLuma(uv + vec2( texel.x,  texel.y));

  float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
  float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
  float lumaRange = lumaMax - lumaMin;

  vec3 result = rgbm.rgb;
  if (lumaRange >= max(kEdgeThresholdMin, lumaMax * kEdgeThreshold)) {
    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max(
        (lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * kDirReduceMul),
        kDirReduceMin);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2(-kSpanMax), vec2(kSpanMax)) * texel;

    vec3 rgbA =
        0.5 *
        (textureLod(xe_fxaa_source, uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb +
         textureLod(xe_fxaa_source, uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb);
    vec3 rgbB =
        rgbA * 0.5 +
        0.25 *
            (textureLod(xe_fxaa_source, uv + dir * -0.5, 0.0).rgb +
             textureLod(xe_fxaa_source, uv + dir * 0.5, 0.0).rgb);
    float lumaB = dot(rgbB, kLumaWeights);
    result = (lumaB < lumaMin || lumaB > lumaMax) ? rgbA : rgbB;
  }

  imageStore(xe_fxaa_dest, ivec2(pixel), vec4(result, 1.0));
}
)";
}

const char* GetSwapApplyGammaTablePixelRbSwapSource() {
  return R"(#version 450
layout(set = 0, binding = 0) uniform textureBuffer xe_apply_gamma_ramp;
layout(set = 1, binding = 0) uniform texture2D xe_apply_gamma_source;

layout(location = 0) out vec4 xe_apply_gamma_color;

void main() {
  ivec2 pixel = ivec2(gl_FragCoord.xy);
  uvec3 source = uvec3(texelFetch(xe_apply_gamma_source, pixel, 0).rgb * 255.0 + vec3(0.5)).bgr;
  xe_apply_gamma_color = vec4(
      texelFetch(xe_apply_gamma_ramp, int(source.r)).b,
      texelFetch(xe_apply_gamma_ramp, int(source.g)).g,
      texelFetch(xe_apply_gamma_ramp, int(source.b)).r,
      1.0);
}
)";
}

const char* GetSwapApplyGammaPwlPixelRbSwapSource() {
  return R"(#version 450
layout(set = 0, binding = 0) uniform utextureBuffer xe_apply_gamma_ramp;
layout(set = 1, binding = 0) uniform texture2D xe_apply_gamma_source;

layout(location = 0) out vec4 xe_apply_gamma_color;

float ApplyPwl(uint source_value_10b, uint channel) {
  uint source_value_base = source_value_10b >> 3u;
  uvec4 pwl = texelFetch(xe_apply_gamma_ramp, int(source_value_base * 3u + channel));
  float value = float(pwl.x) + float((source_value_10b & 7u) * pwl.y) * 0.125;
  return clamp(value * 1.52737048e-05, 0.0, 1.0);
}

void main() {
  ivec2 pixel = ivec2(gl_FragCoord.xy);
  uvec3 source = uvec3(texelFetch(xe_apply_gamma_source, pixel, 0).rgb * 1023.0 + vec3(0.5)).bgr;
  xe_apply_gamma_color = vec4(
      ApplyPwl(source.r, 0u),
      ApplyPwl(source.g, 1u),
      ApplyPwl(source.b, 2u),
      1.0);
}
)";
}

const char* GetSwapApplyGammaTableComputeRbSwapSource() {
  return R"(#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform XeApplyGammaRampConstants {
  uvec2 xe_apply_gamma_size;
};

layout(set = 0, binding = 0) uniform textureBuffer xe_apply_gamma_ramp;
layout(set = 1, binding = 0) uniform texture2D xe_apply_gamma_source;
layout(set = 2, binding = 0, rgb10_a2) writeonly uniform image2D xe_apply_gamma_dest;

void main() {
  ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(uvec2(pixel), xe_apply_gamma_size))) {
    return;
  }

  uvec3 source = uvec3(texelFetch(xe_apply_gamma_source, pixel, 0).rgb * 255.0 + vec3(0.5)).bgr;
  imageStore(
      xe_apply_gamma_dest, pixel,
      vec4(texelFetch(xe_apply_gamma_ramp, int(source.r)).b,
           texelFetch(xe_apply_gamma_ramp, int(source.g)).g,
           texelFetch(xe_apply_gamma_ramp, int(source.b)).r, 1.0));
}
)";
}

const char* GetSwapApplyGammaPwlComputeRbSwapSource() {
  return R"(#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform XeApplyGammaRampConstants {
  uvec2 xe_apply_gamma_size;
};

layout(set = 0, binding = 0) uniform utextureBuffer xe_apply_gamma_ramp;
layout(set = 1, binding = 0) uniform texture2D xe_apply_gamma_source;
layout(set = 2, binding = 0, rgb10_a2) writeonly uniform image2D xe_apply_gamma_dest;

float ApplyPwl(uint source_value_10b, uint channel) {
  uint source_value_base = source_value_10b >> 3u;
  uvec4 pwl = texelFetch(xe_apply_gamma_ramp, int(source_value_base * 3u + channel));
  float value = float(pwl.x) + float((source_value_10b & 7u) * pwl.y) * 0.125;
  return clamp(value * 1.52737048e-05, 0.0, 1.0);
}

void main() {
  ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(uvec2(pixel), xe_apply_gamma_size))) {
    return;
  }

  uvec3 source = uvec3(texelFetch(xe_apply_gamma_source, pixel, 0).rgb * 1023.0 + vec3(0.5)).bgr;
  imageStore(xe_apply_gamma_dest, pixel,
             vec4(ApplyPwl(source.r, 0u), ApplyPwl(source.g, 1u),
                  ApplyPwl(source.b, 2u), 1.0));
}
)";
}

const char* GetSwapApplyGammaTableFxaaLumaComputeRbSwapSource() {
  return R"(#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform XeApplyGammaRampConstants {
  uvec2 xe_apply_gamma_size;
};

layout(set = 0, binding = 0) uniform textureBuffer xe_apply_gamma_ramp;
layout(set = 1, binding = 0) uniform texture2D xe_apply_gamma_source;
layout(set = 2, binding = 0, rgba16f) writeonly uniform image2D xe_apply_gamma_dest;

const vec3 kLumaWeights = vec3(0.299, 0.587, 0.114);

void main() {
  ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(uvec2(pixel), xe_apply_gamma_size))) {
    return;
  }

  uvec3 source = uvec3(texelFetch(xe_apply_gamma_source, pixel, 0).rgb * 255.0 + vec3(0.5)).bgr;
  vec3 rgb = vec3(texelFetch(xe_apply_gamma_ramp, int(source.r)).b,
                  texelFetch(xe_apply_gamma_ramp, int(source.g)).g,
                  texelFetch(xe_apply_gamma_ramp, int(source.b)).r);
  imageStore(xe_apply_gamma_dest, pixel, vec4(rgb, dot(rgb, kLumaWeights)));
}
)";
}

const char* GetSwapApplyGammaPwlFxaaLumaComputeRbSwapSource() {
  return R"(#version 450
layout(local_size_x = 16, local_size_y = 8, local_size_z = 1) in;

layout(push_constant) uniform XeApplyGammaRampConstants {
  uvec2 xe_apply_gamma_size;
};

layout(set = 0, binding = 0) uniform utextureBuffer xe_apply_gamma_ramp;
layout(set = 1, binding = 0) uniform texture2D xe_apply_gamma_source;
layout(set = 2, binding = 0, rgba16f) writeonly uniform image2D xe_apply_gamma_dest;

const vec3 kLumaWeights = vec3(0.299, 0.587, 0.114);

float ApplyPwl(uint source_value_10b, uint channel) {
  uint source_value_base = source_value_10b >> 3u;
  uvec4 pwl = texelFetch(xe_apply_gamma_ramp, int(source_value_base * 3u + channel));
  float value = float(pwl.x) + float((source_value_10b & 7u) * pwl.y) * 0.125;
  return clamp(value * 1.52737048e-05, 0.0, 1.0);
}

void main() {
  ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
  if (any(greaterThanEqual(uvec2(pixel), xe_apply_gamma_size))) {
    return;
  }

  uvec3 source = uvec3(texelFetch(xe_apply_gamma_source, pixel, 0).rgb * 1023.0 + vec3(0.5)).bgr;
  vec3 rgb = vec3(ApplyPwl(source.r, 0u), ApplyPwl(source.g, 1u),
                  ApplyPwl(source.b, 2u));
  imageStore(xe_apply_gamma_dest, pixel, vec4(rgb, dot(rgb, kLumaWeights)));
}
)";
}

bool CompileGlslToSpirvInternal(EShLanguage stage, std::string_view source,
                                std::vector<uint32_t>& spirv_out, std::string& error_out) {
  static std::once_flag glslang_initialize_once;
  std::call_once(glslang_initialize_once, []() { glslang::InitializeProcess(); });

  const char* source_c_str = source.data();
  glslang::TShader shader(stage);
  shader.setStrings(&source_c_str, 1);
  shader.setEnvInput(glslang::EShSourceGlsl, stage, glslang::EShClientVulkan, 450);
  shader.setEnvClient(glslang::EShClientVulkan, glslang::EShTargetVulkan_1_0);
  shader.setEnvTarget(glslang::EShTargetSpv, glslang::EShTargetSpv_1_0);

  EShMessages messages = EShMessages(EShMsgSpvRules | EShMsgVulkanRules);
  if (!shader.parse(&kGlslangDefaultTBuiltInResource, 450, false, messages)) {
    error_out = "glslang shader parse failed";
    if (const char* shader_log = shader.getInfoLog();
        shader_log != nullptr && shader_log[0] != '\0') {
      error_out += ": ";
      error_out += shader_log;
    }
    return false;
  }

  glslang::TProgram program;
  program.addShader(&shader);
  if (!program.link(messages)) {
    error_out = "glslang program link failed";
    if (const char* program_log = program.getInfoLog();
        program_log != nullptr && program_log[0] != '\0') {
      error_out += ": ";
      error_out += program_log;
    }
    return false;
  }

  const glslang::TIntermediate* intermediate = program.getIntermediate(stage);
  if (intermediate == nullptr) {
    error_out = "glslang produced no stage intermediate";
    return false;
  }

  glslang::SpvOptions spv_options = {};
  spv_options.disableOptimizer = true;
  spv_options.optimizeSize = false;
  glslang::GlslangToSpv(*intermediate, spirv_out, &spv_options);
  if (spirv_out.empty()) {
    error_out = "glslang produced empty SPIR-V";
    return false;
  }
  return true;
}

}  // namespace

// Generated with `xb buildshaders`.
namespace shaders {
#include "../shaders/vulkan_spirv/apply_gamma_pwl_cs.h"
#include "../shaders/vulkan_spirv/apply_gamma_pwl_fxaa_luma_ps.h"
#include "../shaders/vulkan_spirv/apply_gamma_pwl_fxaa_luma_cs.h"
#include "../shaders/vulkan_spirv/apply_gamma_pwl_ps.h"
#include "../shaders/vulkan_spirv/apply_gamma_table_cs.h"
#include "../shaders/vulkan_spirv/apply_gamma_table_fxaa_luma_ps.h"
#include "../shaders/vulkan_spirv/apply_gamma_table_fxaa_luma_cs.h"
#include "../shaders/vulkan_spirv/apply_gamma_table_ps.h"
#include "../shaders/vulkan_spirv/fullscreen_cw_vs.h"
#include "../shaders/vulkan_spirv/resolve_downscale_cs.h"
}  // namespace shaders

const VkDescriptorPoolSize VulkanCommandProcessor::kDescriptorPoolSizeUniformBuffer = {
    VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
    SpirvShaderTranslator::kConstantBufferCount* kLinkedTypeDescriptorPoolSetCount};

const VkDescriptorPoolSize VulkanCommandProcessor::kDescriptorPoolSizeStorageBuffer = {
    VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 * kLinkedTypeDescriptorPoolSetCount};

// 2x descriptors for texture images because of unsigned and signed bindings.
const VkDescriptorPoolSize VulkanCommandProcessor::kDescriptorPoolSizeTextures[2] = {
    {VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 2 * kLinkedTypeDescriptorPoolSetCount},
    {VK_DESCRIPTOR_TYPE_SAMPLER, kLinkedTypeDescriptorPoolSetCount},
};

VulkanCommandProcessor::VulkanCommandProcessor(VulkanGraphicsSystem* graphics_system,
                                               system::KernelState* kernel_state)
    : CommandProcessor(graphics_system, kernel_state),
      deferred_command_buffer_(*this),
      prologue_deferred_command_buffer_(*this),
      transient_descriptor_allocator_uniform_buffer_(
          static_cast<const ui::vulkan::VulkanProvider*>(graphics_system->provider())
              ->vulkan_device(),
          &kDescriptorPoolSizeUniformBuffer, 1, kLinkedTypeDescriptorPoolSetCount),
      transient_descriptor_allocator_storage_buffer_(
          static_cast<const ui::vulkan::VulkanProvider*>(graphics_system->provider())
              ->vulkan_device(),
          &kDescriptorPoolSizeStorageBuffer, 1, kLinkedTypeDescriptorPoolSetCount),
      transient_descriptor_allocator_textures_(
          static_cast<const ui::vulkan::VulkanProvider*>(graphics_system->provider())
              ->vulkan_device(),
          kDescriptorPoolSizeTextures, uint32_t(rex::countof(kDescriptorPoolSizeTextures)),
          kLinkedTypeDescriptorPoolSetCount) {
  legacy_readback_memexport_cvar_name_ = "vulkan_readback_memexport";
}

VulkanCommandProcessor::~VulkanCommandProcessor() = default;

void VulkanCommandProcessor::ClearCaches() {
  CommandProcessor::ClearCaches();
  InvalidateAllVertexBufferResidency();
  cache_clear_requested_ = true;
}

void VulkanCommandProcessor::InvalidateGpuMemory() {
  if (shared_memory_) {
    shared_memory_->InvalidateAllPages();
  }
}

void VulkanCommandProcessor::InvalidateAllVertexBufferResidency() {
  vertex_buffers_in_sync_[0] = 0;
  vertex_buffers_in_sync_[1] = 0;
  for (VertexBufferState& state : vertex_buffer_states_) {
    state.address = UINT32_MAX;
    state.size = UINT32_MAX;
  }
}

void VulkanCommandProcessor::InvalidateVertexBufferResidency(uint32_t vfetch_index) {
  if (vfetch_index >= vertex_buffer_states_.size()) {
    return;
  }
  vertex_buffers_in_sync_[vfetch_index >> 6] &= ~(uint64_t(1) << (vfetch_index & 63));
}

void VulkanCommandProcessor::InvalidateVertexBufferResidencyRange(uint32_t first_vfetch,
                                                                  uint32_t last_vfetch) {
  if (first_vfetch > last_vfetch) {
    std::swap(first_vfetch, last_vfetch);
  }
  if (first_vfetch >= vertex_buffer_states_.size()) {
    return;
  }
  last_vfetch = std::min(last_vfetch, uint32_t(vertex_buffer_states_.size() - 1));
  for (uint32_t vfetch_index = first_vfetch; vfetch_index <= last_vfetch; ++vfetch_index) {
    InvalidateVertexBufferResidency(vfetch_index);
  }
}

void VulkanCommandProcessor::InitializeShaderStorage(const std::filesystem::path& cache_root,
                                                     uint32_t title_id, bool blocking) {
  CommandProcessor::InitializeShaderStorage(cache_root, title_id, blocking);
  pipeline_cache_->InitializeShaderStorage(cache_root, title_id, blocking);
}

bool VulkanCommandProcessor::ExecutePacketType3_EVENT_WRITE_ZPD(memory::RingBuffer* reader,
                                                                uint32_t packet, uint32_t count) {
  if (!REXCVAR_GET(occlusion_query_enable) || !occlusion_query_resources_available_) {
    return CommandProcessor::ExecutePacketType3_EVENT_WRITE_ZPD(reader, packet, count);
  }

  const uint32_t kQueryFinished = rex::byte_swap(0xFFFFFEED);
  assert_true(count == 1);
  uint32_t initiator = reader->ReadAndSwap<uint32_t>();
  WriteRegister(XE_GPU_REG_VGT_EVENT_INITIATOR, initiator & 0x3F);

  uint32_t sample_count_addr = register_file_->values[XE_GPU_REG_RB_SAMPLE_COUNT_ADDR];
  auto* sample_counts =
      memory_->TranslatePhysical<xenos::xe_gpu_depth_sample_counts*>(sample_count_addr);
  if (!sample_counts) {
    DisableHostOcclusionQueries();
    return true;
  }

  auto write_fallback_result = [sample_counts]() -> bool {
    auto fake_sample_count = REXCVAR_GET(query_occlusion_fake_sample_count);
    if (fake_sample_count < 0) {
      return true;
    }
    bool is_end_via_z_pass =
        sample_counts->ZPass_A == kQueryFinished || sample_counts->ZPass_B == kQueryFinished;
    bool is_end_via_z_fail =
        sample_counts->ZFail_A == kQueryFinished || sample_counts->ZFail_B == kQueryFinished;
    std::memset(sample_counts, 0, sizeof(xenos::xe_gpu_depth_sample_counts));
    if (is_end_via_z_pass || is_end_via_z_fail) {
      sample_counts->ZPass_A = fake_sample_count;
      sample_counts->Total_A = fake_sample_count;
    }
    return true;
  };

  bool is_end_via_z_pass =
      sample_counts->ZPass_A == kQueryFinished || sample_counts->ZPass_B == kQueryFinished;
  bool is_end_via_z_fail =
      sample_counts->ZFail_A == kQueryFinished || sample_counts->ZFail_B == kQueryFinished;
  bool is_end = is_end_via_z_pass || is_end_via_z_fail;

  if (!is_end) {
    if (active_occlusion_query_.valid &&
        active_occlusion_query_.sample_count_address != sample_count_addr) {
      DisableHostOcclusionQueries();
      return write_fallback_result();
    }
    if (!BeginGuestOcclusionQuery(sample_count_addr)) {
      return write_fallback_result();
    }
    return true;
  }

  if (!active_occlusion_query_.valid ||
      active_occlusion_query_.sample_count_address != sample_count_addr) {
    DisableHostOcclusionQueries();
    return write_fallback_result();
  }

  if (!EndGuestOcclusionQuery(sample_count_addr)) {
    return write_fallback_result();
  }
  return true;
}

std::string VulkanCommandProcessor::GetWindowTitleText() const {
  std::ostringstream title;
  title << "Vulkan";
  if (render_target_cache_) {
    switch (render_target_cache_->GetPath()) {
      case RenderTargetCache::Path::kHostRenderTargets:
        title << " - FBO";
        break;
      case RenderTargetCache::Path::kPixelShaderInterlock:
        title << " - FSI";
        break;
      default:
        break;
    }
    uint32_t draw_resolution_scale_x =
        texture_cache_ ? texture_cache_->draw_resolution_scale_x() : 1;
    uint32_t draw_resolution_scale_y =
        texture_cache_ ? texture_cache_->draw_resolution_scale_y() : 1;
    if (draw_resolution_scale_x > 1 || draw_resolution_scale_y > 1) {
      title << ' ' << draw_resolution_scale_x << 'x' << draw_resolution_scale_y;
    }
  }
  title << " - HEAVILY INCOMPLETE, early development";
  return title.str();
}

bool VulkanCommandProcessor::CompileGlslToSpirv(VkShaderStageFlagBits stage,
                                                std::string_view source,
                                                std::vector<uint32_t>& spirv_out,
                                                std::string& error_out) const {
  EShLanguage glslang_stage;
  switch (stage) {
    case VK_SHADER_STAGE_VERTEX_BIT:
      glslang_stage = EShLangVertex;
      break;
    case VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT:
      glslang_stage = EShLangTessControl;
      break;
    case VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT:
      glslang_stage = EShLangTessEvaluation;
      break;
    case VK_SHADER_STAGE_GEOMETRY_BIT:
      glslang_stage = EShLangGeometry;
      break;
    case VK_SHADER_STAGE_FRAGMENT_BIT:
      glslang_stage = EShLangFragment;
      break;
    case VK_SHADER_STAGE_COMPUTE_BIT:
      glslang_stage = EShLangCompute;
      break;
    default:
      error_out = fmt::format("Unsupported Vulkan shader stage mask {}", uint32_t(stage));
      return false;
  }
  return CompileGlslToSpirvInternal(glslang_stage, source, spirv_out, error_out);
}

void VulkanCommandProcessor::LogFragmentDiagnostic(std::string_view message) {
  const std::string line = fmt::format("{} session={}", message, diagnostic_storage_session_);
  REXGPU_WARN("{}", line);
  // Also retain the bounded evidence independently of logger levels/rotation.
  // stderr is captured by the runner even if the status path is not writable.
  std::fprintf(stderr, "%s\n", line.c_str());
  std::fflush(stderr);
  if (!diagnostics::AppendFragmentDiagnosticStatus(
          diagnostics::kFragmentDiagnosticStatusPath, line)) {
    REXGPU_ERROR("Fragment diagnostic persistence failed: path='{}'",
                 diagnostics::kFragmentDiagnosticStatusPath);
    std::fprintf(stderr, "Fragment diagnostic persistence failed: path='%s'\n",
                 diagnostics::kFragmentDiagnosticStatusPath);
    std::fflush(stderr);
  }
}

void VulkanCommandProcessor::LogFragmentDiagnosticOutcome(uint64_t frame, uint64_t vertex_hash,
                                                          uint64_t pixel_hash, uint32_t draw,
                                                          std::string_view status, uint32_t count,
                                                          bool overflow, std::string_view error) {
  LogFragmentDiagnostic(
      fmt::format("Fragment diagnostic capture: frame={} VS={:016X} PS={:016X} draw={} "
                  "status={} count={} overflow={} error={}",
                  frame, vertex_hash, pixel_hash, draw, status, count, overflow, error));
}

bool VulkanCommandProcessor::SetupContext() {
  if (!CommandProcessor::SetupContext()) {
    REXGPU_ERROR("Failed to initialize base command processor context");
    return false;
  }
  InvalidateAllVertexBufferResidency();

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const ui::vulkan::VulkanDevice::Properties& device_properties = vulkan_device->properties();

  diagnostics::RunStandaloneDerivativeProbeIfRequested();

  constexpr auto& kDiagnosticStorageEnablePath = diagnostics::kFragmentDiagnosticEnablePath;
  constexpr uint32_t kDiagnosticStorageCapacity = VulkanDiagnosticStorageBuffer::kMaxRecords;
  if (!diagnostic_storage_gate_latched_) {
    std::ifstream enable_file(kDiagnosticStorageEnablePath);
    diagnostic_storage_enabled_ = enable_file.good();
    if (diagnostic_storage_enabled_) {
      const std::string content((std::istreambuf_iterator<char>(enable_file)), {});
      diagnostic_storage_frame_ = diagnostics::ParseFragmentDiagnosticFrame(content);
      diagnostic_storage_shader_target_ =
          diagnostics::ParseFragmentDiagnosticShaderTarget(content);
      diagnostic_storage_overwrite_census_mode_ =
          diagnostics::ParseFragmentDiagnosticOverwriteCensusMode(content);
      diagnostic_storage_local_overwrite_census_mode_ =
          diagnostics::ParseFragmentDiagnosticLocalOverwriteCensusMode(content);
      diagnostic_storage_local_overwrite_tracking_started_ = false;
      diagnostic_storage_census_mode_ = diagnostics::ParseFragmentDiagnosticCensusMode(content) ||
                                        diagnostic_storage_overwrite_census_mode_ ||
                                        diagnostic_storage_local_overwrite_census_mode_;
      diagnostic_storage_local_candidate_triples_mode_ =
          diagnostics::ParseFragmentDiagnosticLocalCandidateTriplesMode(content);
      diagnostic_storage_local_candidate_output_mode_ =
          diagnostics::ParseFragmentDiagnosticLocalCandidateOutputMode(content);
      diagnostic_storage_ben_occlusion_mode_ =
          diagnostics::ParseBenOcclusionVertexOutputMode(content);
      diagnostic_storage_vertex_input_dump_mode_ =
          diagnostic_storage_ben_occlusion_mode_ ||
          diagnostics::ParseVertexInputDiagnosticMode(content) ||
          (diagnostic_storage_local_candidate_output_mode_ &&
           diagnostics::ParseCombinedLocalCandidateVertexOutputMode(content));
      diagnostic_storage_combined_vertex_output_mode_ =
          diagnostic_storage_local_candidate_output_mode_ &&
          diagnostics::ParseCombinedLocalCandidateVertexOutputMode(content);
      diagnostic_storage_tf14_filter_mode_ =
          diagnostics::ParseFragmentDiagnosticTf14FilterMode(content);
      diagnostic_storage_tf0_mip_filter_mode_ =
          diagnostics::ParseFragmentDiagnosticTf0MipFilterMode(content);
      diagnostic_storage_draw_log_ = diagnostics::FragmentDiagnosticDrawLog(
          diagnostic_storage_frame_, diagnostic_storage_overwrite_census_mode_ ||
                                          diagnostic_storage_local_overwrite_census_mode_ ||
                                          diagnostic_storage_local_candidate_output_mode_ ||
                                          diagnostic_storage_vertex_input_dump_mode_,
          diagnostics::GetFragmentDiagnosticPixelHash(diagnostic_storage_shader_target_));
    }
    diagnostic_storage_gate_latched_ = true;
    diagnostic_storage_session_ =
        uint64_t(std::chrono::system_clock::now().time_since_epoch().count());
    LogFragmentDiagnostic(fmt::format(
        "Fragment diagnostic gate: abi={} capacity={} enabled={} mode={} sampler_mode={} "
        "tf0_mip_mode={} path='{}' status_path='{}' target_frame={} expected_PS={:016X}",
        VulkanDiagnosticStorageBuffer::kAbiVersion, kDiagnosticStorageCapacity,
        diagnostic_storage_enabled_,
        diagnostic_storage_overwrite_census_mode_ ? "overwrite-census"
        : diagnostic_storage_local_overwrite_census_mode_ ? "local-overwrite-census"
        : diagnostic_storage_local_candidate_output_mode_ ? "local-candidate-output"
        : diagnostic_storage_local_candidate_triples_mode_ ? "local-candidate-triples"
        : diagnostic_storage_census_mode_                  ? "census"
                                                          : "triples",
        diagnostics::FragmentDiagnosticTf14FilterModeName(diagnostic_storage_tf14_filter_mode_),
        diagnostics::FragmentDiagnosticTf0MipFilterModeName(
            diagnostic_storage_tf0_mip_filter_mode_),
        kDiagnosticStorageEnablePath, diagnostics::kFragmentDiagnosticStatusPath,
        diagnostic_storage_frame_,
        diagnostics::GetFragmentDiagnosticPixelHash(diagnostic_storage_shader_target_)));
    if (diagnostic_storage_vertex_input_dump_mode_) {
      LogFragmentDiagnostic(fmt::format(
          "Vertex output diagnostic gate: ben_occlusion={} frames={},{} VS={:016X} PS={:016X} "
          "max_bytes={}",
          diagnostic_storage_ben_occlusion_mode_,
          diagnostics::kVertexInputDiagnosticFirstFrame,
          diagnostics::kVertexInputDiagnosticSecondFrame,
          diagnostics::kVertexInputDiagnosticVertexHash,
          diagnostics::kVertexInputDiagnosticPixelHash,
          diagnostics::kVertexInputDiagnosticMaxBytes));
    }
  }

  // The unconditional inclusion of the vertex shader stage also covers the case
  // of manual index / factor buffer fetch (the system constants and the shared
  // memory are needed for that) in the tessellation vertex shader when
  // fullDrawIndexUint32 is not supported.
  guest_shader_pipeline_stages_ =
      VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
  guest_shader_vertex_stages_ = VK_SHADER_STAGE_VERTEX_BIT;
  if (device_properties.tessellationShader) {
    guest_shader_pipeline_stages_ |= VK_PIPELINE_STAGE_TESSELLATION_CONTROL_SHADER_BIT |
                                     VK_PIPELINE_STAGE_TESSELLATION_EVALUATION_SHADER_BIT;
    guest_shader_vertex_stages_ |=
        VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT | VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT;
  }
  if (!device_properties.vertexPipelineStoresAndAtomics) {
    // For memory export from vertex shaders converted to compute shaders.
    guest_shader_pipeline_stages_ |= VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    guest_shader_vertex_stages_ |= VK_SHADER_STAGE_COMPUTE_BIT;
  }

  // 16384 is bigger than any single uniform buffer that Xenia needs, but is the
  // minimum maxUniformBufferRange, thus the safe minimum amount.
  uniform_buffer_pool_ = std::make_unique<ui::vulkan::VulkanUploadBufferPool>(
      vulkan_device, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
      rex::align(std::max(ui::GraphicsUploadBufferPool::kDefaultPageSize, size_t(16384)),
                 size_t(device_properties.minUniformBufferOffsetAlignment)));

  // Descriptor set layouts that don't depend on the setup of other subsystems.
  VkShaderStageFlags guest_shader_stages =
      guest_shader_vertex_stages_ | VK_SHADER_STAGE_FRAGMENT_BIT;
  // Empty.
  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
  descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptor_set_layout_create_info.pNext = nullptr;
  descriptor_set_layout_create_info.flags = 0;
  descriptor_set_layout_create_info.bindingCount = 0;
  descriptor_set_layout_create_info.pBindings = nullptr;
  if (dfn.vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create_info, nullptr,
                                      &descriptor_set_layout_empty_) != VK_SUCCESS) {
    REXGPU_ERROR("Failed to create an empty Vulkan descriptor set layout");
    return false;
  }
  // Guest draw constants.
  VkDescriptorSetLayoutBinding
      descriptor_set_layout_bindings_constants[SpirvShaderTranslator::kConstantBufferCount] = {};
  for (uint32_t i = 0; i < SpirvShaderTranslator::kConstantBufferCount; ++i) {
    VkDescriptorSetLayoutBinding& constants_binding = descriptor_set_layout_bindings_constants[i];
    constants_binding.binding = i;
    constants_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    constants_binding.descriptorCount = 1;
    constants_binding.pImmutableSamplers = nullptr;
  }
  descriptor_set_layout_bindings_constants[SpirvShaderTranslator::kConstantBufferSystem]
      .stageFlags =
      guest_shader_stages |
      (device_properties.tessellationShader ? VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT : 0) |
      (device_properties.geometryShader ? VK_SHADER_STAGE_GEOMETRY_BIT : 0);
  descriptor_set_layout_bindings_constants[SpirvShaderTranslator::kConstantBufferFloatVertex]
      .stageFlags = guest_shader_vertex_stages_;
  descriptor_set_layout_bindings_constants[SpirvShaderTranslator::kConstantBufferFloatPixel]
      .stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
  descriptor_set_layout_bindings_constants[SpirvShaderTranslator::kConstantBufferBoolLoop]
      .stageFlags = guest_shader_stages;
  descriptor_set_layout_bindings_constants[SpirvShaderTranslator::kConstantBufferFetch].stageFlags =
      guest_shader_stages;
  descriptor_set_layout_create_info.bindingCount =
      uint32_t(rex::countof(descriptor_set_layout_bindings_constants));
  descriptor_set_layout_create_info.pBindings = descriptor_set_layout_bindings_constants;
  if (dfn.vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create_info, nullptr,
                                      &descriptor_set_layout_constants_) != VK_SUCCESS) {
    REXGPU_ERROR(
        "Failed to create a Vulkan descriptor set layout for guest draw "
        "constant buffers");
    return false;
  }
  // Persistent ring buffer for Set 1 constants (Pillar 7).
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
          vulkan_device, kConstantRingBufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
          ui::vulkan::util::MemoryPurpose::kUpload, constant_ring_buffer_,
          constant_ring_buffer_memory_, &constant_ring_upload_memory_type_,
          &constant_ring_upload_memory_size_)) {
    REXGPU_ERROR("Failed to create the constant ring buffer");
    return false;
  }
  if (dfn.vkMapMemory(device, constant_ring_buffer_memory_, 0, VK_WHOLE_SIZE, 0,
                      reinterpret_cast<void**>(&constant_ring_mapping_)) != VK_SUCCESS) {
    REXGPU_ERROR("Failed to map the constant ring buffer");
    return false;
  }

  // Constant descriptor pool - exactly one set per frame in flight, each
  // written once here and never updated per draw again.
  VkDescriptorPoolSize pool_size_constants;
  pool_size_constants.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
  pool_size_constants.descriptorCount =
      SpirvShaderTranslator::kConstantBufferCount * kMaxFramesInFlight;
  VkDescriptorPoolCreateInfo pool_info_constants = {};
  pool_info_constants.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  pool_info_constants.maxSets = kMaxFramesInFlight;
  pool_info_constants.poolSizeCount = 1;
  pool_info_constants.pPoolSizes = &pool_size_constants;
  if (dfn.vkCreateDescriptorPool(device, &pool_info_constants, nullptr,
                                 &constant_descriptor_pool_) != VK_SUCCESS) {
    REXGPU_ERROR("Failed to create the constant descriptor pool");
    return false;
  }

  VkDescriptorSetAllocateInfo constant_descriptor_set_allocate_info = {};
  constant_descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  constant_descriptor_set_allocate_info.descriptorPool = constant_descriptor_pool_;
  constant_descriptor_set_allocate_info.descriptorSetCount = 1;
  constant_descriptor_set_allocate_info.pSetLayouts = &descriptor_set_layout_constants_;

  VkDescriptorBufferInfo constant_buffer_infos[SpirvShaderTranslator::kConstantBufferCount] = {};
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferSystem].buffer =
      constant_ring_buffer_;
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferSystem].offset = 0;
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferSystem].range =
      sizeof(SpirvShaderTranslator::SystemConstants);
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferFloatVertex].buffer =
      constant_ring_buffer_;
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferFloatVertex].offset = 0;
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferFloatVertex].range =
      sizeof(float) * 4 * 256;
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferFloatPixel].buffer =
      constant_ring_buffer_;
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferFloatPixel].offset = 0;
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferFloatPixel].range =
      sizeof(float) * 4 * 256;
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferBoolLoop].buffer =
      constant_ring_buffer_;
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferBoolLoop].offset = 0;
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferBoolLoop].range =
      sizeof(uint32_t) * (8 + 32);
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferFetch].buffer = constant_ring_buffer_;
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferFetch].offset = 0;
  constant_buffer_infos[SpirvShaderTranslator::kConstantBufferFetch].range =
      sizeof(uint32_t) * 6 * 32;

  VkWriteDescriptorSet constant_writes[SpirvShaderTranslator::kConstantBufferCount] = {};
  for (uint32_t b = 0; b < SpirvShaderTranslator::kConstantBufferCount; ++b) {
    constant_writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    constant_writes[b].dstBinding = b;
    constant_writes[b].descriptorCount = 1;
    constant_writes[b].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC;
    constant_writes[b].pBufferInfo = &constant_buffer_infos[b];
  }
  for (uint32_t f = 0; f < kMaxFramesInFlight; ++f) {
    if (dfn.vkAllocateDescriptorSets(device, &constant_descriptor_set_allocate_info,
                                     &constant_descriptor_sets_[f]) != VK_SUCCESS) {
      REXGPU_ERROR("Failed to allocate the constant descriptor sets");
      return false;
    }
    for (uint32_t b = 0; b < SpirvShaderTranslator::kConstantBufferCount; ++b) {
      constant_writes[b].dstSet = constant_descriptor_sets_[f];
    }
    dfn.vkUpdateDescriptorSets(device, SpirvShaderTranslator::kConstantBufferCount, constant_writes,
                               0, nullptr);
  }
  // Transient: storage buffer for compute shaders.
  VkDescriptorSetLayoutBinding descriptor_set_layout_binding_transient;
  descriptor_set_layout_binding_transient.binding = 0;
  descriptor_set_layout_binding_transient.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptor_set_layout_binding_transient.descriptorCount = 1;
  descriptor_set_layout_binding_transient.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  descriptor_set_layout_binding_transient.pImmutableSamplers = nullptr;
  descriptor_set_layout_create_info.bindingCount = 1;
  descriptor_set_layout_create_info.pBindings = &descriptor_set_layout_binding_transient;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &descriptor_set_layout_create_info, nullptr,
          &descriptor_set_layouts_single_transient_[size_t(
              SingleTransientDescriptorLayout::kStorageBufferCompute)]) != VK_SUCCESS) {
    REXGPU_ERROR(
        "Failed to create a Vulkan descriptor set layout for a storage buffer "
        "bound to the compute shader");
    return false;
  }
  // Transient: two storage buffers for compute shaders.
  VkDescriptorSetLayoutBinding descriptor_set_layout_bindings_transient_pair[2];
  descriptor_set_layout_bindings_transient_pair[0] = descriptor_set_layout_binding_transient;
  descriptor_set_layout_bindings_transient_pair[1] = descriptor_set_layout_binding_transient;
  descriptor_set_layout_bindings_transient_pair[1].binding = 1;
  descriptor_set_layout_create_info.bindingCount = 2;
  descriptor_set_layout_create_info.pBindings = descriptor_set_layout_bindings_transient_pair;
  if (dfn.vkCreateDescriptorSetLayout(
          device, &descriptor_set_layout_create_info, nullptr,
          &descriptor_set_layouts_single_transient_[size_t(
              SingleTransientDescriptorLayout::kStorageBufferPairCompute)]) != VK_SUCCESS) {
    REXGPU_ERROR(
        "Failed to create a Vulkan descriptor set layout for two storage "
        "buffers bound to the compute shader");
    return false;
  }

  shared_memory_ =
      std::make_unique<VulkanSharedMemory>(*this, *memory_, guest_shader_pipeline_stages_);
  if (!shared_memory_->Initialize()) {
    REXGPU_ERROR("Failed to initialize shared memory");
    return false;
  }

  primitive_processor_ =
      std::make_unique<VulkanPrimitiveProcessor>(*register_file_, *memory_, *shared_memory_, *this);
  if (!primitive_processor_->Initialize()) {
    REXGPU_ERROR("Failed to initialize the geometric primitive processor");
    return false;
  }

  uint32_t shared_memory_binding_count_log2 =
      SpirvShaderTranslator::GetSharedMemoryStorageBufferCountLog2(
          device_properties.maxStorageBufferRange);
  uint32_t shared_memory_binding_count = UINT32_C(1) << shared_memory_binding_count_log2;

  uint32_t draw_resolution_scale_x, draw_resolution_scale_y;
  bool draw_resolution_scale_not_clamped =
      TextureCache::GetConfigDrawResolutionScale(draw_resolution_scale_x, draw_resolution_scale_y);
  if (!draw_resolution_scale_not_clamped) {
    REXGPU_WARN(
        "The requested draw resolution scale is not supported by the "
        "emulator, reducing to {}x{}",
        draw_resolution_scale_x, draw_resolution_scale_y);
  }
  if (draw_resolution_scale_x > 1 || draw_resolution_scale_y > 1) {
    REXGPU_WARN(
        "Vulkan draw resolution scaling is experimental and may not affect all "
        "titles correctly");
  }
  if (!device_properties.fragmentStoresAndAtomics) {
    if (diagnostic_storage_enabled_) {
      LogFragmentDiagnosticOutcome(frame_current_, 0, 0, 0, "failed", 0, false,
                                    "fragment_stores_and_atomics_unsupported");
    }
    REXGPU_ERROR(
        diagnostic_storage_enabled_
            ? "Fragment diagnostic storage initialization requires Vulkan fragmentStoresAndAtomics, "
              "but the selected device does not support it"
            : "Vulkan fragmentStoresAndAtomics is required for GPU emulation and "
              "D3D12 parity, but unsupported by the selected device");
    return false;
  }
  if (diagnostic_storage_enabled_) {
    diagnostic_storage_buffer_ = std::make_unique<VulkanDiagnosticStorageBuffer>();
    if (!diagnostic_storage_buffer_->Initialize(vulkan_device, kDiagnosticStorageCapacity)) {
      diagnostic_storage_buffer_.reset();
      LogFragmentDiagnosticOutcome(frame_current_, 0, 0, 0, "failed", 0, false,
                                    "storage_initialization_failed");
      REXGPU_ERROR(
          "Failed to initialize enabled fragment diagnostic storage buffer "
          "(path='{}', capacity={} records)",
          kDiagnosticStorageEnablePath, kDiagnosticStorageCapacity);
      return false;
    }
  }
  if (!device_properties.vertexPipelineStoresAndAtomics) {
    REXGPU_ERROR(
        "Vulkan vertexPipelineStoresAndAtomics is required for GPU emulation and "
        "D3D12 parity, but unsupported by the selected device");
    return false;
  }
  if (!device_properties.geometryShader) {
    if (REXCVAR_GET(vulkan_require_geometry_shader)) {
      REXGPU_ERROR(
          "Vulkan geometryShader is required for GPU emulation "
          "(vulkan_require_geometry_shader=true), but unsupported by the "
          "selected device");
      return false;
    }
    REXGPU_WARN(
        "Vulkan geometryShader is not supported by the device; primitive "
        "fallback conversion/expansion paths will be used");
  }
  if (!device_properties.fillModeNonSolid) {
    if (REXCVAR_GET(vulkan_require_fill_mode_non_solid)) {
      REXGPU_ERROR(
          "Vulkan fillModeNonSolid is required for GPU emulation "
          "(vulkan_require_fill_mode_non_solid=true), but unsupported by the "
          "selected device");
      return false;
    }
    REXGPU_WARN(
        "Vulkan fillModeNonSolid is not supported by the device; line/point "
        "polygon modes will fall back to solid fill");
  }

  // Requires the transient descriptor set layouts.
  render_target_cache_ = std::make_unique<VulkanRenderTargetCache>(
      *register_file_, *memory_, draw_resolution_scale_x, draw_resolution_scale_y, *this);
  if (!render_target_cache_->Initialize(shared_memory_binding_count)) {
    REXGPU_ERROR("Failed to initialize the render target cache");
    return false;
  }

  // Shared memory and EDRAM descriptor set layout.
  bool edram_fragment_shader_interlock =
      render_target_cache_->GetPath() == RenderTargetCache::Path::kPixelShaderInterlock;
  VkDescriptorSetLayoutBinding shared_memory_and_edram_descriptor_set_layout_bindings[3];
  shared_memory_and_edram_descriptor_set_layout_bindings[0].binding = 0;
  shared_memory_and_edram_descriptor_set_layout_bindings[0].descriptorType =
      VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  shared_memory_and_edram_descriptor_set_layout_bindings[0].descriptorCount =
      shared_memory_binding_count;
  shared_memory_and_edram_descriptor_set_layout_bindings[0].stageFlags = guest_shader_stages;
  shared_memory_and_edram_descriptor_set_layout_bindings[0].pImmutableSamplers = nullptr;
  VkDescriptorSetLayoutCreateInfo shared_memory_and_edram_descriptor_set_layout_create_info;
  shared_memory_and_edram_descriptor_set_layout_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  shared_memory_and_edram_descriptor_set_layout_create_info.pNext = nullptr;
  shared_memory_and_edram_descriptor_set_layout_create_info.flags = 0;
  shared_memory_and_edram_descriptor_set_layout_create_info.pBindings =
      shared_memory_and_edram_descriptor_set_layout_bindings;
  if (edram_fragment_shader_interlock) {
    // EDRAM.
    shared_memory_and_edram_descriptor_set_layout_bindings[1].binding = 1;
    shared_memory_and_edram_descriptor_set_layout_bindings[1].descriptorType =
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    shared_memory_and_edram_descriptor_set_layout_bindings[1].descriptorCount = 1;
    shared_memory_and_edram_descriptor_set_layout_bindings[1].stageFlags =
        VK_SHADER_STAGE_FRAGMENT_BIT;
    shared_memory_and_edram_descriptor_set_layout_bindings[1].pImmutableSamplers = nullptr;
    shared_memory_and_edram_descriptor_set_layout_create_info.bindingCount = 2;
  } else {
    shared_memory_and_edram_descriptor_set_layout_create_info.bindingCount = 1;
  }
  if (diagnostic_storage_enabled_) {
    // Append at the next populated slot. The binding number remains 2, but
    // slot 1 is needed when EDRAM interlock (binding 1) is absent.
    VkDescriptorSetLayoutBinding& diagnostic_storage_binding =
        shared_memory_and_edram_descriptor_set_layout_bindings[
            shared_memory_and_edram_descriptor_set_layout_create_info.bindingCount];
    diagnostic_storage_binding.binding = 2;
    diagnostic_storage_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    diagnostic_storage_binding.descriptorCount = 1;
    diagnostic_storage_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    diagnostic_storage_binding.pImmutableSamplers = nullptr;
    ++shared_memory_and_edram_descriptor_set_layout_create_info.bindingCount;
  }
  if (dfn.vkCreateDescriptorSetLayout(
          device, &shared_memory_and_edram_descriptor_set_layout_create_info, nullptr,
          &descriptor_set_layout_shared_memory_and_edram_) != VK_SUCCESS) {
    REXGPU_ERROR(
        "Failed to create a Vulkan descriptor set layout for the shared memory "
        "and the EDRAM");
    return false;
  }

  pipeline_cache_ = std::make_unique<VulkanPipelineCache>(
      *this, *register_file_, *render_target_cache_, guest_shader_vertex_stages_);
  if (!pipeline_cache_->Initialize()) {
    REXGPU_ERROR("Failed to initialize the graphics pipeline cache");
    return false;
  }

  // Requires the transient descriptor set layouts.
  texture_cache_ =
      VulkanTextureCache::Create(*register_file_, *shared_memory_, draw_resolution_scale_x,
                                 draw_resolution_scale_y, *this, guest_shader_pipeline_stages_);
  if (!texture_cache_) {
    REXGPU_ERROR("Failed to initialize the texture cache");
    return false;
  }

  // Shared memory and EDRAM common bindings.
  VkDescriptorPoolSize descriptor_pool_sizes[1];
  descriptor_pool_sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  descriptor_pool_sizes[0].descriptorCount =
      shared_memory_binding_count + uint32_t(edram_fragment_shader_interlock) +
      uint32_t(diagnostic_storage_enabled_);
  VkDescriptorPoolCreateInfo descriptor_pool_create_info;
  descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  descriptor_pool_create_info.pNext = nullptr;
  descriptor_pool_create_info.flags = 0;
  descriptor_pool_create_info.maxSets = 1;
  descriptor_pool_create_info.poolSizeCount = 1;
  descriptor_pool_create_info.pPoolSizes = descriptor_pool_sizes;
  if (dfn.vkCreateDescriptorPool(device, &descriptor_pool_create_info, nullptr,
                                 &shared_memory_and_edram_descriptor_pool_) != VK_SUCCESS) {
    REXGPU_ERROR(
        "Failed to create the Vulkan descriptor pool for shared memory and "
        "EDRAM");
    return false;
  }
  VkDescriptorSetAllocateInfo descriptor_set_allocate_info;
  descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  descriptor_set_allocate_info.pNext = nullptr;
  descriptor_set_allocate_info.descriptorPool = shared_memory_and_edram_descriptor_pool_;
  descriptor_set_allocate_info.descriptorSetCount = 1;
  descriptor_set_allocate_info.pSetLayouts = &descriptor_set_layout_shared_memory_and_edram_;
  if (dfn.vkAllocateDescriptorSets(device, &descriptor_set_allocate_info,
                                   &shared_memory_and_edram_descriptor_set_) != VK_SUCCESS) {
    REXGPU_ERROR(
        "Failed to allocate the Vulkan descriptor set for shared memory and "
        "EDRAM");
    return false;
  }
  VkDescriptorBufferInfo
      shared_memory_descriptor_buffers_info[SharedMemory::kBufferSize / (128 << 20)];
  uint32_t shared_memory_binding_range =
      SharedMemory::kBufferSize >> shared_memory_binding_count_log2;
  for (uint32_t i = 0; i < shared_memory_binding_count; ++i) {
    VkDescriptorBufferInfo& shared_memory_descriptor_buffer_info =
        shared_memory_descriptor_buffers_info[i];
    shared_memory_descriptor_buffer_info.buffer = shared_memory_->buffer();
    shared_memory_descriptor_buffer_info.offset = shared_memory_binding_range * i;
    shared_memory_descriptor_buffer_info.range = shared_memory_binding_range;
  }
  VkWriteDescriptorSet write_descriptor_sets[3];
  VkWriteDescriptorSet& write_descriptor_set_shared_memory = write_descriptor_sets[0];
  write_descriptor_set_shared_memory.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  write_descriptor_set_shared_memory.pNext = nullptr;
  write_descriptor_set_shared_memory.dstSet = shared_memory_and_edram_descriptor_set_;
  write_descriptor_set_shared_memory.dstBinding = 0;
  write_descriptor_set_shared_memory.dstArrayElement = 0;
  write_descriptor_set_shared_memory.descriptorCount = shared_memory_binding_count;
  write_descriptor_set_shared_memory.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
  write_descriptor_set_shared_memory.pImageInfo = nullptr;
  write_descriptor_set_shared_memory.pBufferInfo = shared_memory_descriptor_buffers_info;
  write_descriptor_set_shared_memory.pTexelBufferView = nullptr;
  uint32_t descriptor_write_count = 1;
  VkDescriptorBufferInfo edram_descriptor_buffer_info;
  if (edram_fragment_shader_interlock) {
    edram_descriptor_buffer_info.buffer = render_target_cache_->edram_buffer();
    edram_descriptor_buffer_info.offset = 0;
    edram_descriptor_buffer_info.range = VK_WHOLE_SIZE;
    VkWriteDescriptorSet& write_descriptor_set_edram =
        write_descriptor_sets[descriptor_write_count++];
    write_descriptor_set_edram.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set_edram.pNext = nullptr;
    write_descriptor_set_edram.dstSet = shared_memory_and_edram_descriptor_set_;
    write_descriptor_set_edram.dstBinding = 1;
    write_descriptor_set_edram.dstArrayElement = 0;
    write_descriptor_set_edram.descriptorCount = 1;
    write_descriptor_set_edram.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set_edram.pImageInfo = nullptr;
    write_descriptor_set_edram.pBufferInfo = &edram_descriptor_buffer_info;
    write_descriptor_set_edram.pTexelBufferView = nullptr;
  }
  VkDescriptorBufferInfo diagnostic_storage_buffer_info;
  if (diagnostic_storage_enabled_) {
    assert_true(diagnostic_storage_buffer_ && diagnostic_storage_buffer_->buffer() != VK_NULL_HANDLE);
    diagnostic_storage_buffer_info.buffer = diagnostic_storage_buffer_->buffer();
    diagnostic_storage_buffer_info.offset = 0;
    diagnostic_storage_buffer_info.range = diagnostic_storage_buffer_->size();
    VkWriteDescriptorSet& write_descriptor_set_diagnostic_storage =
        write_descriptor_sets[descriptor_write_count++];
    write_descriptor_set_diagnostic_storage.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set_diagnostic_storage.pNext = nullptr;
    write_descriptor_set_diagnostic_storage.dstSet = shared_memory_and_edram_descriptor_set_;
    write_descriptor_set_diagnostic_storage.dstBinding = 2;
    write_descriptor_set_diagnostic_storage.dstArrayElement = 0;
    write_descriptor_set_diagnostic_storage.descriptorCount = 1;
    write_descriptor_set_diagnostic_storage.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set_diagnostic_storage.pImageInfo = nullptr;
    write_descriptor_set_diagnostic_storage.pBufferInfo = &diagnostic_storage_buffer_info;
    write_descriptor_set_diagnostic_storage.pTexelBufferView = nullptr;
  }
  dfn.vkUpdateDescriptorSets(device, descriptor_write_count, write_descriptor_sets, 0, nullptr);

  // Swap objects.

  // Gamma ramp, either device-local and host-visible at once, or separate
  // device-local texel buffer and host-visible upload buffer.
  gamma_ramp_256_entry_table_current_frame_ = UINT32_MAX;
  gamma_ramp_pwl_current_frame_ = UINT32_MAX;
  // Try to create a device-local host-visible buffer first, to skip copying.
  constexpr uint32_t kGammaRampSize256EntryTable = sizeof(uint32_t) * 256;
  constexpr uint32_t kGammaRampSizePWL = sizeof(uint16_t) * 2 * 3 * 128;
  constexpr uint32_t kGammaRampSize = kGammaRampSize256EntryTable + kGammaRampSizePWL;
  VkBufferCreateInfo gamma_ramp_host_visible_buffer_create_info;
  gamma_ramp_host_visible_buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  gamma_ramp_host_visible_buffer_create_info.pNext = nullptr;
  gamma_ramp_host_visible_buffer_create_info.flags = 0;
  gamma_ramp_host_visible_buffer_create_info.size = kGammaRampSize * kMaxFramesInFlight;
  gamma_ramp_host_visible_buffer_create_info.usage = VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT;
  gamma_ramp_host_visible_buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  gamma_ramp_host_visible_buffer_create_info.queueFamilyIndexCount = 0;
  gamma_ramp_host_visible_buffer_create_info.pQueueFamilyIndices = nullptr;
  if (dfn.vkCreateBuffer(device, &gamma_ramp_host_visible_buffer_create_info, nullptr,
                         &gamma_ramp_buffer_) == VK_SUCCESS) {
    bool use_gamma_ramp_host_visible_buffer = false;
    VkMemoryRequirements gamma_ramp_host_visible_buffer_memory_requirements;
    dfn.vkGetBufferMemoryRequirements(device, gamma_ramp_buffer_,
                                      &gamma_ramp_host_visible_buffer_memory_requirements);
    uint32_t gamma_ramp_host_visible_buffer_memory_types =
        gamma_ramp_host_visible_buffer_memory_requirements.memoryTypeBits &
        (vulkan_device->memory_types().device_local & vulkan_device->memory_types().host_visible);
    VkMemoryAllocateInfo gamma_ramp_host_visible_buffer_memory_allocate_info;
    // Prefer a host-uncached (because it's write-only) memory type, but try a
    // host-cached host-visible device-local one as well.
    if (rex::bit_scan_forward(
            gamma_ramp_host_visible_buffer_memory_types &
                ~vulkan_device->memory_types().host_cached,
            &(gamma_ramp_host_visible_buffer_memory_allocate_info.memoryTypeIndex)) ||
        rex::bit_scan_forward(
            gamma_ramp_host_visible_buffer_memory_types,
            &(gamma_ramp_host_visible_buffer_memory_allocate_info.memoryTypeIndex))) {
      VkMemoryAllocateInfo* gamma_ramp_host_visible_buffer_memory_allocate_info_last =
          &gamma_ramp_host_visible_buffer_memory_allocate_info;
      gamma_ramp_host_visible_buffer_memory_allocate_info.sType =
          VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      gamma_ramp_host_visible_buffer_memory_allocate_info.pNext = nullptr;
      gamma_ramp_host_visible_buffer_memory_allocate_info.allocationSize =
          gamma_ramp_host_visible_buffer_memory_requirements.size;
      VkMemoryDedicatedAllocateInfo gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info;
      if (vulkan_device->extensions().ext_1_1_KHR_dedicated_allocation) {
        gamma_ramp_host_visible_buffer_memory_allocate_info_last->pNext =
            &gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info;
        gamma_ramp_host_visible_buffer_memory_allocate_info_last =
            reinterpret_cast<VkMemoryAllocateInfo*>(
                &gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info);
        gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info.sType =
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info.pNext = nullptr;
        gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info.image = VK_NULL_HANDLE;
        gamma_ramp_host_visible_buffer_memory_dedicated_allocate_info.buffer = gamma_ramp_buffer_;
      }
      if (dfn.vkAllocateMemory(device, &gamma_ramp_host_visible_buffer_memory_allocate_info,
                               nullptr, &gamma_ramp_buffer_memory_) == VK_SUCCESS) {
        if (dfn.vkBindBufferMemory(device, gamma_ramp_buffer_, gamma_ramp_buffer_memory_, 0) ==
            VK_SUCCESS) {
          if (dfn.vkMapMemory(device, gamma_ramp_buffer_memory_, 0, VK_WHOLE_SIZE, 0,
                              &gamma_ramp_upload_mapping_) == VK_SUCCESS) {
            use_gamma_ramp_host_visible_buffer = true;
            gamma_ramp_upload_memory_size_ =
                gamma_ramp_host_visible_buffer_memory_allocate_info.allocationSize;
            gamma_ramp_upload_memory_type_ =
                gamma_ramp_host_visible_buffer_memory_allocate_info.memoryTypeIndex;
          }
        }
        if (!use_gamma_ramp_host_visible_buffer) {
          dfn.vkFreeMemory(device, gamma_ramp_buffer_memory_, nullptr);
          gamma_ramp_buffer_memory_ = VK_NULL_HANDLE;
        }
      }
    }
    if (!use_gamma_ramp_host_visible_buffer) {
      dfn.vkDestroyBuffer(device, gamma_ramp_buffer_, nullptr);
      gamma_ramp_buffer_ = VK_NULL_HANDLE;
    }
  }
  if (gamma_ramp_buffer_ == VK_NULL_HANDLE) {
    // Create separate buffers for the shader and uploading.
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
            vulkan_device, kGammaRampSize,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_TEXEL_BUFFER_BIT,
            ui::vulkan::util::MemoryPurpose::kDeviceLocal, gamma_ramp_buffer_,
            gamma_ramp_buffer_memory_)) {
      REXGPU_ERROR("Failed to create the gamma ramp buffer");
      return false;
    }
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
            vulkan_device, kGammaRampSize * kMaxFramesInFlight, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            ui::vulkan::util::MemoryPurpose::kUpload, gamma_ramp_upload_buffer_,
            gamma_ramp_upload_buffer_memory_, &gamma_ramp_upload_memory_type_,
            &gamma_ramp_upload_memory_size_)) {
      REXGPU_ERROR("Failed to create the gamma ramp upload buffer");
      return false;
    }
    if (dfn.vkMapMemory(device, gamma_ramp_upload_buffer_memory_, 0, VK_WHOLE_SIZE, 0,
                        &gamma_ramp_upload_mapping_) != VK_SUCCESS) {
      REXGPU_ERROR("Failed to map the gamma ramp upload buffer");
      return false;
    }
  }

  // Gamma ramp buffer views.
  uint32_t gamma_ramp_frame_count =
      gamma_ramp_upload_buffer_ == VK_NULL_HANDLE ? kMaxFramesInFlight : 1;
  VkBufferViewCreateInfo gamma_ramp_buffer_view_create_info;
  gamma_ramp_buffer_view_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_VIEW_CREATE_INFO;
  gamma_ramp_buffer_view_create_info.pNext = nullptr;
  gamma_ramp_buffer_view_create_info.flags = 0;
  gamma_ramp_buffer_view_create_info.buffer = gamma_ramp_buffer_;
  // 256-entry table.
  gamma_ramp_buffer_view_create_info.format = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
  gamma_ramp_buffer_view_create_info.range = kGammaRampSize256EntryTable;
  for (uint32_t i = 0; i < gamma_ramp_frame_count; ++i) {
    gamma_ramp_buffer_view_create_info.offset = kGammaRampSize * i;
    if (dfn.vkCreateBufferView(device, &gamma_ramp_buffer_view_create_info, nullptr,
                               &gamma_ramp_buffer_views_[i * 2]) != VK_SUCCESS) {
      REXGPU_ERROR("Failed to create a 256-entry table gamma ramp buffer view");
      return false;
    }
  }
  // Piecewise linear.
  gamma_ramp_buffer_view_create_info.format = VK_FORMAT_R16G16_UINT;
  gamma_ramp_buffer_view_create_info.range = kGammaRampSizePWL;
  for (uint32_t i = 0; i < gamma_ramp_frame_count; ++i) {
    gamma_ramp_buffer_view_create_info.offset = kGammaRampSize * i + kGammaRampSize256EntryTable;
    if (dfn.vkCreateBufferView(device, &gamma_ramp_buffer_view_create_info, nullptr,
                               &gamma_ramp_buffer_views_[i * 2 + 1]) != VK_SUCCESS) {
      REXGPU_ERROR("Failed to create a PWL gamma ramp buffer view");
      return false;
    }
  }

  // Swap descriptor set layouts.
  VkDescriptorSetLayoutBinding swap_descriptor_set_layout_binding;
  swap_descriptor_set_layout_binding.binding = 0;
  swap_descriptor_set_layout_binding.descriptorCount = 1;
  swap_descriptor_set_layout_binding.stageFlags =
      VK_SHADER_STAGE_FRAGMENT_BIT | VK_SHADER_STAGE_COMPUTE_BIT;
  swap_descriptor_set_layout_binding.pImmutableSamplers = nullptr;
  VkDescriptorSetLayoutCreateInfo swap_descriptor_set_layout_create_info;
  swap_descriptor_set_layout_create_info.sType =
      VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  swap_descriptor_set_layout_create_info.pNext = nullptr;
  swap_descriptor_set_layout_create_info.flags = 0;
  swap_descriptor_set_layout_create_info.bindingCount = 1;
  swap_descriptor_set_layout_create_info.pBindings = &swap_descriptor_set_layout_binding;
  swap_descriptor_set_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
  if (dfn.vkCreateDescriptorSetLayout(device, &swap_descriptor_set_layout_create_info, nullptr,
                                      &swap_descriptor_set_layout_sampled_image_) != VK_SUCCESS) {
    REXGPU_ERROR(
        "Failed to create the presentation sampled image descriptor set "
        "layout");
    return false;
  }
  swap_descriptor_set_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
  if (dfn.vkCreateDescriptorSetLayout(device, &swap_descriptor_set_layout_create_info, nullptr,
                                      &swap_descriptor_set_layout_uniform_texel_buffer_) !=
      VK_SUCCESS) {
    REXGPU_ERROR(
        "Failed to create the presentation uniform texel buffer descriptor set "
        "layout");
    return false;
  }
  swap_descriptor_set_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
  swap_descriptor_set_layout_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  if (dfn.vkCreateDescriptorSetLayout(device, &swap_descriptor_set_layout_create_info, nullptr,
                                      &swap_descriptor_set_layout_combined_image_sampler_) !=
      VK_SUCCESS) {
    REXGPU_ERROR(
        "Failed to create the presentation combined image sampler descriptor "
        "set layout");
    return false;
  }
  swap_descriptor_set_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
  if (dfn.vkCreateDescriptorSetLayout(device, &swap_descriptor_set_layout_create_info, nullptr,
                                      &swap_descriptor_set_layout_storage_image_) != VK_SUCCESS) {
    REXGPU_ERROR(
        "Failed to create the presentation storage image descriptor set "
        "layout");
    return false;
  }

  // Swap descriptor pool.
  std::array<VkDescriptorPoolSize, 4> swap_descriptor_pool_sizes;
  VkDescriptorPoolCreateInfo swap_descriptor_pool_create_info;
  swap_descriptor_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
  swap_descriptor_pool_create_info.pNext = nullptr;
  swap_descriptor_pool_create_info.flags = 0;
  swap_descriptor_pool_create_info.maxSets = 0;
  swap_descriptor_pool_create_info.poolSizeCount = 0;
  swap_descriptor_pool_create_info.pPoolSizes = swap_descriptor_pool_sizes.data();
  {
    VkDescriptorPoolSize& swap_descriptor_pool_size_sampled_image =
        swap_descriptor_pool_sizes[swap_descriptor_pool_create_info.poolSizeCount++];
    swap_descriptor_pool_size_sampled_image.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    // Source images.
    swap_descriptor_pool_size_sampled_image.descriptorCount = kMaxFramesInFlight;
    swap_descriptor_pool_create_info.maxSets += kMaxFramesInFlight;
  }
  {
    VkDescriptorPoolSize& swap_descriptor_pool_size_combined_image_sampler =
        swap_descriptor_pool_sizes[swap_descriptor_pool_create_info.poolSizeCount++];
    swap_descriptor_pool_size_combined_image_sampler.type =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    swap_descriptor_pool_size_combined_image_sampler.descriptorCount = kMaxFramesInFlight;
    swap_descriptor_pool_create_info.maxSets += kMaxFramesInFlight;
  }
  {
    VkDescriptorPoolSize& swap_descriptor_pool_size_storage_image =
        swap_descriptor_pool_sizes[swap_descriptor_pool_create_info.poolSizeCount++];
    swap_descriptor_pool_size_storage_image.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    swap_descriptor_pool_size_storage_image.descriptorCount = kMaxFramesInFlight * 2;
    swap_descriptor_pool_create_info.maxSets += kMaxFramesInFlight * 2;
  }
  // 256-entry table and PWL gamma ramps. If the gamma ramp buffer is
  // host-visible, for multiple frames.
  uint32_t gamma_ramp_buffer_view_count = 2 * gamma_ramp_frame_count;
  {
    VkDescriptorPoolSize& swap_descriptor_pool_size_uniform_texel_buffer =
        swap_descriptor_pool_sizes[swap_descriptor_pool_create_info.poolSizeCount++];
    swap_descriptor_pool_size_uniform_texel_buffer.type = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
    swap_descriptor_pool_size_uniform_texel_buffer.descriptorCount = gamma_ramp_buffer_view_count;
    swap_descriptor_pool_create_info.maxSets += gamma_ramp_buffer_view_count;
  }
  if (dfn.vkCreateDescriptorPool(device, &swap_descriptor_pool_create_info, nullptr,
                                 &swap_descriptor_pool_) != VK_SUCCESS) {
    REXGPU_ERROR("Failed to create the presentation descriptor pool");
    return false;
  }

  // Swap descriptor set allocation.
  VkDescriptorSetAllocateInfo swap_descriptor_set_allocate_info;
  swap_descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
  swap_descriptor_set_allocate_info.pNext = nullptr;
  swap_descriptor_set_allocate_info.descriptorPool = swap_descriptor_pool_;
  swap_descriptor_set_allocate_info.descriptorSetCount = 1;
  swap_descriptor_set_allocate_info.pSetLayouts = &swap_descriptor_set_layout_uniform_texel_buffer_;
  for (uint32_t i = 0; i < gamma_ramp_buffer_view_count; ++i) {
    if (dfn.vkAllocateDescriptorSets(device, &swap_descriptor_set_allocate_info,
                                     &swap_descriptors_gamma_ramp_[i]) != VK_SUCCESS) {
      REXGPU_ERROR("Failed to allocate the gamma ramp descriptor sets");
      return false;
    }
  }
  swap_descriptor_set_allocate_info.pSetLayouts = &swap_descriptor_set_layout_sampled_image_;
  for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
    if (dfn.vkAllocateDescriptorSets(device, &swap_descriptor_set_allocate_info,
                                     &swap_descriptors_source_[i]) != VK_SUCCESS) {
      REXGPU_ERROR("Failed to allocate the presentation source image descriptor sets");
      return false;
    }
  }
  swap_descriptor_set_allocate_info.pSetLayouts =
      &swap_descriptor_set_layout_combined_image_sampler_;
  for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
    if (dfn.vkAllocateDescriptorSets(device, &swap_descriptor_set_allocate_info,
                                     &swap_descriptors_fxaa_source_[i]) != VK_SUCCESS) {
      REXGPU_ERROR(
          "Failed to allocate the presentation FXAA source image descriptor "
          "sets");
      return false;
    }
  }
  swap_descriptor_set_allocate_info.pSetLayouts = &swap_descriptor_set_layout_storage_image_;
  for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
    if (dfn.vkAllocateDescriptorSets(device, &swap_descriptor_set_allocate_info,
                                     &swap_descriptors_destination_storage_[i]) != VK_SUCCESS) {
      REXGPU_ERROR(
          "Failed to allocate the presentation destination storage image "
          "descriptor sets");
      return false;
    }
  }
  for (uint32_t i = 0; i < kMaxFramesInFlight; ++i) {
    if (dfn.vkAllocateDescriptorSets(device, &swap_descriptor_set_allocate_info,
                                     &swap_descriptors_fxaa_destination_storage_[i]) !=
        VK_SUCCESS) {
      REXGPU_ERROR(
          "Failed to allocate the presentation FXAA destination storage image "
          "descriptor sets");
      return false;
    }
  }

  // Gamma ramp descriptor sets.
  VkWriteDescriptorSet gamma_ramp_write_descriptor_set;
  gamma_ramp_write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
  gamma_ramp_write_descriptor_set.pNext = nullptr;
  gamma_ramp_write_descriptor_set.dstBinding = 0;
  gamma_ramp_write_descriptor_set.dstArrayElement = 0;
  gamma_ramp_write_descriptor_set.descriptorCount = 1;
  gamma_ramp_write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER;
  gamma_ramp_write_descriptor_set.pImageInfo = nullptr;
  gamma_ramp_write_descriptor_set.pBufferInfo = nullptr;
  for (uint32_t i = 0; i < gamma_ramp_buffer_view_count; ++i) {
    gamma_ramp_write_descriptor_set.dstSet = swap_descriptors_gamma_ramp_[i];
    gamma_ramp_write_descriptor_set.pTexelBufferView = &gamma_ramp_buffer_views_[i];
    dfn.vkUpdateDescriptorSets(device, 1, &gamma_ramp_write_descriptor_set, 0, nullptr);
  }

  // Linear sampler for FXAA.
  VkSamplerCreateInfo swap_sampler_create_info;
  swap_sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  swap_sampler_create_info.pNext = nullptr;
  swap_sampler_create_info.flags = 0;
  swap_sampler_create_info.magFilter = VK_FILTER_LINEAR;
  swap_sampler_create_info.minFilter = VK_FILTER_LINEAR;
  swap_sampler_create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
  swap_sampler_create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  swap_sampler_create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  swap_sampler_create_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
  swap_sampler_create_info.mipLodBias = 0.0f;
  swap_sampler_create_info.anisotropyEnable = VK_FALSE;
  swap_sampler_create_info.maxAnisotropy = 1.0f;
  swap_sampler_create_info.compareEnable = VK_FALSE;
  swap_sampler_create_info.compareOp = VK_COMPARE_OP_NEVER;
  swap_sampler_create_info.minLod = 0.0f;
  swap_sampler_create_info.maxLod = 0.0f;
  swap_sampler_create_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
  swap_sampler_create_info.unnormalizedCoordinates = VK_FALSE;
  if (dfn.vkCreateSampler(device, &swap_sampler_create_info, nullptr,
                          &swap_sampler_linear_clamp_) != VK_SUCCESS) {
    REXGPU_ERROR("Failed to create the presentation FXAA sampler");
    return false;
  }

  // Gamma ramp application pipeline layout.
  std::array<VkDescriptorSetLayout, kSwapApplyGammaDescriptorSetCount>
      swap_apply_gamma_descriptor_set_layouts{};
  swap_apply_gamma_descriptor_set_layouts[kSwapApplyGammaDescriptorSetRamp] =
      swap_descriptor_set_layout_uniform_texel_buffer_;
  swap_apply_gamma_descriptor_set_layouts[kSwapApplyGammaDescriptorSetSource] =
      swap_descriptor_set_layout_sampled_image_;
  VkPipelineLayoutCreateInfo swap_apply_gamma_pipeline_layout_create_info;
  swap_apply_gamma_pipeline_layout_create_info.sType =
      VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  swap_apply_gamma_pipeline_layout_create_info.pNext = nullptr;
  swap_apply_gamma_pipeline_layout_create_info.flags = 0;
  swap_apply_gamma_pipeline_layout_create_info.setLayoutCount =
      uint32_t(swap_apply_gamma_descriptor_set_layouts.size());
  swap_apply_gamma_pipeline_layout_create_info.pSetLayouts =
      swap_apply_gamma_descriptor_set_layouts.data();
  swap_apply_gamma_pipeline_layout_create_info.pushConstantRangeCount = 0;
  swap_apply_gamma_pipeline_layout_create_info.pPushConstantRanges = nullptr;
  if (dfn.vkCreatePipelineLayout(device, &swap_apply_gamma_pipeline_layout_create_info, nullptr,
                                 &swap_apply_gamma_pipeline_layout_) != VK_SUCCESS) {
    REXGPU_ERROR("Failed to create the gamma ramp application pipeline layout");
    return false;
  }

  // Gamma ramp application compute pipeline layout.
  std::array<VkDescriptorSetLayout, kSwapApplyGammaComputeDescriptorSetCount>
      swap_apply_gamma_compute_descriptor_set_layouts{};
  swap_apply_gamma_compute_descriptor_set_layouts[kSwapApplyGammaComputeDescriptorSetRamp] =
      swap_descriptor_set_layout_uniform_texel_buffer_;
  swap_apply_gamma_compute_descriptor_set_layouts[kSwapApplyGammaComputeDescriptorSetSource] =
      swap_descriptor_set_layout_sampled_image_;
  swap_apply_gamma_compute_descriptor_set_layouts[kSwapApplyGammaComputeDescriptorSetDestination] =
      swap_descriptor_set_layout_storage_image_;
  VkPushConstantRange swap_apply_gamma_compute_push_constant_range;
  swap_apply_gamma_compute_push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  swap_apply_gamma_compute_push_constant_range.offset = 0;
  swap_apply_gamma_compute_push_constant_range.size = sizeof(SwapApplyGammaConstants);
  swap_apply_gamma_pipeline_layout_create_info.setLayoutCount =
      uint32_t(swap_apply_gamma_compute_descriptor_set_layouts.size());
  swap_apply_gamma_pipeline_layout_create_info.pSetLayouts =
      swap_apply_gamma_compute_descriptor_set_layouts.data();
  swap_apply_gamma_pipeline_layout_create_info.pushConstantRangeCount = 1;
  swap_apply_gamma_pipeline_layout_create_info.pPushConstantRanges =
      &swap_apply_gamma_compute_push_constant_range;
  if (dfn.vkCreatePipelineLayout(device, &swap_apply_gamma_pipeline_layout_create_info, nullptr,
                                 &swap_apply_gamma_compute_pipeline_layout_) != VK_SUCCESS) {
    REXGPU_ERROR("Failed to create the gamma ramp application compute pipeline layout");
    return false;
  }

  // FXAA compute pipeline layout.
  std::array<VkDescriptorSetLayout, kSwapFxaaDescriptorSetCount> swap_fxaa_descriptor_set_layouts{};
  swap_fxaa_descriptor_set_layouts[kSwapFxaaDescriptorSetSource] =
      swap_descriptor_set_layout_combined_image_sampler_;
  swap_fxaa_descriptor_set_layouts[kSwapFxaaDescriptorSetDestination] =
      swap_descriptor_set_layout_storage_image_;
  VkPushConstantRange swap_fxaa_push_constant_range;
  swap_fxaa_push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  swap_fxaa_push_constant_range.offset = 0;
  swap_fxaa_push_constant_range.size = sizeof(SwapFxaaConstants);
  swap_apply_gamma_pipeline_layout_create_info.setLayoutCount =
      uint32_t(swap_fxaa_descriptor_set_layouts.size());
  swap_apply_gamma_pipeline_layout_create_info.pSetLayouts =
      swap_fxaa_descriptor_set_layouts.data();
  swap_apply_gamma_pipeline_layout_create_info.pushConstantRangeCount = 1;
  swap_apply_gamma_pipeline_layout_create_info.pPushConstantRanges = &swap_fxaa_push_constant_range;
  if (dfn.vkCreatePipelineLayout(device, &swap_apply_gamma_pipeline_layout_create_info, nullptr,
                                 &swap_fxaa_pipeline_layout_) != VK_SUCCESS) {
    REXGPU_ERROR("Failed to create the FXAA compute pipeline layout");
    return false;
  }

  // Gamma application render pass. Doesn't make assumptions about outer usage
  // (explicit barriers must be used instead) for simplicity of use in different
  // scenarios with different pipelines.
  VkAttachmentDescription swap_apply_gamma_render_pass_attachment;
  swap_apply_gamma_render_pass_attachment.flags = 0;
  swap_apply_gamma_render_pass_attachment.format = ui::vulkan::VulkanPresenter::kGuestOutputFormat;
  swap_apply_gamma_render_pass_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
  swap_apply_gamma_render_pass_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  swap_apply_gamma_render_pass_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
  swap_apply_gamma_render_pass_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
  swap_apply_gamma_render_pass_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
  swap_apply_gamma_render_pass_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  swap_apply_gamma_render_pass_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkAttachmentReference swap_apply_gamma_render_pass_color_attachment;
  swap_apply_gamma_render_pass_color_attachment.attachment = 0;
  swap_apply_gamma_render_pass_color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
  VkSubpassDescription swap_apply_gamma_render_pass_subpass = {};
  swap_apply_gamma_render_pass_subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
  swap_apply_gamma_render_pass_subpass.colorAttachmentCount = 1;
  swap_apply_gamma_render_pass_subpass.pColorAttachments =
      &swap_apply_gamma_render_pass_color_attachment;
  VkSubpassDependency swap_apply_gamma_render_pass_dependencies[2];
  for (uint32_t i = 0; i < 2; ++i) {
    VkSubpassDependency& swap_apply_gamma_render_pass_dependency =
        swap_apply_gamma_render_pass_dependencies[i];
    swap_apply_gamma_render_pass_dependency.srcSubpass = i ? 0 : VK_SUBPASS_EXTERNAL;
    swap_apply_gamma_render_pass_dependency.dstSubpass = i ? VK_SUBPASS_EXTERNAL : 0;
    swap_apply_gamma_render_pass_dependency.srcStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    swap_apply_gamma_render_pass_dependency.dstStageMask =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    swap_apply_gamma_render_pass_dependency.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    swap_apply_gamma_render_pass_dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
    swap_apply_gamma_render_pass_dependency.dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;
  }
  VkRenderPassCreateInfo swap_apply_gamma_render_pass_create_info;
  swap_apply_gamma_render_pass_create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
  swap_apply_gamma_render_pass_create_info.pNext = nullptr;
  swap_apply_gamma_render_pass_create_info.flags = 0;
  swap_apply_gamma_render_pass_create_info.attachmentCount = 1;
  swap_apply_gamma_render_pass_create_info.pAttachments = &swap_apply_gamma_render_pass_attachment;
  swap_apply_gamma_render_pass_create_info.subpassCount = 1;
  swap_apply_gamma_render_pass_create_info.pSubpasses = &swap_apply_gamma_render_pass_subpass;
  swap_apply_gamma_render_pass_create_info.dependencyCount =
      uint32_t(rex::countof(swap_apply_gamma_render_pass_dependencies));
  swap_apply_gamma_render_pass_create_info.pDependencies =
      swap_apply_gamma_render_pass_dependencies;
  if (dfn.vkCreateRenderPass(device, &swap_apply_gamma_render_pass_create_info, nullptr,
                             &swap_apply_gamma_render_pass_) != VK_SUCCESS) {
    REXGPU_ERROR("Failed to create the gamma ramp application render pass");
    return false;
  }

  // Gamma ramp application pipeline.
  // Using a graphics pipeline, not a compute one, because storage image support
  // is optional for VK_FORMAT_A2B10G10R10_UNORM_PACK32.

  enum SwapApplyGammaPixelShader {
    kSwapApplyGammaPixelShader256EntryTable,
    kSwapApplyGammaPixelShaderPWL,
    kSwapApplyGammaPixelShader256EntryTableFxaaLuma,
    kSwapApplyGammaPixelShaderPWLFxaaLuma,

    kSwapApplyGammaPixelShaderCount,
  };
  std::array<VkShaderModule, kSwapApplyGammaPixelShaderCount> swap_apply_gamma_pixel_shaders{};
  bool swap_apply_gamma_pixel_shaders_created =
      (swap_apply_gamma_pixel_shaders[kSwapApplyGammaPixelShader256EntryTable] =
           ui::vulkan::util::CreateShaderModule(vulkan_device, shaders::apply_gamma_table_ps,
                                                sizeof(shaders::apply_gamma_table_ps))) !=
          VK_NULL_HANDLE &&
      (swap_apply_gamma_pixel_shaders[kSwapApplyGammaPixelShaderPWL] =
           ui::vulkan::util::CreateShaderModule(vulkan_device, shaders::apply_gamma_pwl_ps,
                                                sizeof(shaders::apply_gamma_pwl_ps))) !=
          VK_NULL_HANDLE &&
      (swap_apply_gamma_pixel_shaders[kSwapApplyGammaPixelShader256EntryTableFxaaLuma] =
           ui::vulkan::util::CreateShaderModule(
               vulkan_device, shaders::apply_gamma_table_fxaa_luma_ps,
               sizeof(shaders::apply_gamma_table_fxaa_luma_ps))) != VK_NULL_HANDLE &&
      (swap_apply_gamma_pixel_shaders[kSwapApplyGammaPixelShaderPWLFxaaLuma] =
           ui::vulkan::util::CreateShaderModule(
               vulkan_device, shaders::apply_gamma_pwl_fxaa_luma_ps,
               sizeof(shaders::apply_gamma_pwl_fxaa_luma_ps))) != VK_NULL_HANDLE;
  if (!swap_apply_gamma_pixel_shaders_created) {
    REXGPU_ERROR("Failed to create the gamma ramp application pixel shader modules");
    for (VkShaderModule swap_apply_gamma_pixel_shader : swap_apply_gamma_pixel_shaders) {
      if (swap_apply_gamma_pixel_shader != VK_NULL_HANDLE) {
        dfn.vkDestroyShaderModule(device, swap_apply_gamma_pixel_shader, nullptr);
      }
    }
    return false;
  }

  VkPipelineShaderStageCreateInfo swap_apply_gamma_pipeline_stages[2];
  swap_apply_gamma_pipeline_stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  swap_apply_gamma_pipeline_stages[0].pNext = nullptr;
  swap_apply_gamma_pipeline_stages[0].flags = 0;
  swap_apply_gamma_pipeline_stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
  swap_apply_gamma_pipeline_stages[0].module = ui::vulkan::util::CreateShaderModule(
      vulkan_device, shaders::fullscreen_cw_vs, sizeof(shaders::fullscreen_cw_vs));
  if (swap_apply_gamma_pipeline_stages[0].module == VK_NULL_HANDLE) {
    REXGPU_ERROR("Failed to create the gamma ramp application vertex shader module");
    for (VkShaderModule swap_apply_gamma_pixel_shader : swap_apply_gamma_pixel_shaders) {
      assert_true(swap_apply_gamma_pixel_shader != VK_NULL_HANDLE);
      dfn.vkDestroyShaderModule(device, swap_apply_gamma_pixel_shader, nullptr);
    }
    return false;
  }
  swap_apply_gamma_pipeline_stages[0].pName = "main";
  swap_apply_gamma_pipeline_stages[0].pSpecializationInfo = nullptr;
  swap_apply_gamma_pipeline_stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
  swap_apply_gamma_pipeline_stages[1].pNext = nullptr;
  swap_apply_gamma_pipeline_stages[1].flags = 0;
  swap_apply_gamma_pipeline_stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
  // The fragment shader module will be specified later.
  swap_apply_gamma_pipeline_stages[1].pName = "main";
  swap_apply_gamma_pipeline_stages[1].pSpecializationInfo = nullptr;

  VkPipelineVertexInputStateCreateInfo swap_apply_gamma_pipeline_vertex_input_state = {};
  swap_apply_gamma_pipeline_vertex_input_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

  VkPipelineInputAssemblyStateCreateInfo swap_apply_gamma_pipeline_input_assembly_state;
  swap_apply_gamma_pipeline_input_assembly_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
  swap_apply_gamma_pipeline_input_assembly_state.pNext = nullptr;
  swap_apply_gamma_pipeline_input_assembly_state.flags = 0;
  swap_apply_gamma_pipeline_input_assembly_state.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
  swap_apply_gamma_pipeline_input_assembly_state.primitiveRestartEnable = VK_FALSE;

  VkPipelineViewportStateCreateInfo swap_apply_gamma_pipeline_viewport_state;
  swap_apply_gamma_pipeline_viewport_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
  swap_apply_gamma_pipeline_viewport_state.pNext = nullptr;
  swap_apply_gamma_pipeline_viewport_state.flags = 0;
  swap_apply_gamma_pipeline_viewport_state.viewportCount = 1;
  swap_apply_gamma_pipeline_viewport_state.pViewports = nullptr;
  swap_apply_gamma_pipeline_viewport_state.scissorCount = 1;
  swap_apply_gamma_pipeline_viewport_state.pScissors = nullptr;

  VkPipelineRasterizationStateCreateInfo swap_apply_gamma_pipeline_rasterization_state = {};
  swap_apply_gamma_pipeline_rasterization_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
  swap_apply_gamma_pipeline_rasterization_state.polygonMode = VK_POLYGON_MODE_FILL;
  swap_apply_gamma_pipeline_rasterization_state.cullMode = VK_CULL_MODE_NONE;
  swap_apply_gamma_pipeline_rasterization_state.frontFace = VK_FRONT_FACE_CLOCKWISE;
  swap_apply_gamma_pipeline_rasterization_state.lineWidth = 1.0f;

  VkPipelineMultisampleStateCreateInfo swap_apply_gamma_pipeline_multisample_state = {};
  swap_apply_gamma_pipeline_multisample_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
  swap_apply_gamma_pipeline_multisample_state.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

  VkPipelineColorBlendAttachmentState swap_apply_gamma_pipeline_color_blend_attachment_state = {};
  swap_apply_gamma_pipeline_color_blend_attachment_state.colorWriteMask =
      VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
      VK_COLOR_COMPONENT_A_BIT;
  VkPipelineColorBlendStateCreateInfo swap_apply_gamma_pipeline_color_blend_state = {};
  swap_apply_gamma_pipeline_color_blend_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
  swap_apply_gamma_pipeline_color_blend_state.attachmentCount = 1;
  swap_apply_gamma_pipeline_color_blend_state.pAttachments =
      &swap_apply_gamma_pipeline_color_blend_attachment_state;

  static const VkDynamicState kSwapApplyGammaPipelineDynamicStates[] = {
      VK_DYNAMIC_STATE_VIEWPORT,
      VK_DYNAMIC_STATE_SCISSOR,
  };
  VkPipelineDynamicStateCreateInfo swap_apply_gamma_pipeline_dynamic_state;
  swap_apply_gamma_pipeline_dynamic_state.sType =
      VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
  swap_apply_gamma_pipeline_dynamic_state.pNext = nullptr;
  swap_apply_gamma_pipeline_dynamic_state.flags = 0;
  swap_apply_gamma_pipeline_dynamic_state.dynamicStateCount =
      uint32_t(rex::countof(kSwapApplyGammaPipelineDynamicStates));
  swap_apply_gamma_pipeline_dynamic_state.pDynamicStates = kSwapApplyGammaPipelineDynamicStates;

  VkGraphicsPipelineCreateInfo swap_apply_gamma_pipeline_create_info;
  swap_apply_gamma_pipeline_create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
  swap_apply_gamma_pipeline_create_info.pNext = nullptr;
  swap_apply_gamma_pipeline_create_info.flags = 0;
  swap_apply_gamma_pipeline_create_info.stageCount =
      uint32_t(rex::countof(swap_apply_gamma_pipeline_stages));
  swap_apply_gamma_pipeline_create_info.pStages = swap_apply_gamma_pipeline_stages;
  swap_apply_gamma_pipeline_create_info.pVertexInputState =
      &swap_apply_gamma_pipeline_vertex_input_state;
  swap_apply_gamma_pipeline_create_info.pInputAssemblyState =
      &swap_apply_gamma_pipeline_input_assembly_state;
  swap_apply_gamma_pipeline_create_info.pTessellationState = nullptr;
  swap_apply_gamma_pipeline_create_info.pViewportState = &swap_apply_gamma_pipeline_viewport_state;
  swap_apply_gamma_pipeline_create_info.pRasterizationState =
      &swap_apply_gamma_pipeline_rasterization_state;
  swap_apply_gamma_pipeline_create_info.pMultisampleState =
      &swap_apply_gamma_pipeline_multisample_state;
  swap_apply_gamma_pipeline_create_info.pDepthStencilState = nullptr;
  swap_apply_gamma_pipeline_create_info.pColorBlendState =
      &swap_apply_gamma_pipeline_color_blend_state;
  swap_apply_gamma_pipeline_create_info.pDynamicState = &swap_apply_gamma_pipeline_dynamic_state;
  swap_apply_gamma_pipeline_create_info.layout = swap_apply_gamma_pipeline_layout_;
  swap_apply_gamma_pipeline_create_info.renderPass = swap_apply_gamma_render_pass_;
  swap_apply_gamma_pipeline_create_info.subpass = 0;
  swap_apply_gamma_pipeline_create_info.basePipelineHandle = VK_NULL_HANDLE;
  swap_apply_gamma_pipeline_create_info.basePipelineIndex = -1;
  std::array<VkShaderModule, 4> swap_apply_gamma_pipeline_pixel_shaders = {
      swap_apply_gamma_pixel_shaders[kSwapApplyGammaPixelShader256EntryTable],
      swap_apply_gamma_pixel_shaders[kSwapApplyGammaPixelShaderPWL],
      swap_apply_gamma_pixel_shaders[kSwapApplyGammaPixelShader256EntryTableFxaaLuma],
      swap_apply_gamma_pixel_shaders[kSwapApplyGammaPixelShaderPWLFxaaLuma],
  };
  std::array<VkPipeline*, 4> swap_apply_gamma_pipelines = {
      &swap_apply_gamma_256_entry_table_pipeline_,
      &swap_apply_gamma_pwl_pipeline_,
      &swap_apply_gamma_256_entry_table_fxaa_luma_pipeline_,
      &swap_apply_gamma_pwl_fxaa_luma_pipeline_,
  };
  std::array<VkResult, 4> swap_apply_gamma_pipeline_create_results;
  for (size_t i = 0; i < swap_apply_gamma_pipelines.size(); ++i) {
    swap_apply_gamma_pipeline_stages[1].module = swap_apply_gamma_pipeline_pixel_shaders[i];
    swap_apply_gamma_pipeline_create_results[i] = dfn.vkCreateGraphicsPipelines(
        device, VK_NULL_HANDLE, 1, &swap_apply_gamma_pipeline_create_info, nullptr,
        swap_apply_gamma_pipelines[i]);
  }
  if (!vulkan_device->properties().imageViewFormatSwizzle) {
    auto create_rb_swap_pipeline = [&](const char* source, VkPipeline& pipeline_out,
                                       const char* pipeline_name) {
      std::vector<uint32_t> pixel_spirv;
      std::string compile_error;
      if (!CompileGlslToSpirv(VK_SHADER_STAGE_FRAGMENT_BIT, source, pixel_spirv, compile_error)) {
        REXGPU_WARN("Failed to compile {} shader to SPIR-V: {}", pipeline_name, compile_error);
        return;
      }
      VkShaderModule pixel_shader_module = ui::vulkan::util::CreateShaderModule(
          vulkan_device, pixel_spirv.data(), sizeof(uint32_t) * pixel_spirv.size());
      if (pixel_shader_module == VK_NULL_HANDLE) {
        REXGPU_WARN("Failed to create {} shader module", pipeline_name);
        return;
      }
      swap_apply_gamma_pipeline_stages[1].module = pixel_shader_module;
      if (dfn.vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1,
                                        &swap_apply_gamma_pipeline_create_info, nullptr,
                                        &pipeline_out) != VK_SUCCESS) {
        REXGPU_WARN("Failed to create {} pipeline", pipeline_name);
        pipeline_out = VK_NULL_HANDLE;
      }
      dfn.vkDestroyShaderModule(device, pixel_shader_module, nullptr);
    };
    create_rb_swap_pipeline(GetSwapApplyGammaTablePixelRbSwapSource(),
                            swap_apply_gamma_256_entry_table_rb_swap_pipeline_,
                            "swap gamma table red/blue fallback");
    create_rb_swap_pipeline(GetSwapApplyGammaPwlPixelRbSwapSource(),
                            swap_apply_gamma_pwl_rb_swap_pipeline_,
                            "swap gamma PWL red/blue fallback");
  }
  dfn.vkDestroyShaderModule(device, swap_apply_gamma_pipeline_stages[0].module, nullptr);
  for (VkShaderModule swap_apply_gamma_pixel_shader : swap_apply_gamma_pixel_shaders) {
    assert_true(swap_apply_gamma_pixel_shader != VK_NULL_HANDLE);
    dfn.vkDestroyShaderModule(device, swap_apply_gamma_pixel_shader, nullptr);
  }
  if (std::any_of(swap_apply_gamma_pipeline_create_results.begin(),
                  swap_apply_gamma_pipeline_create_results.end(),
                  [](VkResult result) { return result != VK_SUCCESS; })) {
    REXGPU_ERROR("Failed to create the gamma ramp application pipelines");
    return false;
  }

  // Gamma ramp application compute pipelines.
  swap_apply_gamma_compute_256_entry_table_pipeline_ = ui::vulkan::util::CreateComputePipeline(
      vulkan_device, swap_apply_gamma_compute_pipeline_layout_, shaders::apply_gamma_table_cs,
      sizeof(shaders::apply_gamma_table_cs));
  if (swap_apply_gamma_compute_256_entry_table_pipeline_ == VK_NULL_HANDLE) {
    REXGPU_WARN(
        "Failed to create the 256-entry table gamma ramp application compute "
        "pipeline, keeping graphics fallback");
  }
  swap_apply_gamma_compute_256_entry_table_fxaa_luma_pipeline_ =
      ui::vulkan::util::CreateComputePipeline(
          vulkan_device, swap_apply_gamma_compute_pipeline_layout_,
          shaders::apply_gamma_table_fxaa_luma_cs, sizeof(shaders::apply_gamma_table_fxaa_luma_cs));
  if (swap_apply_gamma_compute_256_entry_table_fxaa_luma_pipeline_ == VK_NULL_HANDLE) {
    REXGPU_WARN(
        "Failed to create the 256-entry table gamma ramp application compute "
        "pipeline with luma output");
  }
  swap_apply_gamma_compute_pwl_pipeline_ = ui::vulkan::util::CreateComputePipeline(
      vulkan_device, swap_apply_gamma_compute_pipeline_layout_, shaders::apply_gamma_pwl_cs,
      sizeof(shaders::apply_gamma_pwl_cs));
  if (swap_apply_gamma_compute_pwl_pipeline_ == VK_NULL_HANDLE) {
    REXGPU_WARN("Failed to create the PWL gamma ramp application compute pipeline");
  }
  swap_apply_gamma_compute_pwl_fxaa_luma_pipeline_ = ui::vulkan::util::CreateComputePipeline(
      vulkan_device, swap_apply_gamma_compute_pipeline_layout_,
      shaders::apply_gamma_pwl_fxaa_luma_cs, sizeof(shaders::apply_gamma_pwl_fxaa_luma_cs));
  if (swap_apply_gamma_compute_pwl_fxaa_luma_pipeline_ == VK_NULL_HANDLE) {
    REXGPU_WARN(
        "Failed to create the PWL gamma ramp application compute pipeline with "
        "luma output");
  }
  if (!vulkan_device->properties().imageViewFormatSwizzle) {
    auto create_rb_swap_compute_pipeline = [&](const char* source, VkPipeline& pipeline_out,
                                               const char* pipeline_name) {
      std::vector<uint32_t> compute_spirv;
      std::string compile_error;
      if (!CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, source, compute_spirv, compile_error)) {
        REXGPU_WARN("Failed to compile {} shader to SPIR-V: {}", pipeline_name, compile_error);
        return;
      }
      pipeline_out = ui::vulkan::util::CreateComputePipeline(
          vulkan_device, swap_apply_gamma_compute_pipeline_layout_, compute_spirv.data(),
          sizeof(uint32_t) * compute_spirv.size());
      if (pipeline_out == VK_NULL_HANDLE) {
        REXGPU_WARN("Failed to create {} pipeline", pipeline_name);
      }
    };
    create_rb_swap_compute_pipeline(GetSwapApplyGammaTableComputeRbSwapSource(),
                                    swap_apply_gamma_compute_256_entry_table_rb_swap_pipeline_,
                                    "swap gamma table compute red/blue fallback");
    create_rb_swap_compute_pipeline(GetSwapApplyGammaPwlComputeRbSwapSource(),
                                    swap_apply_gamma_compute_pwl_rb_swap_pipeline_,
                                    "swap gamma PWL compute red/blue fallback");
    create_rb_swap_compute_pipeline(
        GetSwapApplyGammaTableFxaaLumaComputeRbSwapSource(),
        swap_apply_gamma_compute_256_entry_table_fxaa_luma_rb_swap_pipeline_,
        "swap gamma table FXAA-luma compute red/blue fallback");
    create_rb_swap_compute_pipeline(GetSwapApplyGammaPwlFxaaLumaComputeRbSwapSource(),
                                    swap_apply_gamma_compute_pwl_fxaa_luma_rb_swap_pipeline_,
                                    "swap gamma PWL FXAA-luma compute red/blue fallback");
  }

  // FXAA compute pipelines, compiled to SPIR-V at runtime.
  std::vector<uint32_t> swap_fxaa_spirv;
  std::string swap_fxaa_compile_error;
  if (!CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, GetSwapFxaaComputeSource(false),
                          swap_fxaa_spirv, swap_fxaa_compile_error)) {
    REXGPU_WARN("Failed to compile FXAA compute shader to SPIR-V: {}", swap_fxaa_compile_error);
  } else {
    swap_fxaa_pipeline_ = ui::vulkan::util::CreateComputePipeline(
        vulkan_device, swap_fxaa_pipeline_layout_, swap_fxaa_spirv.data(),
        sizeof(uint32_t) * swap_fxaa_spirv.size());
    if (swap_fxaa_pipeline_ == VK_NULL_HANDLE) {
      REXGPU_WARN("Failed to create the FXAA compute pipeline");
    }
  }

  std::vector<uint32_t> swap_fxaa_extreme_spirv;
  std::string swap_fxaa_extreme_compile_error;
  if (!CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT, GetSwapFxaaComputeSource(true),
                          swap_fxaa_extreme_spirv, swap_fxaa_extreme_compile_error)) {
    REXGPU_WARN("Failed to compile extreme FXAA compute shader to SPIR-V: {}",
                swap_fxaa_extreme_compile_error);
  } else {
    swap_fxaa_extreme_pipeline_ = ui::vulkan::util::CreateComputePipeline(
        vulkan_device, swap_fxaa_pipeline_layout_, swap_fxaa_extreme_spirv.data(),
        sizeof(uint32_t) * swap_fxaa_extreme_spirv.size());
    if (swap_fxaa_extreme_pipeline_ == VK_NULL_HANDLE) {
      REXGPU_WARN("Failed to create the extreme-quality FXAA compute pipeline");
    }
  }

  VkPipelineLayoutCreateInfo resolve_downscale_layout_create_info = {};
  resolve_downscale_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  VkDescriptorSetLayout resolve_downscale_set_layout =
      descriptor_set_layouts_single_transient_[size_t(
          SingleTransientDescriptorLayout::kStorageBufferPairCompute)];
  resolve_downscale_layout_create_info.setLayoutCount = 1;
  resolve_downscale_layout_create_info.pSetLayouts = &resolve_downscale_set_layout;
  VkPushConstantRange resolve_downscale_push_constant_range = {};
  resolve_downscale_push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  resolve_downscale_push_constant_range.offset = 0;
  resolve_downscale_push_constant_range.size = sizeof(ResolveDownscaleConstants);
  resolve_downscale_layout_create_info.pushConstantRangeCount = 1;
  resolve_downscale_layout_create_info.pPushConstantRanges = &resolve_downscale_push_constant_range;
  if (dfn.vkCreatePipelineLayout(device, &resolve_downscale_layout_create_info, nullptr,
                                 &resolve_downscale_pipeline_layout_) == VK_SUCCESS) {
    resolve_downscale_pipeline_ = ui::vulkan::util::CreateComputePipeline(
        vulkan_device, resolve_downscale_pipeline_layout_, shaders::resolve_downscale_cs,
        sizeof(shaders::resolve_downscale_cs));
    if (resolve_downscale_pipeline_ == VK_NULL_HANDLE) {
      REXGPU_WARN("Failed to create Vulkan resolve-downscale readback pipeline");
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device,
                                             resolve_downscale_pipeline_layout_);
    }
  } else {
    REXGPU_WARN("Failed to create Vulkan resolve-downscale pipeline layout");
  }

  occlusion_query_resources_available_ = InitializeOcclusionQueryResources();
  InitializeGpuTimingResources();

  // Just not to expose uninitialized memory.
  std::memset(&system_constants_, 0, sizeof(system_constants_));

  // Always log once at GPU bring-up (NativeX may never hit depth-transfer audit).
  REXGPU_INFO(
      "[INTERMEDIATE_DEPTH_XFER] cvar vulkan_elide_intermediate_depth_transfers={} "
      "(rejected default OFF — catacomb lighting)",
      REXCVAR_GET(vulkan_elide_intermediate_depth_transfers) ? "true" : "false");

  return true;
}

void VulkanCommandProcessor::ShutdownContext() {
  AwaitAllQueueOperationsCompletion();
  FlushCompletedAuditRows(true);
  LogTopBarrierTuples();
  ShutdownGpuTimingResources();
  // Keep the diagnostic allocation alive until all submissions using it have
  // completed at the queue/submission barrier above.
  diagnostic_storage_buffer_.reset();
  InvalidateAllVertexBufferResidency();
  ShutdownOcclusionQueryResources();

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  DestroyScratchBuffer();

  for (auto& readback_pair : readback_buffers_) {
    ReadbackBuffer& readback = readback_pair.second;
    for (uint32_t i = 0; i < 2; ++i) {
      if (readback.mapped_data[i] && readback.memories[i] != VK_NULL_HANDLE) {
        dfn.vkUnmapMemory(device, readback.memories[i]);
      }
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, readback.buffers[i]);
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, readback.memories[i]);
      readback.mapped_data[i] = nullptr;
      readback.sizes[i] = 0;
      readback.submission_written[i] = 0;
      readback.written_size[i] = 0;
    }
  }
  readback_buffers_.clear();
  for (auto& readback_pair : memexport_readback_buffers_) {
    ReadbackBuffer& readback = readback_pair.second;
    for (uint32_t i = 0; i < 2; ++i) {
      if (readback.mapped_data[i] && readback.memories[i] != VK_NULL_HANDLE) {
        dfn.vkUnmapMemory(device, readback.memories[i]);
      }
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, readback.buffers[i]);
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, readback.memories[i]);
      readback.mapped_data[i] = nullptr;
      readback.sizes[i] = 0;
      readback.submission_written[i] = 0;
      readback.written_size[i] = 0;
    }
  }
  memexport_readback_buffers_.clear();

  resolve_downscale_buffer_size_ = 0;
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, resolve_downscale_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         resolve_downscale_buffer_memory_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         resolve_downscale_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device,
                                         resolve_downscale_pipeline_layout_);

  for (SwapFramebuffer& swap_framebuffer : swap_framebuffers_) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device,
                                           swap_framebuffer.framebuffer);
  }
  DestroySwapFxaaSourceImage();

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_fxaa_extreme_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device, swap_fxaa_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_apply_gamma_compute_pwl_fxaa_luma_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_apply_gamma_compute_pwl_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_apply_gamma_compute_pwl_fxaa_luma_rb_swap_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_apply_gamma_compute_pwl_rb_swap_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(
      dfn.vkDestroyPipeline, device, swap_apply_gamma_compute_256_entry_table_fxaa_luma_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(
      dfn.vkDestroyPipeline, device,
      swap_apply_gamma_compute_256_entry_table_fxaa_luma_rb_swap_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_apply_gamma_compute_256_entry_table_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(
      dfn.vkDestroyPipeline, device, swap_apply_gamma_compute_256_entry_table_rb_swap_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_apply_gamma_pwl_rb_swap_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_apply_gamma_256_entry_table_rb_swap_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_apply_gamma_pwl_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_apply_gamma_pwl_fxaa_luma_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_apply_gamma_256_entry_table_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipeline, device,
                                         swap_apply_gamma_256_entry_table_fxaa_luma_pipeline_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyRenderPass, device,
                                         swap_apply_gamma_render_pass_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device,
                                         swap_fxaa_pipeline_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device,
                                         swap_apply_gamma_compute_pipeline_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyPipelineLayout, device,
                                         swap_apply_gamma_pipeline_layout_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroySampler, device, swap_sampler_linear_clamp_);

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorPool, device,
                                         swap_descriptor_pool_);

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device,
                                         swap_descriptor_set_layout_uniform_texel_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device,
                                         swap_descriptor_set_layout_storage_image_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device,
                                         swap_descriptor_set_layout_combined_image_sampler_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device,
                                         swap_descriptor_set_layout_sampled_image_);
  for (VkBufferView& gamma_ramp_buffer_view : gamma_ramp_buffer_views_) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBufferView, device, gamma_ramp_buffer_view);
  }
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, gamma_ramp_upload_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                         gamma_ramp_upload_buffer_memory_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, gamma_ramp_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, gamma_ramp_buffer_memory_);

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorPool, device,
                                         shared_memory_and_edram_descriptor_pool_);

  texture_cache_.reset();

  pipeline_cache_.reset();

  render_target_cache_.reset();

  primitive_processor_.reset();

  shared_memory_.reset();

  ClearTransientDescriptorPools();

  for (const auto& pipeline_layout_pair : pipeline_layouts_) {
    dfn.vkDestroyPipelineLayout(device, pipeline_layout_pair.second.GetPipelineLayout(), nullptr);
  }
  pipeline_layouts_.clear();
  for (const auto& descriptor_set_layout_pair : descriptor_set_layouts_textures_) {
    dfn.vkDestroyDescriptorSetLayout(device, descriptor_set_layout_pair.second, nullptr);
  }
  descriptor_set_layouts_textures_.clear();

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device,
                                         descriptor_set_layout_shared_memory_and_edram_);
  for (VkDescriptorSetLayout& descriptor_set_layout_single_transient :
       descriptor_set_layouts_single_transient_) {
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device,
                                           descriptor_set_layout_single_transient);
  }
  if (constant_ring_mapping_ != nullptr) {
    dfn.vkUnmapMemory(device, constant_ring_buffer_memory_);
    constant_ring_mapping_ = nullptr;
  }
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, constant_ring_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, constant_ring_buffer_memory_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorPool, device,
                                         constant_descriptor_pool_);

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device,
                                         descriptor_set_layout_constants_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyDescriptorSetLayout, device,
                                         descriptor_set_layout_empty_);

  uniform_buffer_pool_.reset();

  sparse_bind_wait_stage_mask_ = 0;
  sparse_buffer_binds_.clear();
  sparse_memory_binds_.clear();

  deferred_command_buffer_.Reset();
  for (const auto& command_buffer_pair : command_buffers_submitted_) {
    dfn.vkDestroyCommandPool(device, command_buffer_pair.second.pool, nullptr);
  }
  command_buffers_submitted_.clear();
  for (const CommandBuffer& command_buffer : command_buffers_writable_) {
    dfn.vkDestroyCommandPool(device, command_buffer.pool, nullptr);
  }
  command_buffers_writable_.clear();

  for (const auto& destroy_pair : destroy_framebuffers_) {
    dfn.vkDestroyFramebuffer(device, destroy_pair.second, nullptr);
  }
  destroy_framebuffers_.clear();
  for (const auto& destroy_pair : destroy_image_views_) {
    dfn.vkDestroyImageView(device, destroy_pair.second, nullptr);
  }
  destroy_image_views_.clear();
  for (const auto& destroy_pair : destroy_buffers_) {
    dfn.vkDestroyBuffer(device, destroy_pair.second, nullptr);
  }
  destroy_buffers_.clear();
  for (const auto& destroy_pair : destroy_images_) {
    dfn.vkDestroyImage(device, destroy_pair.second, nullptr);
  }
  destroy_images_.clear();
  for (const auto& destroy_pair : destroy_memory_) {
    dfn.vkFreeMemory(device, destroy_pair.second, nullptr);
  }
  destroy_memory_.clear();

  std::memset(closed_frame_submissions_, 0, sizeof(closed_frame_submissions_));
  frame_completed_ = 0;
  frame_current_ = 1;
  frame_open_ = false;

  for (const auto& semaphore : submissions_in_flight_semaphores_) {
    dfn.vkDestroySemaphore(device, semaphore.second, nullptr);
  }
  submissions_in_flight_semaphores_.clear();
  for (VkFence& fence : submissions_in_flight_fences_) {
    dfn.vkDestroyFence(device, fence, nullptr);
  }
  submissions_in_flight_fences_.clear();
  current_submission_wait_stage_masks_.clear();
  for (VkSemaphore semaphore : current_submission_wait_semaphores_) {
    dfn.vkDestroySemaphore(device, semaphore, nullptr);
  }
  current_submission_wait_semaphores_.clear();
  submission_completed_ = 0;
  submission_open_ = false;

  for (VkSemaphore semaphore : semaphores_free_) {
    dfn.vkDestroySemaphore(device, semaphore, nullptr);
  }
  semaphores_free_.clear();
  for (VkFence fence : fences_free_) {
    dfn.vkDestroyFence(device, fence, nullptr);
  }
  fences_free_.clear();

  device_lost_ = false;

  CommandProcessor::ShutdownContext();
}

void VulkanCommandProcessor::WriteRegister(uint32_t index, uint32_t value) {
  if (frame_current_ >= 8670 && frame_current_ <= 8685) {
    if (index == XE_GPU_REG_RB_COLORCONTROL || index == XE_GPU_REG_RB_ALPHA_REF ||
        index == XE_GPU_REG_RB_DEPTHCONTROL || index == XE_GPU_REG_RB_BLENDCONTROL0) {
      REXGPU_INFO("[RegTrace] frame {} WriteRegister 0x{:04X} = 0x{:08X}", frame_current_, index, value);
    }
  }
  CommandProcessor::WriteRegister(index, value);

  if (index >= XE_GPU_REG_SHADER_CONSTANT_000_X && index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    if (frame_open_) {
      uint32_t float_constant_index = (index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      if (float_constant_index >= 256) {
        float_constant_index -= 256;
        if (current_float_constant_map_pixel_[float_constant_index >> 6] &
            (1ull << (float_constant_index & 63))) {
          current_constant_buffers_up_to_date_ &=
              ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatPixel);
        }
      } else {
        if (current_float_constant_map_vertex_[float_constant_index >> 6] &
            (1ull << (float_constant_index & 63))) {
          current_constant_buffers_up_to_date_ &=
              ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex);
        }
      }
    }
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    current_constant_buffers_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferBoolLoop);
  } else if (index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
             index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    current_constant_buffers_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFetch);
    if (texture_cache_) {
      texture_cache_->TextureFetchConstantWritten((index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) /
                                                  6);
    }
    InvalidateVertexBufferResidency((index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0) / 2);
  }
}

void VulkanCommandProcessor::WriteRegistersFromMem(uint32_t start_index, uint32_t* base,
                                                   uint32_t num_registers) {
  if (!num_registers) {
    return;
  }
  uint32_t end_index = start_index + num_registers - 1;

  auto range_has_any_constant_usage = [](const uint64_t* usage_map, uint32_t first_constant,
                                         uint32_t last_constant) -> bool {
    if (first_constant > last_constant) {
      return false;
    }
    uint32_t first_word = first_constant >> 6;
    uint32_t last_word = last_constant >> 6;
    uint32_t first_bit = first_constant & 63;
    uint32_t last_bit = last_constant & 63;
    if (first_word == last_word) {
      uint32_t bit_count = last_bit - first_bit + 1;
      uint64_t mask = bit_count == 64 ? UINT64_MAX : ((UINT64_C(1) << bit_count) - 1) << first_bit;
      return (usage_map[first_word] & mask) != 0;
    }
    if (usage_map[first_word] & (UINT64_MAX << first_bit)) {
      return true;
    }
    for (uint32_t word = first_word + 1; word < last_word; ++word) {
      if (usage_map[word]) {
        return true;
      }
    }
    uint64_t last_mask = last_bit == 63 ? UINT64_MAX : ((UINT64_C(1) << (last_bit + 1)) - 1);
    return (usage_map[last_word] & last_mask) != 0;
  };

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_000_X &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_511_W) {
    memory::copy_and_swap(register_file_->values + start_index, base, num_registers);
    if (frame_open_) {
      uint32_t first_float_constant = (start_index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      uint32_t last_float_constant = (end_index - XE_GPU_REG_SHADER_CONSTANT_000_X) >> 2;
      if (first_float_constant < 256) {
        uint32_t last_vertex_constant = std::min(last_float_constant, 255u);
        if (range_has_any_constant_usage(current_float_constant_map_vertex_, first_float_constant,
                                         last_vertex_constant)) {
          current_constant_buffers_up_to_date_ &=
              ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex);
        }
      }
      if (last_float_constant >= 256) {
        uint32_t first_pixel_constant =
            first_float_constant >= 256 ? first_float_constant - 256 : 0;
        uint32_t last_pixel_constant = last_float_constant - 256;
        if (range_has_any_constant_usage(current_float_constant_map_pixel_, first_pixel_constant,
                                         last_pixel_constant)) {
          current_constant_buffers_up_to_date_ &=
              ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatPixel);
        }
      }
    }
    return;
  }

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031 &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_LOOP_31) {
    memory::copy_and_swap(register_file_->values + start_index, base, num_registers);
    current_constant_buffers_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferBoolLoop);
    return;
  }

  if (start_index >= XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0 &&
      end_index <= XE_GPU_REG_SHADER_CONSTANT_FETCH_31_5) {
    memory::copy_and_swap(register_file_->values + start_index, base, num_registers);
    current_constant_buffers_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFetch);
    uint32_t first_fetch_dword = start_index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0;
    uint32_t last_fetch_dword = end_index - XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0;
    if (texture_cache_) {
      texture_cache_->TextureFetchConstantsWritten(first_fetch_dword / 6, last_fetch_dword / 6);
    }
    InvalidateVertexBufferResidencyRange(first_fetch_dword / 2, last_fetch_dword / 2);
    return;
  }

  CommandProcessor::WriteRegistersFromMem(start_index, base, num_registers);
}

void VulkanCommandProcessor::SparseBindBuffer(VkBuffer buffer, uint32_t bind_count,
                                              const VkSparseMemoryBind* binds,
                                              VkPipelineStageFlags wait_stage_mask) {
  if (!bind_count) {
    return;
  }
  SparseBufferBind& buffer_bind = sparse_buffer_binds_.emplace_back();
  buffer_bind.buffer = buffer;
  buffer_bind.bind_offset = sparse_memory_binds_.size();
  buffer_bind.bind_count = bind_count;
  sparse_memory_binds_.reserve(sparse_memory_binds_.size() + bind_count);
  sparse_memory_binds_.insert(sparse_memory_binds_.end(), binds, binds + bind_count);
  sparse_bind_wait_stage_mask_ |= wait_stage_mask;
}

void VulkanCommandProcessor::OnGammaRamp256EntryTableValueWritten() {
  gamma_ramp_256_entry_table_current_frame_ = UINT32_MAX;
}

void VulkanCommandProcessor::OnGammaRampPWLValueWritten() {
  gamma_ramp_pwl_current_frame_ = UINT32_MAX;
}

void VulkanCommandProcessor::IssueSwap(uint32_t frontbuffer_ptr, uint32_t frontbuffer_width,
                                       uint32_t frontbuffer_height) {
  SCOPE_profile_cpu_f("gpu");
  FrameCpuTimeline::Scope audit_swap(GetFrameCpuTimeline(), FrameCpuTimeline::Bucket::kSwap);
  audit_swap_start_ = REXCVAR_GET(gpu_frame_audit_enable)
                          ? std::chrono::steady_clock::now()
                          : std::chrono::steady_clock::time_point{};
  const uint64_t provenance_frame = frame_current_;
  const bool provenance_frame_sampled = IsGpuProvenanceFrame(provenance_frame);
  vertex_buffers_in_sync_[0] = 0;
  vertex_buffers_in_sync_[1] = 0;

  {
    static uint64_t s_swap_count = 0;
    static auto s_last_time = std::chrono::steady_clock::now();
    static uint64_t s_last_swap_count = 0;

    ++s_swap_count;
    auto now = std::chrono::steady_clock::now();
    frame_swap_interval_ms_ = previous_swap_time_ == std::chrono::steady_clock::time_point{}
                                  ? 0.0
                                  : std::chrono::duration<double, std::milli>(
                                        now - previous_swap_time_).count();
    previous_swap_time_ = now;
    auto elapsed_ms = std::chrono::duration<double, std::milli>(now - s_last_time).count();
    if (elapsed_ms >= 1000.0) {
      uint64_t frames = s_swap_count - s_last_swap_count;
      double fps = (frames * 1000.0) / elapsed_ms;
      double avg_frame_time = elapsed_ms / frames;
      guest_frame_stats_.fps.store(fps, std::memory_order_relaxed);
      guest_frame_stats_.frame_time_ms.store(avg_frame_time, std::memory_order_relaxed);
      guest_frame_stats_.frame_count.store(s_swap_count, std::memory_order_relaxed);
      if (REXCVAR_GET(gpu_profile_cpu)) {
        REXGPU_INFO("[Guest Swap Rate] {:.1f} swaps/s, {:.2f} ms/swap", fps, avg_frame_time);
      }
      s_last_time = now;
      s_last_swap_count = s_swap_count;
    }
  }

  if (!graphics_system_)
    return;
  if (!host_rendering()) {
    if (submission_open_) {
      EndSubmission(true);
    }
    return;
  }
  ui::Presenter* presenter = graphics_system_->presenter();
  if (!presenter) {
    REXGPU_ERROR("XELOG_GPU PRESENT: NO PRESENTER");
    return;
  }

  // In case the swap command is the only one in the frame.
  if (!BeginSubmission(true)) {
    REXGPU_ERROR("XELOG_GPU PRESENT: BeginSubmission FAILED");
    return;
  }

  bool skip_present_due_async_placeholder = REXCVAR_GET(async_shader_compilation) &&
                                            REXCVAR_GET(vulkan_async_skip_incomplete_frames) &&
                                            frame_used_async_placeholder_pipeline_;
  if (skip_present_due_async_placeholder) {
    if (provenance_frame_sampled) {
      REXGPU_INFO("GPU provenance frame {} status=skipped_async_placeholder", provenance_frame);
    }
    static bool skipped_incomplete_frame_logged = false;
    if (!skipped_incomplete_frame_logged) {
      skipped_incomplete_frame_logged = true;
      REXGPU_WARN(
          "Skipping Vulkan frame presentation due to async placeholder draw "
          "usage in this frame");
    }
    EndSubmission(true);
    return;
  }

  SwapPostEffect swap_post_effect = GetActualSwapPostEffect();

  // Obtain the actual swap source texture size (resolution-scaled if it's a
  // resolve destination, or not otherwise).
  uint32_t frontbuffer_width_scaled, frontbuffer_height_scaled;
  uint32_t frontbuffer_width_unscaled = 0, frontbuffer_height_unscaled = 0;
  xenos::TextureFormat frontbuffer_format;
  bool swap_source_needs_rb_swap = false;
  uint32_t swap_source_guest_base = UINT32_MAX;
  VkImageView swap_texture_view = texture_cache_->RequestSwapTexture(
      frontbuffer_width_scaled, frontbuffer_height_scaled, frontbuffer_format,
      &frontbuffer_width_unscaled, &frontbuffer_height_unscaled, &swap_source_needs_rb_swap,
      &swap_source_guest_base);
  if (swap_texture_view == VK_NULL_HANDLE) {
    if (provenance_frame_sampled) {
      REXGPU_INFO("GPU provenance frame {} slot0_guest_base=0x{:08X} status=no_swap_source",
                  provenance_frame, swap_source_guest_base);
    }
    REXGPU_ERROR("XELOG_GPU PRESENT: swap_texture_view=NULL");
    return;
  }
  // The swap gamma / FXAA pass samples source texels by pixel index, but swap
  // textures may be allocation-padded. Prefer the active frontbuffer region
  // from the swap packet, scaled proportionally to the actual source texture.
  auto get_active_swap_dimension = [](uint32_t packet_unscaled, uint32_t source_unscaled,
                                      uint32_t source_scaled) -> uint32_t {
    if (!source_scaled) {
      return 0;
    }
    uint32_t active_unscaled = packet_unscaled ? packet_unscaled : source_unscaled;
    if (!active_unscaled) {
      return source_scaled;
    }
    if (source_unscaled) {
      active_unscaled = std::min(active_unscaled, source_unscaled);
      uint64_t active_scaled =
          (uint64_t(active_unscaled) * source_scaled + (source_unscaled >> 1)) / source_unscaled;
      return uint32_t(std::clamp<uint64_t>(active_scaled, 1, source_scaled));
    }
    return std::min(active_unscaled, source_scaled);
  };
  uint32_t guest_output_width = get_active_swap_dimension(
      frontbuffer_width, frontbuffer_width_unscaled, frontbuffer_width_scaled);
  uint32_t guest_output_height = get_active_swap_dimension(
      frontbuffer_height, frontbuffer_height_unscaled, frontbuffer_height_scaled);
  if (!guest_output_width) {
    guest_output_width = frontbuffer_width_scaled
                             ? frontbuffer_width_scaled
                             : (frontbuffer_width ? frontbuffer_width : frontbuffer_width_unscaled);
  }
  if (!guest_output_height) {
    guest_output_height =
        frontbuffer_height_scaled
            ? frontbuffer_height_scaled
            : (frontbuffer_height ? frontbuffer_height : frontbuffer_height_unscaled);
  }
  bool swap_source_scaled = frontbuffer_width_unscaled && frontbuffer_height_unscaled &&
                            (frontbuffer_width_scaled != frontbuffer_width_unscaled ||
                             frontbuffer_height_scaled != frontbuffer_height_unscaled);
  if (texture_cache_->IsDrawResolutionScaled()) {
    static bool draw_scale_cache_mismatch_logged = false;
    uint32_t texture_scale_x = texture_cache_->draw_resolution_scale_x();
    uint32_t texture_scale_y = texture_cache_->draw_resolution_scale_y();
    uint32_t rt_scale_x =
        render_target_cache_ ? render_target_cache_->draw_resolution_scale_x() : texture_scale_x;
    uint32_t rt_scale_y =
        render_target_cache_ ? render_target_cache_->draw_resolution_scale_y() : texture_scale_y;
    if (!draw_scale_cache_mismatch_logged &&
        (texture_scale_x != rt_scale_x || texture_scale_y != rt_scale_y)) {
      draw_scale_cache_mismatch_logged = true;
      REXGPU_WARN(
          "Vulkan draw-scale mismatch: texture cache is {}x{}, render target "
          "cache is {}x{}",
          texture_scale_x, texture_scale_y, rt_scale_x, rt_scale_y);
    }
    static bool draw_scale_swap_sizes_logged = false;
    if (!draw_scale_swap_sizes_logged) {
      draw_scale_swap_sizes_logged = true;
      REXGPU_WARN(
          "Vulkan draw-scale swap sizing: packet={}x{}, src_scaled={}x{}, "
          "src_unscaled={}x{}, active={}x{}",
          frontbuffer_width, frontbuffer_height, frontbuffer_width_scaled,
          frontbuffer_height_scaled, frontbuffer_width_unscaled, frontbuffer_height_unscaled,
          guest_output_width, guest_output_height);
    }
  }
  if (texture_cache_->IsDrawResolutionScaled() && !swap_source_scaled) {
    static bool draw_scale_swap_unscaled_logged = false;
    if (!draw_scale_swap_unscaled_logged) {
      draw_scale_swap_unscaled_logged = true;
      REXGPU_WARN(
          "Vulkan draw resolution scaling is enabled, but the swap source is "
          "unscaled ({}x{}). This title may be presenting from an unscaled "
          "resolve path.",
          frontbuffer_width_scaled, frontbuffer_height_scaled);
    }
  }
  REXGPU_DEBUG(
      "XELOG_GPU PRESENT: swap_texture_view={:p} packet_size={}x{} src_size={}x{} "
      "src_unscaled={}x{} guest_output_size={}x{} format={}",
      static_cast<void*>(swap_texture_view), frontbuffer_width, frontbuffer_height,
      frontbuffer_width_scaled, frontbuffer_height_scaled, frontbuffer_width_unscaled,
      frontbuffer_height_unscaled, guest_output_width, guest_output_height,
      static_cast<uint32_t>(frontbuffer_format));

  system::X_VIDEO_MODE video_mode;
  kernel::xboxkrnl::VdQueryVideoMode(&video_mode);
  uint32_t display_width = std::max(uint32_t(1), uint32_t(video_mode.display_width));
  uint32_t display_height = std::max(uint32_t(1), uint32_t(video_mode.display_height));

  bool guest_output_refreshed = presenter->RefreshGuestOutput(
      guest_output_width, guest_output_height, display_width, display_height,
      [this, guest_output_width, guest_output_height, frontbuffer_format, swap_texture_view,
       swap_post_effect,
       swap_source_needs_rb_swap](ui::Presenter::GuestOutputRefreshContext& context) -> bool {
        // In case the swap command is the only one in the frame.
        if (!BeginSubmission(true)) {
          return false;
        }

        auto& vulkan_context =
            static_cast<ui::vulkan::VulkanPresenter::VulkanGuestOutputRefreshContext&>(context);
        uint64_t guest_output_image_version = vulkan_context.image_version();

        const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
        const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
        const VkDevice device = vulkan_device->device();

        uint32_t swap_frame_index = uint32_t(frame_current_ % kMaxFramesInFlight);
        bool use_fxaa = swap_post_effect == SwapPostEffect::kFxaa ||
                        swap_post_effect == SwapPostEffect::kFxaaExtreme;

        // This is according to D3D::InitializePresentationParameters from a
        // game executable, which initializes the 256-entry table gamma ramp for
        // 8_8_8_8 output and the PWL gamma ramp for 2_10_10_10.
        // TODO(Triang3l): Choose between the table and PWL based on
        // DC_LUTA_CONTROL, support both for all formats (and also different
        // increments for PWL).
        bool use_pwl_gamma_ramp =
            frontbuffer_format == xenos::TextureFormat::k_2_10_10_10 ||
            frontbuffer_format == xenos::TextureFormat::k_2_10_10_10_AS_16_16_16_16;
        bool swap_source_requires_compute_rb_swap =
            !vulkan_device->properties().imageViewFormatSwizzle && swap_source_needs_rb_swap;
        auto select_swap_apply_gamma_compute_pipeline = [&](bool use_pwl,
                                                            bool use_fxaa_luma) -> VkPipeline {
          if (use_pwl) {
            if (use_fxaa_luma) {
              return swap_source_requires_compute_rb_swap
                         ? swap_apply_gamma_compute_pwl_fxaa_luma_rb_swap_pipeline_
                         : swap_apply_gamma_compute_pwl_fxaa_luma_pipeline_;
            }
            return swap_source_requires_compute_rb_swap
                       ? swap_apply_gamma_compute_pwl_rb_swap_pipeline_
                       : swap_apply_gamma_compute_pwl_pipeline_;
          }
          if (use_fxaa_luma) {
            return swap_source_requires_compute_rb_swap
                       ? swap_apply_gamma_compute_256_entry_table_fxaa_luma_rb_swap_pipeline_
                       : swap_apply_gamma_compute_256_entry_table_fxaa_luma_pipeline_;
          }
          return swap_source_requires_compute_rb_swap
                     ? swap_apply_gamma_compute_256_entry_table_rb_swap_pipeline_
                     : swap_apply_gamma_compute_256_entry_table_pipeline_;
        };
        VkPipeline swap_apply_gamma_compute_pipeline =
            select_swap_apply_gamma_compute_pipeline(use_pwl_gamma_ramp, use_fxaa);

        if (use_fxaa) {
          if (swap_apply_gamma_compute_pipeline == VK_NULL_HANDLE ||
              swap_fxaa_pipeline_ == VK_NULL_HANDLE ||
              swap_fxaa_extreme_pipeline_ == VK_NULL_HANDLE) {
            static bool fxaa_pipelines_unavailable_logged = false;
            if (!fxaa_pipelines_unavailable_logged) {
              if (swap_source_requires_compute_rb_swap) {
                REXGPU_WARN(
                    "Vulkan FXAA swap effect requested but FXAA compute "
                    "pipelines (including RB-swap fallback) are unavailable, "
                    "falling back to gamma only");
              } else {
                REXGPU_WARN(
                    "Vulkan FXAA swap effect requested but FXAA compute "
                    "pipelines are unavailable, falling back to gamma only");
              }
              fxaa_pipelines_unavailable_logged = true;
            }
            use_fxaa = false;
          } else if (!EnsureSwapFxaaSourceImage(guest_output_width, guest_output_height)) {
            static bool fxaa_source_image_failed_logged = false;
            if (!fxaa_source_image_failed_logged) {
              REXGPU_WARN(
                  "Failed to create the Vulkan FXAA source image, falling "
                  "back to gamma-only presentation");
              fxaa_source_image_failed_logged = true;
            }
            use_fxaa = false;
          }
        }
        if (!use_fxaa) {
          swap_apply_gamma_compute_pipeline =
              select_swap_apply_gamma_compute_pipeline(use_pwl_gamma_ramp, false);
        }
        if (swap_source_requires_compute_rb_swap &&
            swap_apply_gamma_compute_pipeline == VK_NULL_HANDLE) {
          static bool compute_rb_swap_fallback_missing_logged = false;
          if (!compute_rb_swap_fallback_missing_logged) {
            compute_rb_swap_fallback_missing_logged = true;
            REXGPU_WARN(
                "Vulkan imageViewFormatSwizzle is unavailable and the swap "
                "source needs red/blue swizzle, but the compute fallback "
                "pipeline is unavailable; falling back to graphics "
                "presentation path");
          }
        }
        bool use_compute_gamma = swap_apply_gamma_compute_pipeline != VK_NULL_HANDLE;

        // TODO(Triang3l): FXAA can result in more than 8 bits of precision.
        context.SetIs8bpc(!use_pwl_gamma_ramp && !use_fxaa);

        // Update the gamma ramp if it's out of date.
        uint32_t& gamma_ramp_frame_index_ref = use_pwl_gamma_ramp
                                                   ? gamma_ramp_pwl_current_frame_
                                                   : gamma_ramp_256_entry_table_current_frame_;
        if (gamma_ramp_frame_index_ref == UINT32_MAX) {
          constexpr uint32_t kGammaRampSize256EntryTable = sizeof(uint32_t) * 256;
          constexpr uint32_t kGammaRampSizePWL = sizeof(uint16_t) * 2 * 3 * 128;
          constexpr uint32_t kGammaRampSize = kGammaRampSize256EntryTable + kGammaRampSizePWL;
          uint32_t gamma_ramp_offset_in_frame =
              use_pwl_gamma_ramp ? kGammaRampSize256EntryTable : 0;
          uint32_t gamma_ramp_upload_offset =
              kGammaRampSize * swap_frame_index + gamma_ramp_offset_in_frame;
          uint32_t gamma_ramp_size =
              use_pwl_gamma_ramp ? kGammaRampSizePWL : kGammaRampSize256EntryTable;
          void* gamma_ramp_frame_upload =
              reinterpret_cast<uint8_t*>(gamma_ramp_upload_mapping_) + gamma_ramp_upload_offset;
          if (std::endian::native != std::endian::little && use_pwl_gamma_ramp) {
            // R16G16 is first R16, where the shader expects the base, and
            // second G16, where the delta should be, but gamma_ramp_pwl_rgb()
            // is an array of 32-bit DC_LUT_PWL_DATA registers - swap 16 bits in
            // each 32.
            auto gamma_ramp_pwl_upload =
                reinterpret_cast<reg::DC_LUT_PWL_DATA*>(gamma_ramp_frame_upload);
            const reg::DC_LUT_PWL_DATA* gamma_ramp_pwl = gamma_ramp_pwl_rgb();
            for (size_t i = 0; i < 128 * 3; ++i) {
              reg::DC_LUT_PWL_DATA& gamma_ramp_pwl_upload_entry = gamma_ramp_pwl_upload[i];
              reg::DC_LUT_PWL_DATA gamma_ramp_pwl_entry = gamma_ramp_pwl[i];
              gamma_ramp_pwl_upload_entry.base = gamma_ramp_pwl_entry.delta;
              gamma_ramp_pwl_upload_entry.delta = gamma_ramp_pwl_entry.base;
            }
          } else {
            std::memcpy(gamma_ramp_frame_upload,
                        use_pwl_gamma_ramp ? static_cast<const void*>(gamma_ramp_pwl_rgb())
                                           : static_cast<const void*>(gamma_ramp_256_entry_table()),
                        gamma_ramp_size);
          }
          bool gamma_ramp_has_upload_buffer = gamma_ramp_upload_buffer_memory_ != VK_NULL_HANDLE;
          VkPipelineStageFlags gamma_ramp_read_stage_mask =
              VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
          ui::vulkan::util::FlushMappedMemoryRange(
              vulkan_device,
              gamma_ramp_has_upload_buffer ? gamma_ramp_upload_buffer_memory_
                                           : gamma_ramp_buffer_memory_,
              gamma_ramp_upload_memory_type_, gamma_ramp_upload_offset,
              gamma_ramp_upload_memory_size_, gamma_ramp_size);
          if (gamma_ramp_has_upload_buffer) {
            // Copy from the host-visible buffer to the device-local one.
            PushBufferMemoryBarrier(gamma_ramp_buffer_, gamma_ramp_offset_in_frame, gamma_ramp_size,
                                    gamma_ramp_read_stage_mask, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_ACCESS_SHADER_READ_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
            SubmitBarriers(true);
            VkBufferCopy gamma_ramp_buffer_copy;
            gamma_ramp_buffer_copy.srcOffset = gamma_ramp_upload_offset;
            gamma_ramp_buffer_copy.dstOffset = gamma_ramp_offset_in_frame;
            gamma_ramp_buffer_copy.size = gamma_ramp_size;
            deferred_command_buffer_.CmdVkCopyBuffer(gamma_ramp_upload_buffer_, gamma_ramp_buffer_,
                                                     1, &gamma_ramp_buffer_copy);
            PushBufferMemoryBarrier(gamma_ramp_buffer_, gamma_ramp_offset_in_frame, gamma_ramp_size,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT, gamma_ramp_read_stage_mask,
                                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
                                    VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
          }
          // The device-local, but not host-visible, gamma ramp buffer doesn't
          // have per-frame sets of gamma ramps.
          gamma_ramp_frame_index_ref = gamma_ramp_has_upload_buffer ? 0 : swap_frame_index;
        }

        VkDescriptorSet swap_descriptor_source = swap_descriptors_source_[swap_frame_index];
        VkDescriptorImageInfo swap_descriptor_source_image_info;
        swap_descriptor_source_image_info.sampler = VK_NULL_HANDLE;
        swap_descriptor_source_image_info.imageView = swap_texture_view;
        swap_descriptor_source_image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        VkWriteDescriptorSet swap_descriptor_source_write;
        swap_descriptor_source_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        swap_descriptor_source_write.pNext = nullptr;
        swap_descriptor_source_write.dstSet = swap_descriptor_source;
        swap_descriptor_source_write.dstBinding = 0;
        swap_descriptor_source_write.dstArrayElement = 0;
        swap_descriptor_source_write.descriptorCount = 1;
        swap_descriptor_source_write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
        swap_descriptor_source_write.pImageInfo = &swap_descriptor_source_image_info;
        swap_descriptor_source_write.pBufferInfo = nullptr;
        swap_descriptor_source_write.pTexelBufferView = nullptr;
        dfn.vkUpdateDescriptorSets(device, 1, &swap_descriptor_source_write, 0, nullptr);

        VkImageSubresourceRange guest_output_subresource_range =
            ui::vulkan::util::InitializeSubresourceRange();
        if (use_compute_gamma) {
          // Transition the destination image for compute writes. Contents are
          // fully overwritten, so old layout can always be UNDEFINED.
          PushImageMemoryBarrier(vulkan_context.image(), guest_output_subresource_range,
                                 vulkan_context.image_ever_written_previously()
                                     ? ui::vulkan::VulkanPresenter::kGuestOutputInternalStageMask
                                     : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 vulkan_context.image_ever_written_previously()
                                     ? ui::vulkan::VulkanPresenter::kGuestOutputInternalAccessMask
                                     : 0,
                                 VK_ACCESS_SHADER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                 VK_IMAGE_LAYOUT_GENERAL);
          if (use_fxaa) {
            PushImageMemoryBarrier(swap_fxaa_source_image_, guest_output_subresource_range,
                                   swap_fxaa_source_stage_mask_ ? swap_fxaa_source_stage_mask_
                                                                : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   swap_fxaa_source_access_mask_, VK_ACCESS_SHADER_WRITE_BIT,
                                   swap_fxaa_source_layout_, VK_IMAGE_LAYOUT_GENERAL);
          }
          SubmitBarriers(true);

          if (use_fxaa) {
            swap_fxaa_source_stage_mask_ = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            swap_fxaa_source_access_mask_ = VK_ACCESS_SHADER_WRITE_BIT;
            swap_fxaa_source_layout_ = VK_IMAGE_LAYOUT_GENERAL;
            swap_fxaa_source_image_submission_ = GetCurrentSubmission();
          }

          VkDescriptorSet swap_descriptor_destination_storage =
              swap_descriptors_destination_storage_[swap_frame_index];
          VkDescriptorImageInfo swap_descriptor_destination_storage_image_info;
          swap_descriptor_destination_storage_image_info.sampler = VK_NULL_HANDLE;
          swap_descriptor_destination_storage_image_info.imageView =
              use_fxaa ? swap_fxaa_source_image_view_ : vulkan_context.image_view();
          swap_descriptor_destination_storage_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
          VkWriteDescriptorSet swap_descriptor_destination_storage_write;
          swap_descriptor_destination_storage_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
          swap_descriptor_destination_storage_write.pNext = nullptr;
          swap_descriptor_destination_storage_write.dstSet = swap_descriptor_destination_storage;
          swap_descriptor_destination_storage_write.dstBinding = 0;
          swap_descriptor_destination_storage_write.dstArrayElement = 0;
          swap_descriptor_destination_storage_write.descriptorCount = 1;
          swap_descriptor_destination_storage_write.descriptorType =
              VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
          swap_descriptor_destination_storage_write.pImageInfo =
              &swap_descriptor_destination_storage_image_info;
          swap_descriptor_destination_storage_write.pBufferInfo = nullptr;
          swap_descriptor_destination_storage_write.pTexelBufferView = nullptr;
          dfn.vkUpdateDescriptorSets(device, 1, &swap_descriptor_destination_storage_write, 0,
                                     nullptr);

          std::array<VkDescriptorSet, kSwapApplyGammaComputeDescriptorSetCount>
              swap_apply_gamma_compute_descriptor_sets{};
          swap_apply_gamma_compute_descriptor_sets[kSwapApplyGammaComputeDescriptorSetRamp] =
              swap_descriptors_gamma_ramp_[2 * gamma_ramp_frame_index_ref +
                                           uint32_t(use_pwl_gamma_ramp)];
          swap_apply_gamma_compute_descriptor_sets[kSwapApplyGammaComputeDescriptorSetSource] =
              swap_descriptor_source;
          swap_apply_gamma_compute_descriptor_sets[kSwapApplyGammaComputeDescriptorSetDestination] =
              swap_descriptor_destination_storage;
          deferred_command_buffer_.CmdVkBindDescriptorSets(
              VK_PIPELINE_BIND_POINT_COMPUTE, swap_apply_gamma_compute_pipeline_layout_, 0,
              uint32_t(swap_apply_gamma_compute_descriptor_sets.size()),
              swap_apply_gamma_compute_descriptor_sets.data(), 0, nullptr);
          SwapApplyGammaConstants swap_apply_gamma_constants = {
              {guest_output_width, guest_output_height}};
          deferred_command_buffer_.CmdVkPushConstants(
              swap_apply_gamma_compute_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
              sizeof(swap_apply_gamma_constants), &swap_apply_gamma_constants);
          BindExternalComputePipeline(swap_apply_gamma_compute_pipeline);
          uint32_t group_count_x = (guest_output_width + 15) / 16;
          uint32_t group_count_y = (guest_output_height + 7) / 8;
          uint32_t gamma_scope = BeginScopeProfile(
              ScopeCategory::kCompute, guest_output_width, guest_output_height, "GAMMA",
              VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
              VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
              "swap_gamma");
          deferred_command_buffer_.CmdVkDispatch(group_count_x, group_count_y, 1, WorkPurpose::kSwap);
          EndScopeProfile(gamma_scope);

          if (use_fxaa) {
            // Make the FXAA source image readable and bind a separate
            // destination storage descriptor targeting the guest output image.
            PushImageMemoryBarrier(swap_fxaa_source_image_, guest_output_subresource_range,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                   VK_ACCESS_SHADER_READ_BIT, VK_IMAGE_LAYOUT_GENERAL,
                                   VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            SubmitBarriers(true);
            swap_fxaa_source_stage_mask_ = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            swap_fxaa_source_access_mask_ = VK_ACCESS_SHADER_READ_BIT;
            swap_fxaa_source_layout_ = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            swap_fxaa_source_image_submission_ = GetCurrentSubmission();

            VkDescriptorSet swap_descriptor_fxaa_source =
                swap_descriptors_fxaa_source_[swap_frame_index];
            VkDescriptorImageInfo swap_descriptor_fxaa_source_image_info;
            swap_descriptor_fxaa_source_image_info.sampler = swap_sampler_linear_clamp_;
            swap_descriptor_fxaa_source_image_info.imageView = swap_fxaa_source_image_view_;
            swap_descriptor_fxaa_source_image_info.imageLayout =
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            VkWriteDescriptorSet swap_descriptor_fxaa_source_write;
            swap_descriptor_fxaa_source_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            swap_descriptor_fxaa_source_write.pNext = nullptr;
            swap_descriptor_fxaa_source_write.dstSet = swap_descriptor_fxaa_source;
            swap_descriptor_fxaa_source_write.dstBinding = 0;
            swap_descriptor_fxaa_source_write.dstArrayElement = 0;
            swap_descriptor_fxaa_source_write.descriptorCount = 1;
            swap_descriptor_fxaa_source_write.descriptorType =
                VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            swap_descriptor_fxaa_source_write.pImageInfo = &swap_descriptor_fxaa_source_image_info;
            swap_descriptor_fxaa_source_write.pBufferInfo = nullptr;
            swap_descriptor_fxaa_source_write.pTexelBufferView = nullptr;
            dfn.vkUpdateDescriptorSets(device, 1, &swap_descriptor_fxaa_source_write, 0, nullptr);

            VkDescriptorSet swap_descriptor_fxaa_destination_storage =
                swap_descriptors_fxaa_destination_storage_[swap_frame_index];
            VkDescriptorImageInfo swap_descriptor_fxaa_destination_storage_image_info;
            swap_descriptor_fxaa_destination_storage_image_info.sampler = VK_NULL_HANDLE;
            swap_descriptor_fxaa_destination_storage_image_info.imageView =
                vulkan_context.image_view();
            swap_descriptor_fxaa_destination_storage_image_info.imageLayout =
                VK_IMAGE_LAYOUT_GENERAL;
            VkWriteDescriptorSet swap_descriptor_fxaa_destination_storage_write =
                swap_descriptor_destination_storage_write;
            swap_descriptor_fxaa_destination_storage_write.dstSet =
                swap_descriptor_fxaa_destination_storage;
            swap_descriptor_fxaa_destination_storage_write.pImageInfo =
                &swap_descriptor_fxaa_destination_storage_image_info;
            dfn.vkUpdateDescriptorSets(device, 1, &swap_descriptor_fxaa_destination_storage_write,
                                       0, nullptr);

            std::array<VkDescriptorSet, kSwapFxaaDescriptorSetCount> swap_fxaa_descriptor_sets{};
            swap_fxaa_descriptor_sets[kSwapFxaaDescriptorSetSource] = swap_descriptor_fxaa_source;
            swap_fxaa_descriptor_sets[kSwapFxaaDescriptorSetDestination] =
                swap_descriptor_fxaa_destination_storage;
            deferred_command_buffer_.CmdVkBindDescriptorSets(
                VK_PIPELINE_BIND_POINT_COMPUTE, swap_fxaa_pipeline_layout_, 0,
                uint32_t(swap_fxaa_descriptor_sets.size()), swap_fxaa_descriptor_sets.data(), 0,
                nullptr);
            SwapFxaaConstants swap_fxaa_constants = {
                {guest_output_width, guest_output_height},
                {1.0f / float(guest_output_width), 1.0f / float(guest_output_height)}};
            deferred_command_buffer_.CmdVkPushConstants(
                swap_fxaa_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                sizeof(swap_fxaa_constants), &swap_fxaa_constants);
            BindExternalComputePipeline(swap_post_effect == SwapPostEffect::kFxaaExtreme
                                            ? swap_fxaa_extreme_pipeline_
                                            : swap_fxaa_pipeline_);
            uint32_t fxaa_scope = BeginScopeProfile(
                ScopeCategory::kCompute, guest_output_width, guest_output_height, "FXAA",
                VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
                VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
                "swap_fxaa");
            deferred_command_buffer_.CmdVkDispatch(group_count_x, group_count_y, 1, WorkPurpose::kFxaa);
            EndScopeProfile(fxaa_scope);
          }

          // Insert the release barrier.
          PushImageMemoryBarrier(vulkan_context.image(), guest_output_subresource_range,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 ui::vulkan::VulkanPresenter::kGuestOutputInternalStageMask,
                                 VK_ACCESS_SHADER_WRITE_BIT,
                                 ui::vulkan::VulkanPresenter::kGuestOutputInternalAccessMask,
                                 VK_IMAGE_LAYOUT_GENERAL,
                                 ui::vulkan::VulkanPresenter::kGuestOutputInternalLayout);
        } else {
          // Make sure a framebuffer is available for the current guest output
          // image version.
          size_t swap_framebuffer_index = SIZE_MAX;
          size_t swap_framebuffer_new_index = SIZE_MAX;
          // Try to find the existing framebuffer for the current guest output
          // image version, or an unused (without an existing framebuffer, or
          // with one, but that has never actually been used dynamically) slot.
          for (size_t i = 0; i < swap_framebuffers_.size(); ++i) {
            const SwapFramebuffer& existing_swap_framebuffer = swap_framebuffers_[i];
            if (existing_swap_framebuffer.framebuffer != VK_NULL_HANDLE &&
                existing_swap_framebuffer.version == guest_output_image_version) {
              swap_framebuffer_index = i;
              break;
            }
            if (existing_swap_framebuffer.framebuffer == VK_NULL_HANDLE ||
                !existing_swap_framebuffer.last_submission) {
              swap_framebuffer_new_index = i;
            }
          }
          if (swap_framebuffer_index == SIZE_MAX) {
            if (swap_framebuffer_new_index == SIZE_MAX) {
              // Replace the earliest used framebuffer.
              swap_framebuffer_new_index = 0;
              for (size_t i = 1; i < swap_framebuffers_.size(); ++i) {
                if (swap_framebuffers_[i].last_submission <
                    swap_framebuffers_[swap_framebuffer_new_index].last_submission) {
                  swap_framebuffer_new_index = i;
                }
              }
            }
            swap_framebuffer_index = swap_framebuffer_new_index;
            SwapFramebuffer& new_swap_framebuffer = swap_framebuffers_[swap_framebuffer_new_index];
            if (new_swap_framebuffer.framebuffer != VK_NULL_HANDLE) {
              if (submission_completed_ >= new_swap_framebuffer.last_submission) {
                dfn.vkDestroyFramebuffer(device, new_swap_framebuffer.framebuffer, nullptr);
              } else {
                destroy_framebuffers_.emplace_back(new_swap_framebuffer.last_submission,
                                                   new_swap_framebuffer.framebuffer);
              }
              new_swap_framebuffer.framebuffer = VK_NULL_HANDLE;
            }
            VkImageView guest_output_image_view = vulkan_context.image_view();
            VkFramebufferCreateInfo swap_framebuffer_create_info;
            swap_framebuffer_create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            swap_framebuffer_create_info.pNext = nullptr;
            swap_framebuffer_create_info.flags = 0;
            swap_framebuffer_create_info.renderPass = swap_apply_gamma_render_pass_;
            swap_framebuffer_create_info.attachmentCount = 1;
            swap_framebuffer_create_info.pAttachments = &guest_output_image_view;
            swap_framebuffer_create_info.width = guest_output_width;
            swap_framebuffer_create_info.height = guest_output_height;
            swap_framebuffer_create_info.layers = 1;
            if (dfn.vkCreateFramebuffer(device, &swap_framebuffer_create_info, nullptr,
                                        &new_swap_framebuffer.framebuffer) != VK_SUCCESS) {
              REXGPU_ERROR("Failed to create the Vulkan framebuffer for presentation");
              return false;
            }
            new_swap_framebuffer.version = guest_output_image_version;
            // The actual submission index will be set if the framebuffer is
            // actually used, not dropped due to some error.
            new_swap_framebuffer.last_submission = 0;
          }

          if (vulkan_context.image_ever_written_previously()) {
            // Insert a barrier after the last presenter's usage of the guest
            // output image. Will be overwriting all the contents, so oldLayout
            // layout is UNDEFINED. The render pass will do the layout
            // transition, but newLayout must not be UNDEFINED.
            PushImageMemoryBarrier(vulkan_context.image(), guest_output_subresource_range,
                                   ui::vulkan::VulkanPresenter::kGuestOutputInternalStageMask,
                                   VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                   ui::vulkan::VulkanPresenter::kGuestOutputInternalAccessMask,
                                   VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
                                   VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
          }

          // End the current render pass before inserting barriers and starting
          // a new one, and insert the barrier.
          SubmitBarriers(true);

          SwapFramebuffer& swap_framebuffer = swap_framebuffers_[swap_framebuffer_index];
          swap_framebuffer.last_submission = GetCurrentSubmission();

          VkRenderPassBeginInfo render_pass_begin_info;
          render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
          render_pass_begin_info.pNext = nullptr;
          render_pass_begin_info.renderPass = swap_apply_gamma_render_pass_;
          render_pass_begin_info.framebuffer = swap_framebuffer.framebuffer;
          render_pass_begin_info.renderArea.offset.x = 0;
          render_pass_begin_info.renderArea.offset.y = 0;
          render_pass_begin_info.renderArea.extent.width = guest_output_width;
          render_pass_begin_info.renderArea.extent.height = guest_output_height;
          render_pass_begin_info.clearValueCount = 0;
          render_pass_begin_info.pClearValues = nullptr;
          deferred_command_buffer_.CmdVkBeginRenderPass(&render_pass_begin_info,
                                                        VK_SUBPASS_CONTENTS_INLINE);

          VkViewport viewport;
          viewport.x = 0.0f;
          viewport.y = 0.0f;
          viewport.width = float(guest_output_width);
          viewport.height = float(guest_output_height);
          viewport.minDepth = 0.0f;
          viewport.maxDepth = 1.0f;
          SetViewport(viewport);
          VkRect2D scissor;
          scissor.offset.x = 0;
          scissor.offset.y = 0;
          scissor.extent.width = guest_output_width;
          scissor.extent.height = guest_output_height;
          SetScissor(scissor);

          VkPipeline swap_apply_gamma_pipeline = use_pwl_gamma_ramp
                                                     ? swap_apply_gamma_pwl_pipeline_
                                                     : swap_apply_gamma_256_entry_table_pipeline_;
          if (!vulkan_device->properties().imageViewFormatSwizzle && swap_source_needs_rb_swap) {
            VkPipeline swap_apply_gamma_rb_swap_pipeline =
                use_pwl_gamma_ramp ? swap_apply_gamma_pwl_rb_swap_pipeline_
                                   : swap_apply_gamma_256_entry_table_rb_swap_pipeline_;
            if (swap_apply_gamma_rb_swap_pipeline != VK_NULL_HANDLE) {
              swap_apply_gamma_pipeline = swap_apply_gamma_rb_swap_pipeline;
            } else {
              static bool swap_rb_swap_fallback_missing_logged = false;
              if (!swap_rb_swap_fallback_missing_logged) {
                swap_rb_swap_fallback_missing_logged = true;
                REXGPU_WARN(
                    "Vulkan imageViewFormatSwizzle is unavailable and the "
                    "swap source needs red/blue swizzle, but the graphics "
                    "fallback RB-swap shader pipeline is unavailable");
              }
            }
          }
          BindExternalGraphicsPipeline(swap_apply_gamma_pipeline);

          std::array<VkDescriptorSet, kSwapApplyGammaDescriptorSetCount> swap_descriptor_sets{};
          swap_descriptor_sets[kSwapApplyGammaDescriptorSetRamp] =
              swap_descriptors_gamma_ramp_[2 * gamma_ramp_frame_index_ref +
                                           uint32_t(use_pwl_gamma_ramp)];
          swap_descriptor_sets[kSwapApplyGammaDescriptorSetSource] = swap_descriptor_source;
          deferred_command_buffer_.CmdVkBindDescriptorSets(
              VK_PIPELINE_BIND_POINT_GRAPHICS, swap_apply_gamma_pipeline_layout_, 0,
              uint32_t(swap_descriptor_sets.size()), swap_descriptor_sets.data(), 0, nullptr);

          deferred_command_buffer_.CmdVkDraw(3, 1, 0, 0);
          deferred_command_buffer_.CmdVkEndRenderPass();

          // Insert the release barrier.
          PushImageMemoryBarrier(vulkan_context.image(), guest_output_subresource_range,
                                 VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                                 ui::vulkan::VulkanPresenter::kGuestOutputInternalStageMask,
                                 VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT,
                                 ui::vulkan::VulkanPresenter::kGuestOutputInternalAccessMask,
                                 VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                 ui::vulkan::VulkanPresenter::kGuestOutputInternalLayout);
        }

        // Need to submit all the commands before giving the image back to the
        // presenter so it can submit its own commands for displaying it to the
        // queue, and also need to submit the release barrier.
        EndSubmission(true);
        return true;
      });

  if (provenance_frame_sampled) {
    REXGPU_INFO(
        "GPU provenance frame {} swap slot0_guest_base=0x{:08X} frontbuffer_ptr=0x{:08X} "
        "image_view={:p} size={}x{} format={} output_refreshed={}",
        provenance_frame, swap_source_guest_base, frontbuffer_ptr,
        static_cast<void*>(swap_texture_view), frontbuffer_width_scaled, frontbuffer_height_scaled,
        uint32_t(frontbuffer_format), guest_output_refreshed);
    if (guest_output_refreshed) {
      ui::RawImage captured_output;
      if (presenter->CaptureGuestOutput(captured_output)) {
        if (FlickerDiagEnabled()) {
          // [flicker-diag] Readback-race check: re-read the same guest output
          // image after the device is fully idle. Compared BEFORE
          // LogGpuProvenanceOutput stamps the FPS overlay into captured_output,
          // so any difference is a genuine timing difference, not the overlay.
          // A non-zero result here would mean the PPM for that frame was
          // produced from incompletely submitted work.
          AwaitAllQueueOperationsCompletion();
          ui::RawImage reread_output;
          if (presenter->CaptureGuestOutput(reread_output)) {
            size_t compare_size = std::min(captured_output.data.size(), reread_output.data.size());
            size_t differ_bytes = 0;
            uint32_t max_delta = 0;
            for (size_t i = 0; i < compare_size; ++i) {
              uint8_t a = captured_output.data[i];
              uint8_t b = reread_output.data[i];
              if (a != b) {
                ++differ_bytes;
                uint32_t delta = a > b ? uint32_t(a - b) : uint32_t(b - a);
                if (delta > max_delta) {
                  max_delta = delta;
                }
              }
            }
            REXGPU_INFO(
                "[flicker-diag] reread frame {} compared={} differ_bytes={} max_delta={} "
                "identical={}",
                provenance_frame, uint64_t(compare_size), uint64_t(differ_bytes), max_delta,
                differ_bytes == 0 && captured_output.data.size() == reread_output.data.size());
          } else {
            REXGPU_WARN("[flicker-diag] reread frame {} capture failed", provenance_frame);
          }
        }
        LogGpuProvenanceOutput(provenance_frame, captured_output,
                               guest_frame_stats_.fps.load(std::memory_order_relaxed),
                               guest_frame_stats_.frame_time_ms.load(std::memory_order_relaxed));
      } else {
        REXGPU_WARN("GPU provenance frame {} guest output capture failed", provenance_frame);
      }
    }
  }

  // End the frame even if did not present for any reason (the image refresher
  // was not called), to prevent leaking per-frame resources.
  EndSubmission(true);
}

bool VulkanCommandProcessor::PushBufferMemoryBarrier(
    VkBuffer buffer, VkDeviceSize offset, VkDeviceSize size, VkPipelineStageFlags src_stage_mask,
    VkPipelineStageFlags dst_stage_mask, VkAccessFlags src_access_mask,
    VkAccessFlags dst_access_mask, uint32_t src_queue_family_index, uint32_t dst_queue_family_index,
    bool skip_if_equal, BarrierRequester requester) {
  if (skip_if_equal && src_stage_mask == dst_stage_mask && src_access_mask == dst_access_mask &&
      src_queue_family_index == dst_queue_family_index) {
    return false;
  }

  // Separate different barriers for overlapping buffer ranges into different
  // pipeline barrier commands.
  for (const VkBufferMemoryBarrier& other_buffer_memory_barrier :
       pending_barriers_buffer_memory_barriers_) {
    if (other_buffer_memory_barrier.buffer != buffer ||
        (size != VK_WHOLE_SIZE && offset + size <= other_buffer_memory_barrier.offset) ||
        (other_buffer_memory_barrier.size != VK_WHOLE_SIZE &&
         other_buffer_memory_barrier.offset + other_buffer_memory_barrier.size <= offset)) {
      continue;
    }
    if (other_buffer_memory_barrier.offset == offset && other_buffer_memory_barrier.size == size &&
        other_buffer_memory_barrier.srcAccessMask == src_access_mask &&
        other_buffer_memory_barrier.dstAccessMask == dst_access_mask &&
        other_buffer_memory_barrier.srcQueueFamilyIndex == src_queue_family_index &&
        other_buffer_memory_barrier.dstQueueFamilyIndex == dst_queue_family_index) {
      // The barrier is already pending.
      current_pending_barrier_.src_stage_mask |= src_stage_mask;
      current_pending_barrier_.dst_stage_mask |= dst_stage_mask;
      return true;
    }
    SplitPendingBarrier();
    break;
  }

  current_pending_barrier_.src_stage_mask |= src_stage_mask;
  current_pending_barrier_.dst_stage_mask |= dst_stage_mask;
  VkBufferMemoryBarrier& buffer_memory_barrier =
      pending_barriers_buffer_memory_barriers_.emplace_back();
  buffer_memory_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
  buffer_memory_barrier.pNext = nullptr;
  buffer_memory_barrier.srcAccessMask = src_access_mask;
  buffer_memory_barrier.dstAccessMask = dst_access_mask;
  buffer_memory_barrier.srcQueueFamilyIndex = src_queue_family_index;
  buffer_memory_barrier.dstQueueFamilyIndex = dst_queue_family_index;
  buffer_memory_barrier.buffer = buffer;
  buffer_memory_barrier.offset = offset;
  buffer_memory_barrier.size = size;
  pending_buffer_barrier_requesters_.push_back(requester);
  return true;
}

bool VulkanCommandProcessor::PushImageMemoryBarrier(
    VkImage image, const VkImageSubresourceRange& subresource_range,
    VkPipelineStageFlags src_stage_mask, VkPipelineStageFlags dst_stage_mask,
    VkAccessFlags src_access_mask, VkAccessFlags dst_access_mask, VkImageLayout old_layout,
    VkImageLayout new_layout, uint32_t src_queue_family_index, uint32_t dst_queue_family_index,
    bool skip_if_equal, BarrierRequester requester) {
  if (skip_if_equal && src_stage_mask == dst_stage_mask && src_access_mask == dst_access_mask &&
      old_layout == new_layout && src_queue_family_index == dst_queue_family_index) {
    return false;
  }

  // Separate different barriers for overlapping image subresource ranges into
  // different pipeline barrier commands.
  for (const VkImageMemoryBarrier& other_image_memory_barrier :
       pending_barriers_image_memory_barriers_) {
    if (other_image_memory_barrier.image != image ||
        !(other_image_memory_barrier.subresourceRange.aspectMask & subresource_range.aspectMask) ||
        (subresource_range.levelCount != VK_REMAINING_MIP_LEVELS &&
         subresource_range.baseMipLevel + subresource_range.levelCount <=
             other_image_memory_barrier.subresourceRange.baseMipLevel) ||
        (other_image_memory_barrier.subresourceRange.levelCount != VK_REMAINING_MIP_LEVELS &&
         other_image_memory_barrier.subresourceRange.baseMipLevel +
                 other_image_memory_barrier.subresourceRange.levelCount <=
             subresource_range.baseMipLevel) ||
        (subresource_range.layerCount != VK_REMAINING_ARRAY_LAYERS &&
         subresource_range.baseArrayLayer + subresource_range.layerCount <=
             other_image_memory_barrier.subresourceRange.baseArrayLayer) ||
        (other_image_memory_barrier.subresourceRange.layerCount != VK_REMAINING_ARRAY_LAYERS &&
         other_image_memory_barrier.subresourceRange.baseArrayLayer +
                 other_image_memory_barrier.subresourceRange.layerCount <=
             subresource_range.baseArrayLayer)) {
      continue;
    }
    if (other_image_memory_barrier.subresourceRange.aspectMask == subresource_range.aspectMask &&
        other_image_memory_barrier.subresourceRange.baseMipLevel ==
            subresource_range.baseMipLevel &&
        other_image_memory_barrier.subresourceRange.levelCount == subresource_range.levelCount &&
        other_image_memory_barrier.subresourceRange.baseArrayLayer ==
            subresource_range.baseArrayLayer &&
        other_image_memory_barrier.subresourceRange.layerCount == subresource_range.layerCount &&
        other_image_memory_barrier.srcAccessMask == src_access_mask &&
        other_image_memory_barrier.dstAccessMask == dst_access_mask &&
        other_image_memory_barrier.oldLayout == old_layout &&
        other_image_memory_barrier.newLayout == new_layout &&
        other_image_memory_barrier.srcQueueFamilyIndex == src_queue_family_index &&
        other_image_memory_barrier.dstQueueFamilyIndex == dst_queue_family_index) {
      // The barrier is already pending.
      current_pending_barrier_.src_stage_mask |= src_stage_mask;
      current_pending_barrier_.dst_stage_mask |= dst_stage_mask;
      return true;
    }
    SplitPendingBarrier();
    break;
  }

  current_pending_barrier_.src_stage_mask |= src_stage_mask;
  current_pending_barrier_.dst_stage_mask |= dst_stage_mask;
  VkImageMemoryBarrier& image_memory_barrier =
      pending_barriers_image_memory_barriers_.emplace_back();
  image_memory_barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  image_memory_barrier.pNext = nullptr;
  image_memory_barrier.srcAccessMask = src_access_mask;
  image_memory_barrier.dstAccessMask = dst_access_mask;
  image_memory_barrier.oldLayout = old_layout;
  image_memory_barrier.newLayout = new_layout;
  image_memory_barrier.srcQueueFamilyIndex = src_queue_family_index;
  image_memory_barrier.dstQueueFamilyIndex = dst_queue_family_index;
  image_memory_barrier.image = image;
  image_memory_barrier.subresourceRange = subresource_range;
  pending_image_barrier_requesters_.push_back(requester);
  return true;
}

bool VulkanCommandProcessor::SubmitBarriers(bool force_end_render_pass) {
  assert_true(submission_open_);
  SplitPendingBarrier();
  if (pending_barriers_.empty()) {
    if (force_end_render_pass) {
      if (in_render_pass_ && REXCVAR_GET(gpu_frame_audit_enable)) ++frame_cpu_profiling_stats_.break_force;
      EndRenderPass();
    }
    return false;
  }

  const ui::vulkan::VulkanDevice* vulkan_device = GetVulkanDevice();
  const bool use_dynamic_rendering =
      REXCVAR_GET(vulkan_dynamic_rendering) && vulkan_device->properties().dynamicRendering;

  bool must_end_render_pass = force_end_render_pass || !use_dynamic_rendering;
  if (!must_end_render_pass && in_render_pass_) {
    if (REXCVAR_GET(vulkan_defer_non_conflicting_barriers)) {
      for (const auto& image_barrier : pending_barriers_image_memory_barriers_) {
        if (render_target_cache_->IsImageActiveAttachment(image_barrier.image)) {
          must_end_render_pass = true;
          break;
        }
      }
    } else {
      must_end_render_pass = true;
    }
  }

  if (in_render_pass_ && must_end_render_pass) {
    if (REXCVAR_GET(gpu_frame_audit_enable)) {
      ++frame_cpu_profiling_stats_.break_barrier;
      RecordBarrierBreakAttribution();
    }
    EndRenderPass();
  }
  for (auto it = pending_barriers_.cbegin(); it != pending_barriers_.cend(); ++it) {
    auto it_next = std::next(it);
    bool is_last = it_next == pending_barriers_.cend();
    // .data() + offset, not &[offset], for buffer and image barriers, because
    // if there are no buffer or image memory barriers in the last pipeline
    // barriers, the offsets may be equal to the sizes of the vectors.
    deferred_command_buffer_.CmdVkPipelineBarrier(
        it->src_stage_mask ? it->src_stage_mask : VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
        it->dst_stage_mask ? it->dst_stage_mask : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0,
        nullptr,
        uint32_t((is_last ? pending_barriers_buffer_memory_barriers_.size()
                          : it_next->buffer_memory_barriers_offset) -
                 it->buffer_memory_barriers_offset),
        pending_barriers_buffer_memory_barriers_.data() + it->buffer_memory_barriers_offset,
        uint32_t((is_last ? pending_barriers_image_memory_barriers_.size()
                          : it_next->image_memory_barriers_offset) -
                 it->image_memory_barriers_offset),
        pending_barriers_image_memory_barriers_.data() + it->image_memory_barriers_offset);
  }
  pending_barriers_.clear();
  pending_barriers_buffer_memory_barriers_.clear();
  pending_barriers_image_memory_barriers_.clear();
  pending_buffer_barrier_requesters_.clear();
  pending_image_barrier_requesters_.clear();
  current_pending_barrier_.buffer_memory_barriers_offset = 0;
  current_pending_barrier_.image_memory_barriers_offset = 0;
  return true;
}

void VulkanCommandProcessor::SubmitBarriersAndEnterRenderTargetCacheRenderPass(
    VkRenderPass render_pass, const VulkanRenderTargetCache::Framebuffer* framebuffer) {
  if (render_target_cache_->HasPendingDeferredClear()) {
    if (!render_target_cache_->CanEmitDeferredClearInFramebuffer(framebuffer)) {
      render_target_cache_->FlushPendingDeferredClear();
    }
  }
  SubmitBarriers(false);
  const ui::vulkan::VulkanDevice* vulkan_device = GetVulkanDevice();
  bool use_dynamic_rendering =
      REXCVAR_GET(vulkan_dynamic_rendering) && vulkan_device->properties().dynamicRendering;

  reg::RB_DEPTHCONTROL normalized_depth_control =
      draw_util::GetNormalizedDepthControl(*register_file_);
  const auto rpk = render_target_cache_->last_update_render_pass_key();
  bool has_depth = (rpk.depth_and_color_used & 0b1) != 0;
  bool needs_stencil =
      has_depth && normalized_depth_control.stencil_enable;
  uint32_t expected_color_count =
      32 - rex::lzcnt(uint32_t(rpk.depth_and_color_used >> 1));
  const bool pending_clear = render_target_cache_->HasAnyPendingClear();
  // UpdateDynamicState has already recorded this draw's host scissor. Never
  // shrink clear load operations, and never coalesce a draw outside the active
  // area: attachment loads/stores are restricted to VkRenderingInfo::renderArea.
  const VkRect2D render_area = DrawRenderArea(
      dynamic_scissor_, framebuffer->host_extent,
      use_dynamic_rendering &&
          render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets &&
          REXCVAR_GET(vulkan_scissor_render_area), pending_clear);

  if (use_dynamic_rendering) {
    if (REXCVAR_GET(vulkan_reuse_transfer_scope_for_guest) && in_render_pass_ &&
        current_dynamic_rendering_is_transfer_ && current_framebuffer_ &&
        current_render_pass_ == VK_NULL_HANDLE && !pending_clear &&
        RenderAreaContains(current_dynamic_rendering_area_, render_area) &&
        CanReuseTransferAttachment(
            rpk.depth_and_color_used, current_dynamic_rendering_transfer_dest_is_depth_,
            current_dynamic_rendering_transfer_dest_view_,
            render_target_cache_->GetLastUpdateDrawAttachmentView(
                current_dynamic_rendering_transfer_dest_is_depth_ ? 0 : 1),
            current_framebuffer_->host_extent, framebuffer->host_extent,
            current_dynamic_rendering_has_depth_,
            current_dynamic_rendering_transfer_depth_view_,
            render_target_cache_->GetLastUpdateDrawAttachmentView(0))) {
      // Same image view, aspects, extent and sample count as the active scope.
      // SubmitBarriers above has already ended it if an intervening dependency
      // required a break. Keep attachment load/store and all transfer commands.
      current_framebuffer_ = framebuffer;
      current_dynamic_rendering_is_transfer_ = false;
      current_dynamic_rendering_transfer_dest_view_ = VK_NULL_HANDLE;
      current_dynamic_rendering_transfer_depth_view_ = VK_NULL_HANDLE;
      current_dynamic_rendering_transfer_dest_is_depth_ = false;
      current_dynamic_rendering_render_pass_key_ = rpk;
      if (REXCVAR_GET(gpu_frame_audit_enable)) {
        ++frame_cpu_profiling_stats_.transfer_guest_scope_reuses;
      }
      if (REXCVAR_GET(vulkan_coalesce_adjacent_clears) && render_target_cache_->HasPendingDeferredClear()) {
        render_target_cache_->TryEmitDeferredClearInDrawPass(framebuffer);
      }
      return;
    }
    if (in_render_pass_ && !current_dynamic_rendering_is_transfer_ &&
        current_framebuffer_ == framebuffer &&
        current_render_pass_ == VK_NULL_HANDLE &&
        (!has_depth || current_dynamic_rendering_has_depth_) &&
        (!needs_stencil || current_dynamic_rendering_has_stencil_) &&
        (expected_color_count <= current_dynamic_rendering_color_count_) &&
        !pending_clear) {
      if (REXCVAR_GET(vulkan_coalesce_adjacent_clears) && render_target_cache_->HasPendingDeferredClear()) {
        render_target_cache_->TryEmitDeferredClearInDrawPass(framebuffer);
      }
      if (REXCVAR_GET(vulkan_merge_matching_attachment_passes)) {
        frame_pass_stats_.merged_guest_passes++;
        return;
      } else if (RenderAreaContains(current_dynamic_rendering_area_, render_area)) {
        return;
      }
    }
  } else {
    if (current_render_pass_ == render_pass && current_framebuffer_ == framebuffer) {
      return;
    }
  }

  if (use_dynamic_rendering && in_render_pass_ && REXCVAR_GET(gpu_frame_audit_enable)) {
    // Reasons may overlap; they describe failed reuse predicates, not disjoint breaks.
    auto& stats = frame_cpu_profiling_stats_;
    stats.break_transfer += current_dynamic_rendering_is_transfer_;
    stats.break_framebuffer += current_framebuffer_ != framebuffer;
    stats.break_depth += has_depth && !current_dynamic_rendering_has_depth_;
    stats.break_stencil += needs_stencil && !current_dynamic_rendering_has_stencil_;
    stats.break_color += expected_color_count > current_dynamic_rendering_color_count_;
    stats.break_clear += pending_clear;
    stats.break_area += !RenderAreaContains(current_dynamic_rendering_area_, render_area);
    const bool growth = (has_depth && !current_dynamic_rendering_has_depth_) ||
                        (needs_stencil && !current_dynamic_rendering_has_stencil_) ||
                        expected_color_count > current_dynamic_rendering_color_count_;
    const bool framebuffer_changed = current_framebuffer_ != framebuffer;
    const bool other_reuse_failure = current_dynamic_rendering_is_transfer_ || pending_clear ||
        !RenderAreaContains(current_dynamic_rendering_area_, render_area) ||
        current_render_pass_ != VK_NULL_HANDLE;
    stats.break_growth_only += growth && !framebuffer_changed && !other_reuse_failure;
    stats.break_growth_framebuffer += growth && framebuffer_changed;
    stats.break_framebuffer_only += framebuffer_changed && !growth && !other_reuse_failure;
  }
  const bool expanding_render_area =
      use_dynamic_rendering && in_render_pass_ &&
      !current_dynamic_rendering_is_transfer_ && current_framebuffer_ == framebuffer &&
      !RenderAreaContains(current_dynamic_rendering_area_, render_area);
  if (in_render_pass_) {
    if (use_dynamic_rendering) {
      deferred_command_buffer_.CmdVkEndRendering();
    } else {
      deferred_command_buffer_.CmdVkEndRenderPass();
    }
    EndScopeProfile(active_scope_index_);
    in_render_pass_ = false;
  }

  if (expanding_render_area) {
    // Dynamic rendering has no implicit external dependency. Preserve the
    // previous scope's stores before loading the same attachments again.
    const VkMemoryBarrier attachment_barrier{
        VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr,
        VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT,
        VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT |
            VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT};
    constexpr VkPipelineStageFlags attachment_stages =
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
        VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deferred_command_buffer_.CmdVkPipelineBarrier(
        attachment_stages, attachment_stages, VK_DEPENDENCY_BY_REGION_BIT,
        1, &attachment_barrier, 0, nullptr, 0, nullptr);
  }

  current_render_pass_ = use_dynamic_rendering ? VK_NULL_HANDLE : render_pass;
  current_framebuffer_ = framebuffer;

  if (use_dynamic_rendering) {
    VkRenderingAttachmentInfo color_attachments[xenos::kMaxColorRenderTargets];
    VkRenderingAttachmentInfo depth_attachment;
    VkRenderingAttachmentInfo stencil_attachment;
    uint32_t color_attachment_count = 0;
    render_target_cache_->GetLastUpdateRenderingAttachments(
        color_attachments, &color_attachment_count, &depth_attachment, &stencil_attachment);
    bool has_depth = depth_attachment.sType == VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    bool has_stencil = stencil_attachment.sType == VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    for (uint32_t i = 0; i < color_attachment_count; ++i) {
      if (color_attachments[i].loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
        frame_work_counters_.color_clears++;
      }
    }
    if (has_depth && depth_attachment.loadOp == VK_ATTACHMENT_LOAD_OP_CLEAR) {
      frame_work_counters_.depth_clears++;
    }
    current_dynamic_rendering_is_transfer_ = false;
    current_dynamic_rendering_has_depth_ = has_depth;
    current_dynamic_rendering_has_stencil_ = has_stencil;
    current_dynamic_rendering_color_count_ = color_attachment_count;
    current_dynamic_rendering_transfer_dest_view_ = VK_NULL_HANDLE;
    current_dynamic_rendering_transfer_depth_view_ = VK_NULL_HANDLE;
    current_dynamic_rendering_transfer_dest_is_depth_ = false;
    current_dynamic_rendering_render_pass_key_ = rpk;
    VkRect2D pass_render_area = render_area;
    if (REXCVAR_GET(vulkan_merge_matching_attachment_passes)) {
      pass_render_area.offset = {0, 0};
      pass_render_area.extent = framebuffer->host_extent;
    }
    current_dynamic_rendering_area_ = pass_render_area;

    VkRenderingInfo rendering_info = {};
    rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering_info.pNext = nullptr;
    rendering_info.flags = 0;
    rendering_info.renderArea = pass_render_area;
    rendering_info.layerCount = 1;
    rendering_info.viewMask = 0;
    rendering_info.colorAttachmentCount = color_attachment_count;
    rendering_info.pColorAttachments = color_attachment_count ? color_attachments : nullptr;
    rendering_info.pDepthAttachment = has_depth ? &depth_attachment : nullptr;
    rendering_info.pStencilAttachment = has_stencil ? &stencil_attachment : nullptr;

    std::string format_str;
    const xenos::ColorRenderTargetFormat color_formats[] = {
        rpk.color_0_view_format, rpk.color_1_view_format,
        rpk.color_2_view_format, rpk.color_3_view_format};
    for (uint32_t i = 0; i < color_attachment_count; ++i) {
      if (!(rpk.depth_and_color_used & (2u << i))) continue;
      if (!format_str.empty()) format_str += "+";
      format_str += fmt::format("C{}:{}[load={},store={}]", i,
          xenos::GetColorRenderTargetFormatName(color_formats[i]),
          int(color_attachments[i].loadOp), int(color_attachments[i].storeOp));
    }
    if (rpk.depth_and_color_used & 1) {
      if (!format_str.empty()) format_str += "+";
      format_str += xenos::GetDepthRenderTargetFormatName(rpk.depth_format);
    }
    if (format_str.empty()) format_str = "NONE";

    VkAttachmentLoadOp c_load = color_attachment_count ? color_attachments[0].loadOp : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    VkAttachmentStoreOp c_store = color_attachment_count ? color_attachments[0].storeOp : VK_ATTACHMENT_STORE_OP_DONT_CARE;
    VkAttachmentLoadOp d_load = has_depth ? depth_attachment.loadOp : VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    VkAttachmentStoreOp d_store = has_depth ? depth_attachment.storeOp : VK_ATTACHMENT_STORE_OP_DONT_CARE;

    BeginScopeProfile(ScopeCategory::kMainPass, pass_render_area.extent.width, pass_render_area.extent.height,
                      format_str, c_load, c_store, d_load, d_store, "main_pass");
    deferred_command_buffer_.CmdVkBeginRendering(&rendering_info);
  } else {
    VkRenderPassBeginInfo render_pass_begin_info;
    render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_begin_info.pNext = nullptr;
    render_pass_begin_info.renderPass = render_pass;
    render_pass_begin_info.framebuffer = framebuffer->framebuffer;
    render_pass_begin_info.renderArea.offset.x = 0;
    render_pass_begin_info.renderArea.offset.y = 0;
    // TODO(Triang3l): Actual dirty width / height in the deferred command
    // buffer.
    render_pass_begin_info.renderArea.extent = framebuffer->host_extent;
    render_pass_begin_info.clearValueCount = 0;
    render_pass_begin_info.pClearValues = nullptr;
    BeginScopeProfile(ScopeCategory::kMainPass, framebuffer->host_extent.width, framebuffer->host_extent.height,
                      "CLASSIC_RP", VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
                      VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, "main_pass_classic");
    deferred_command_buffer_.CmdVkBeginRenderPass(&render_pass_begin_info,
                                                  VK_SUBPASS_CONTENTS_INLINE);
  }
  in_render_pass_ = true;
  if (REXCVAR_GET(vulkan_coalesce_adjacent_clears) && render_target_cache_->HasPendingDeferredClear()) {
    render_target_cache_->TryEmitDeferredClearInDrawPass(framebuffer);
  }
  frame_pass_stats_.pass_count++;
  frame_pass_stats_.main_pass_count++;
  frame_pass_stats_.main_render_area_pixels +=
      uint64_t(render_area.extent.width) * render_area.extent.height;
  frame_pass_stats_.main_full_area_pixels +=
      uint64_t(framebuffer->host_extent.width) * framebuffer->host_extent.height;
  if (render_area.offset.x || render_area.offset.y ||
      render_area.extent.width != framebuffer->host_extent.width ||
      render_area.extent.height != framebuffer->host_extent.height) {
    frame_pass_stats_.bounded_main_pass_count++;
  }
}

void VulkanCommandProcessor::SubmitBarriersAndEnterRenderTargetCacheRenderPass(
    VkRenderPass render_pass, const VulkanRenderTargetCache::Framebuffer* framebuffer,
    VkImageView transfer_dest_view, bool transfer_dest_is_depth,
    VkImageView transfer_depth_view) {
  if (render_target_cache_->HasPendingDeferredClear()) {
    render_target_cache_->FlushPendingDeferredClear();
  }
  SubmitBarriers(false);
  const ui::vulkan::VulkanDevice* vulkan_device = GetVulkanDevice();
  bool use_dynamic_rendering =
      REXCVAR_GET(vulkan_dynamic_rendering) && vulkan_device->properties().dynamicRendering;

  if (use_dynamic_rendering) {
    if (in_render_pass_ && current_dynamic_rendering_is_transfer_ &&
        current_framebuffer_ == framebuffer &&
        current_render_pass_ == VK_NULL_HANDLE &&
        current_dynamic_rendering_transfer_dest_view_ == transfer_dest_view &&
        current_dynamic_rendering_transfer_depth_view_ == transfer_depth_view &&
        current_dynamic_rendering_transfer_dest_is_depth_ == transfer_dest_is_depth) {
      return;
    }
  } else {
    if (current_render_pass_ == render_pass && current_framebuffer_ == framebuffer) {
      return;
    }
  }

  if (in_render_pass_) {
    if (use_dynamic_rendering) {
      deferred_command_buffer_.CmdVkEndRendering();
    } else {
      deferred_command_buffer_.CmdVkEndRenderPass();
    }
    EndScopeProfile(active_scope_index_);
    in_render_pass_ = false;
  }

  current_render_pass_ = use_dynamic_rendering ? VK_NULL_HANDLE : render_pass;
  current_framebuffer_ = framebuffer;

  if (use_dynamic_rendering) {
    VkRenderingAttachmentInfo color_attachment = {};
    VkRenderingAttachmentInfo depth_attachment = {};
    VkRenderingAttachmentInfo stencil_attachment = {};

    if (transfer_dest_is_depth) {
      depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
      depth_attachment.pNext = nullptr;
      depth_attachment.imageView = transfer_dest_view;
      depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
      depth_attachment.resolveMode = VK_RESOLVE_MODE_NONE;
      depth_attachment.resolveImageView = VK_NULL_HANDLE;
      depth_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      depth_attachment.clearValue = {};
      stencil_attachment = depth_attachment;
      if (REXCVAR_GET(vulkan_prune_transfer_stencil)) {
        stencil_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        stencil_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
      }
    } else {
      color_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
      color_attachment.pNext = nullptr;
      color_attachment.imageView = transfer_dest_view;
      color_attachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
      color_attachment.resolveMode = VK_RESOLVE_MODE_NONE;
      color_attachment.resolveImageView = VK_NULL_HANDLE;
      color_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
      color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
      color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
      color_attachment.clearValue = {};

      if (transfer_depth_view != VK_NULL_HANDLE) {
        depth_attachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
        depth_attachment.pNext = nullptr;
        depth_attachment.imageView = transfer_depth_view;
        depth_attachment.imageLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
        depth_attachment.resolveMode = VK_RESOLVE_MODE_NONE;
        depth_attachment.resolveImageView = VK_NULL_HANDLE;
        depth_attachment.resolveImageLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        depth_attachment.clearValue = {};
        stencil_attachment = depth_attachment;
        if (REXCVAR_GET(vulkan_prune_transfer_stencil)) {
          stencil_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
          stencil_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        }
      }
    }

    VkRenderingInfo rendering_info = {};
    rendering_info.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    rendering_info.pNext = nullptr;
    rendering_info.flags = 0;
    rendering_info.renderArea.offset.x = 0;
    rendering_info.renderArea.offset.y = 0;
    rendering_info.renderArea.extent = framebuffer->host_extent;
    current_dynamic_rendering_area_ = rendering_info.renderArea;
    rendering_info.layerCount = 1;
    rendering_info.viewMask = 0;
    current_dynamic_rendering_is_transfer_ = true;
    current_dynamic_rendering_transfer_dest_is_depth_ = transfer_dest_is_depth;
    const bool has_depth = transfer_dest_is_depth || (transfer_depth_view != VK_NULL_HANDLE);
    current_dynamic_rendering_has_depth_ = has_depth;
    current_dynamic_rendering_has_stencil_ = has_depth;
    current_dynamic_rendering_color_count_ = transfer_dest_is_depth ? 0 : 1;
    current_dynamic_rendering_transfer_dest_view_ = transfer_dest_view;
    current_dynamic_rendering_transfer_depth_view_ = transfer_depth_view;
    current_dynamic_rendering_render_pass_key_ = {};
    rendering_info.colorAttachmentCount = transfer_dest_is_depth ? 0 : 1;
    rendering_info.pColorAttachments = transfer_dest_is_depth ? nullptr : &color_attachment;
    rendering_info.pDepthAttachment = has_depth ? &depth_attachment : nullptr;
    rendering_info.pStencilAttachment = has_depth ? &stencil_attachment : nullptr;
    BeginScopeProfile(ScopeCategory::kTransferPass, framebuffer->host_extent.width, framebuffer->host_extent.height,
                      transfer_dest_is_depth ? "DEPTH_TRANSFER" : "COLOR_TRANSFER",
                      transfer_dest_is_depth ? VK_ATTACHMENT_LOAD_OP_DONT_CARE : VK_ATTACHMENT_LOAD_OP_LOAD,
                      transfer_dest_is_depth ? VK_ATTACHMENT_STORE_OP_DONT_CARE : VK_ATTACHMENT_STORE_OP_STORE,
                      transfer_dest_is_depth ? VK_ATTACHMENT_LOAD_OP_LOAD : VK_ATTACHMENT_LOAD_OP_DONT_CARE,
                      transfer_dest_is_depth ? VK_ATTACHMENT_STORE_OP_STORE : VK_ATTACHMENT_STORE_OP_DONT_CARE,
                      REXCVAR_GET(vulkan_prune_transfer_stencil) ? "transfer_pass_stencil_pruned" : "transfer_pass");
    deferred_command_buffer_.CmdVkBeginRendering(&rendering_info);
  } else {
    VkRenderPassBeginInfo render_pass_begin_info;
    render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render_pass_begin_info.pNext = nullptr;
    render_pass_begin_info.renderPass = render_pass;
    render_pass_begin_info.framebuffer = framebuffer->framebuffer;
    render_pass_begin_info.renderArea.offset.x = 0;
    render_pass_begin_info.renderArea.offset.y = 0;
    render_pass_begin_info.renderArea.extent = framebuffer->host_extent;
    render_pass_begin_info.clearValueCount = 0;
    render_pass_begin_info.pClearValues = nullptr;
    BeginScopeProfile(ScopeCategory::kTransferPass, framebuffer->host_extent.width, framebuffer->host_extent.height,
                      "CLASSIC_TRANSFER", VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE,
                      VK_ATTACHMENT_LOAD_OP_LOAD, VK_ATTACHMENT_STORE_OP_STORE, "transfer_pass_classic");
    deferred_command_buffer_.CmdVkBeginRenderPass(&render_pass_begin_info,
                                                  VK_SUBPASS_CONTENTS_INLINE);
  }
  in_render_pass_ = true;
  if (REXCVAR_GET(vulkan_coalesce_adjacent_clears) && render_target_cache_->HasPendingDeferredClear()) {
    render_target_cache_->TryEmitDeferredClearInDrawPass(framebuffer);
  }
  frame_pass_stats_.pass_count++;
  frame_pass_stats_.transfer_pass_count++;
}

void VulkanCommandProcessor::EndRenderPass() {
  assert_true(submission_open_);
  if (!in_render_pass_) {
    return;
  }
  if (current_render_pass_ == VK_NULL_HANDLE) {
    deferred_command_buffer_.CmdVkEndRendering();
  } else {
    deferred_command_buffer_.CmdVkEndRenderPass();
  }
  EndScopeProfile(active_scope_index_);
  current_render_pass_ = VK_NULL_HANDLE;
  current_framebuffer_ = nullptr;
  in_render_pass_ = false;
  current_dynamic_rendering_is_transfer_ = false;
  current_dynamic_rendering_has_depth_ = false;
  current_dynamic_rendering_has_stencil_ = false;
  current_dynamic_rendering_color_count_ = 0;
  current_dynamic_rendering_transfer_dest_view_ = VK_NULL_HANDLE;
  current_dynamic_rendering_render_pass_key_ = {};
}

VkDescriptorSet VulkanCommandProcessor::AllocateSingleTransientDescriptor(
    SingleTransientDescriptorLayout transient_descriptor_layout) {
  assert_true(frame_open_);
  VkDescriptorSet descriptor_set;
  std::vector<VkDescriptorSet>& transient_descriptors_free =
      single_transient_descriptors_free_[size_t(transient_descriptor_layout)];
  if (!transient_descriptors_free.empty()) {
    descriptor_set = transient_descriptors_free.back();
    transient_descriptors_free.pop_back();
  } else {
    const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
    [[maybe_unused]] const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    [[maybe_unused]] const VkDevice device = vulkan_device->device();
    bool is_storage_buffer =
        transient_descriptor_layout == SingleTransientDescriptorLayout::kStorageBufferCompute ||
        transient_descriptor_layout == SingleTransientDescriptorLayout::kStorageBufferPairCompute;
    ui::vulkan::LinkedTypeDescriptorSetAllocator& transient_descriptor_allocator =
        is_storage_buffer ? transient_descriptor_allocator_storage_buffer_
                          : transient_descriptor_allocator_uniform_buffer_;
    VkDescriptorPoolSize descriptor_count;
    descriptor_count.type =
        is_storage_buffer ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER : VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    descriptor_count.descriptorCount =
        transient_descriptor_layout == SingleTransientDescriptorLayout::kStorageBufferPairCompute
            ? 2
            : 1;
    descriptor_set = transient_descriptor_allocator.Allocate(
        GetSingleTransientDescriptorLayout(transient_descriptor_layout), &descriptor_count, 1);
    if (descriptor_set == VK_NULL_HANDLE) {
      return VK_NULL_HANDLE;
    }
  }
  UsedSingleTransientDescriptor used_descriptor;
  used_descriptor.frame = frame_current_;
  used_descriptor.layout = transient_descriptor_layout;
  used_descriptor.set = descriptor_set;
  single_transient_descriptors_used_.emplace_back(used_descriptor);
  return descriptor_set;
}

VkDescriptorSetLayout VulkanCommandProcessor::GetTextureDescriptorSetLayout(bool is_vertex,
                                                                            size_t texture_count,
                                                                            size_t sampler_count) {
  size_t binding_count = texture_count + sampler_count;
  if (!binding_count) {
    return descriptor_set_layout_empty_;
  }

  TextureDescriptorSetLayoutKey texture_descriptor_set_layout_key;
  texture_descriptor_set_layout_key.texture_count = uint32_t(texture_count);
  texture_descriptor_set_layout_key.sampler_count = uint32_t(sampler_count);
  texture_descriptor_set_layout_key.is_vertex = uint32_t(is_vertex);
  auto it_existing = descriptor_set_layouts_textures_.find(texture_descriptor_set_layout_key);
  if (it_existing != descriptor_set_layouts_textures_.end()) {
    return it_existing->second;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  descriptor_set_layout_bindings_.clear();
  descriptor_set_layout_bindings_.reserve(binding_count);
  VkShaderStageFlags stage_flags =
      is_vertex ? guest_shader_vertex_stages_ : VK_SHADER_STAGE_FRAGMENT_BIT;
  for (size_t i = 0; i < texture_count; ++i) {
    VkDescriptorSetLayoutBinding& descriptor_set_layout_binding =
        descriptor_set_layout_bindings_.emplace_back();
    descriptor_set_layout_binding.binding = uint32_t(i);
    descriptor_set_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_set_layout_binding.descriptorCount = 1;
    descriptor_set_layout_binding.stageFlags = stage_flags;
  }
  for (size_t i = 0; i < sampler_count; ++i) {
    VkDescriptorSetLayoutBinding& descriptor_set_layout_binding =
        descriptor_set_layout_bindings_.emplace_back();
    descriptor_set_layout_binding.binding = uint32_t(texture_count + i);
    descriptor_set_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    descriptor_set_layout_binding.descriptorCount = 1;
    descriptor_set_layout_binding.stageFlags = stage_flags;
  }
  VkDescriptorSetLayoutCreateInfo descriptor_set_layout_create_info;
  descriptor_set_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptor_set_layout_create_info.pNext = nullptr;
  descriptor_set_layout_create_info.flags = 0;
  descriptor_set_layout_create_info.bindingCount = uint32_t(binding_count);
  descriptor_set_layout_create_info.pBindings = descriptor_set_layout_bindings_.data();
  VkDescriptorSetLayout texture_descriptor_set_layout;
  if (dfn.vkCreateDescriptorSetLayout(device, &descriptor_set_layout_create_info, nullptr,
                                      &texture_descriptor_set_layout) != VK_SUCCESS) {
    return VK_NULL_HANDLE;
  }
  descriptor_set_layouts_textures_.emplace(texture_descriptor_set_layout_key,
                                           texture_descriptor_set_layout);
  return texture_descriptor_set_layout;
}

const VulkanPipelineCache::PipelineLayoutProvider* VulkanCommandProcessor::GetPipelineLayout(
    size_t texture_count_pixel, size_t sampler_count_pixel, size_t texture_count_vertex,
    size_t sampler_count_vertex) {
  PipelineLayoutKey pipeline_layout_key;
  pipeline_layout_key.texture_count_pixel = uint16_t(texture_count_pixel);
  pipeline_layout_key.sampler_count_pixel = uint16_t(sampler_count_pixel);
  pipeline_layout_key.texture_count_vertex = uint16_t(texture_count_vertex);
  pipeline_layout_key.sampler_count_vertex = uint16_t(sampler_count_vertex);
  {
    auto it = pipeline_layouts_.find(pipeline_layout_key);
    if (it != pipeline_layouts_.end()) {
      return &it->second;
    }
  }

  VkDescriptorSetLayout descriptor_set_layout_textures_vertex =
      GetTextureDescriptorSetLayout(true, texture_count_vertex, sampler_count_vertex);
  if (descriptor_set_layout_textures_vertex == VK_NULL_HANDLE) {
    REXGPU_ERROR(
        "Failed to obtain a Vulkan descriptor set layout for {} sampled images "
        "and {} samplers for guest vertex shaders",
        texture_count_vertex, sampler_count_vertex);
    return nullptr;
  }
  VkDescriptorSetLayout descriptor_set_layout_textures_pixel =
      GetTextureDescriptorSetLayout(false, texture_count_pixel, sampler_count_pixel);
  if (descriptor_set_layout_textures_pixel == VK_NULL_HANDLE) {
    REXGPU_ERROR(
        "Failed to obtain a Vulkan descriptor set layout for {} sampled images "
        "and {} samplers for guest pixel shaders",
        texture_count_pixel, sampler_count_pixel);
    return nullptr;
  }

  VkDescriptorSetLayout descriptor_set_layouts[SpirvShaderTranslator::kDescriptorSetCount];
  // Immutable layouts.
  descriptor_set_layouts[SpirvShaderTranslator::kDescriptorSetSharedMemoryAndEdram] =
      descriptor_set_layout_shared_memory_and_edram_;
  descriptor_set_layouts[SpirvShaderTranslator::kDescriptorSetConstants] =
      descriptor_set_layout_constants_;
  // Mutable layouts.
  descriptor_set_layouts[SpirvShaderTranslator::kDescriptorSetTexturesVertex] =
      descriptor_set_layout_textures_vertex;
  descriptor_set_layouts[SpirvShaderTranslator::kDescriptorSetTexturesPixel] =
      descriptor_set_layout_textures_pixel;

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  VkPipelineLayoutCreateInfo pipeline_layout_create_info;
  pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  pipeline_layout_create_info.pNext = nullptr;
  pipeline_layout_create_info.flags = 0;
  pipeline_layout_create_info.setLayoutCount = uint32_t(rex::countof(descriptor_set_layouts));
  pipeline_layout_create_info.pSetLayouts = descriptor_set_layouts;
  pipeline_layout_create_info.pushConstantRangeCount = 0;
  pipeline_layout_create_info.pPushConstantRanges = nullptr;
  VkPipelineLayout pipeline_layout;
  if (dfn.vkCreatePipelineLayout(device, &pipeline_layout_create_info, nullptr, &pipeline_layout) !=
      VK_SUCCESS) {
    REXGPU_ERROR(
        "Failed to create a Vulkan pipeline layout for guest drawing with {} "
        "pixel shader and {} vertex shader textures",
        texture_count_pixel, texture_count_vertex);
    return nullptr;
  }
  auto emplaced_pair = pipeline_layouts_.emplace(
      std::piecewise_construct, std::forward_as_tuple(pipeline_layout_key),
      std::forward_as_tuple(pipeline_layout, descriptor_set_layout_textures_vertex,
                            descriptor_set_layout_textures_pixel));
  // unordered_map insertion doesn't invalidate element references.
  return &emplaced_pair.first->second;
}

VulkanCommandProcessor::ScratchBufferAcquisition VulkanCommandProcessor::AcquireScratchGpuBuffer(
    VkDeviceSize size, VkPipelineStageFlags initial_stage_mask, VkAccessFlags initial_access_mask) {
  assert_true(submission_open_);
  assert_false(scratch_buffer_used_);
  if (!submission_open_ || scratch_buffer_used_ || !size) {
    return ScratchBufferAcquisition();
  }

  uint64_t submission_current = GetCurrentSubmission();

  if (scratch_buffer_ != VK_NULL_HANDLE && size <= scratch_buffer_size_) {
    // Already used previously - transition.
    PushBufferMemoryBarrier(scratch_buffer_, 0, VK_WHOLE_SIZE, scratch_buffer_last_stage_mask_,
                            initial_stage_mask, scratch_buffer_last_access_mask_,
                            initial_access_mask);
    scratch_buffer_last_stage_mask_ = initial_stage_mask;
    scratch_buffer_last_access_mask_ = initial_access_mask;
    scratch_buffer_last_usage_submission_ = submission_current;
    scratch_buffer_used_ = true;
    return ScratchBufferAcquisition(*this, scratch_buffer_, initial_stage_mask,
                                    initial_access_mask);
  }

  size = rex::align(size, kScratchBufferSizeIncrement);

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();

  VkDeviceMemory new_scratch_buffer_memory;
  VkBuffer new_scratch_buffer;
  // VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT for
  // texture loading.
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
          vulkan_device, size,
          VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
          ui::vulkan::util::MemoryPurpose::kDeviceLocal, new_scratch_buffer,
          new_scratch_buffer_memory)) {
    REXGPU_ERROR("VulkanCommandProcessor: Failed to create a {} MB scratch GPU buffer", size >> 20);
    return ScratchBufferAcquisition();
  }

  if (submission_completed_ >= scratch_buffer_last_usage_submission_) {
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    if (scratch_buffer_ != VK_NULL_HANDLE) {
      dfn.vkDestroyBuffer(device, scratch_buffer_, nullptr);
    }
    if (scratch_buffer_memory_ != VK_NULL_HANDLE) {
      dfn.vkFreeMemory(device, scratch_buffer_memory_, nullptr);
    }
  } else {
    if (scratch_buffer_ != VK_NULL_HANDLE) {
      destroy_buffers_.emplace_back(scratch_buffer_last_usage_submission_, scratch_buffer_);
    }
    if (scratch_buffer_memory_ != VK_NULL_HANDLE) {
      destroy_memory_.emplace_back(scratch_buffer_last_usage_submission_, scratch_buffer_memory_);
    }
  }

  scratch_buffer_memory_ = new_scratch_buffer_memory;
  scratch_buffer_ = new_scratch_buffer;
  scratch_buffer_size_ = size;
  // Not used yet, no need for a barrier.
  scratch_buffer_last_stage_mask_ = initial_access_mask;
  scratch_buffer_last_access_mask_ = initial_stage_mask;
  scratch_buffer_last_usage_submission_ = submission_current;
  scratch_buffer_used_ = true;
  return ScratchBufferAcquisition(*this, new_scratch_buffer, initial_stage_mask,
                                  initial_access_mask);
}

void VulkanCommandProcessor::BindExternalGraphicsPipeline(VkPipeline pipeline,
                                                          bool keep_dynamic_depth_bias,
                                                          bool keep_dynamic_blend_constants,
                                                          bool keep_dynamic_stencil_mask_ref) {
  if (!keep_dynamic_depth_bias) {
    dynamic_depth_bias_update_needed_ = true;
  }
  if (!keep_dynamic_blend_constants) {
    dynamic_blend_constants_update_needed_ = true;
  }
  if (!keep_dynamic_stencil_mask_ref) {
    dynamic_stencil_compare_mask_front_update_needed_ = true;
    dynamic_stencil_compare_mask_back_update_needed_ = true;
    dynamic_stencil_write_mask_front_update_needed_ = true;
    dynamic_stencil_write_mask_back_update_needed_ = true;
    dynamic_stencil_reference_front_update_needed_ = true;
    dynamic_stencil_reference_back_update_needed_ = true;
  }
  if (current_external_graphics_pipeline_ == pipeline) {
    return;
  }
  deferred_command_buffer_.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
  current_external_graphics_pipeline_ = pipeline;
  current_guest_graphics_pipeline_ = VK_NULL_HANDLE;
  current_guest_graphics_pipeline_layout_ = VK_NULL_HANDLE;
}

void VulkanCommandProcessor::BindExternalComputePipeline(VkPipeline pipeline) {
  if (current_external_compute_pipeline_ == pipeline) {
    return;
  }
  deferred_command_buffer_.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
  current_external_compute_pipeline_ = pipeline;
}

void VulkanCommandProcessor::SetViewport(const VkViewport& viewport) {
  VkViewport effective_viewport = viewport;
  double scale = REXCVAR_GET(vulkan_ablate_viewport_scale);
  if (scale > 0.0 && scale < 1.0 && frame_current_ >= 3500) {
    effective_viewport.width *= float(scale);
    effective_viewport.height *= float(scale);
  }
  if (!dynamic_viewport_update_needed_) {
    dynamic_viewport_update_needed_ |= dynamic_viewport_.x != effective_viewport.x;
    dynamic_viewport_update_needed_ |= dynamic_viewport_.y != effective_viewport.y;
    dynamic_viewport_update_needed_ |= dynamic_viewport_.width != effective_viewport.width;
    dynamic_viewport_update_needed_ |= dynamic_viewport_.height != effective_viewport.height;
    dynamic_viewport_update_needed_ |= dynamic_viewport_.minDepth != effective_viewport.minDepth;
    dynamic_viewport_update_needed_ |= dynamic_viewport_.maxDepth != effective_viewport.maxDepth;
  }
  if (dynamic_viewport_update_needed_) {
    dynamic_viewport_ = effective_viewport;
    deferred_command_buffer_.CmdVkSetViewport(0, 1, &dynamic_viewport_);
    dynamic_viewport_update_needed_ = false;
  }
}

void VulkanCommandProcessor::SetScissor(const VkRect2D& scissor) {
  VkRect2D effective_scissor = scissor;
  double scale = REXCVAR_GET(vulkan_ablate_viewport_scale);
  if (scale > 0.0 && scale < 1.0 && frame_current_ >= 3500) {
    effective_scissor.extent.width = std::max(1u, uint32_t(float(effective_scissor.extent.width) * float(scale)));
    effective_scissor.extent.height = std::max(1u, uint32_t(float(effective_scissor.extent.height) * float(scale)));
  }
  if (!dynamic_scissor_update_needed_) {
    dynamic_scissor_update_needed_ |= dynamic_scissor_.offset.x != effective_scissor.offset.x;
    dynamic_scissor_update_needed_ |= dynamic_scissor_.offset.y != effective_scissor.offset.y;
    dynamic_scissor_update_needed_ |= dynamic_scissor_.extent.width != effective_scissor.extent.width;
    dynamic_scissor_update_needed_ |= dynamic_scissor_.extent.height != effective_scissor.extent.height;
  }
  if (dynamic_scissor_update_needed_) {
    dynamic_scissor_ = effective_scissor;
    deferred_command_buffer_.CmdVkSetScissor(0, 1, &dynamic_scissor_);
    dynamic_scissor_update_needed_ = false;
  }
}

void VulkanCommandProcessor::OnPrimaryBufferEnd() {
  if (REXCVAR_GET(vulkan_submit_on_primary_buffer_end) && submission_open_ &&
      CanEndSubmissionImmediately()) {
    EndSubmission(false);
  }
}

Shader* VulkanCommandProcessor::LoadShader(xenos::ShaderType shader_type, uint32_t guest_address,
                                           const uint32_t* host_address, uint32_t dword_count) {
  return pipeline_cache_->LoadShader(shader_type, host_address, dword_count);
}

struct BenDeltaStats {
  uint32_t frame = 0;
  uint32_t char_draw_count = 0;
  uint64_t texture_hash = 0;
  uint64_t blend_hash = 0;
  uint64_t constant_hash = 0;
  uint32_t submesh_counts[8] = {0};
};
static BenDeltaStats s_current_ben_delta{};
static BenDeltaStats s_prev_ben_delta{};

bool VulkanCommandProcessor::ShouldProfileFrameDraws(uint64_t frame) const {
  if (IsGpuProvenanceFrame(frame)) {
    return true;
  }
  if (REXCVAR_GET(gpu_frame_audit_enable) || REXCVAR_GET(gpu_profile_scope_timestamps)) {
    uint64_t start = REXCVAR_GET(gpu_frame_audit_window_start);
    uint64_t end = REXCVAR_GET(gpu_frame_audit_window_end);
    if ((start == 0 || frame >= start) && (end == 0 || frame <= end)) {
      return true;
    }
  }
  if (REXCVAR_GET(gpu_profile_cpu)) {
    return (frame % 120 == 0);
  }
  return false;
}

void VulkanCommandProcessor::GpuCompletionTracker::Start(
    VkDevice device, const ui::vulkan::VulkanDevice::Functions* dfn) {
  device_ = device;
  dfn_ = dfn;
  running_ = true;
  thread_ = std::thread([this]() { Run(); });
}

void VulkanCommandProcessor::GpuCompletionTracker::Stop() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
    cv_.notify_all();
  }
  if (thread_.joinable()) {
    thread_.join();
  }
}

void VulkanCommandProcessor::GpuCompletionTracker::TrackFrameClosingFence(uint64_t frame,
                                                                         VkFence fence) {
  std::lock_guard<std::mutex> lock(mutex_);
  pending_fences_.push_back({frame, fence});
  cv_.notify_one();
}

bool VulkanCommandProcessor::GpuCompletionTracker::GetCompletionInterval(uint64_t frame,
                                                                        double* interval_ms) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = completed_intervals_.find(frame);
  if (it != completed_intervals_.end()) {
    *interval_ms = it->second;
    completed_intervals_.erase(it);
    return true;
  }
  return false;
}

void VulkanCommandProcessor::GpuCompletionTracker::Run() {
  while (true) {
    PendingFrame pending;
    {
      std::unique_lock<std::mutex> lock(mutex_);
      cv_.wait(lock, [this]() { return !running_ || !pending_fences_.empty(); });
      if (!running_ && pending_fences_.empty()) {
        break;
      }
      pending = pending_fences_.front();
    }

    while (running_) {
      VkResult res =
          dfn_->vkWaitForFences(device_, 1, &pending.fence, VK_TRUE, 500'000 /* 500 us */);
      if (res == VK_SUCCESS) {
        auto now = std::chrono::steady_clock::now();
        double interval = 0.0;
        if (has_prev_) {
          interval = std::chrono::duration<double, std::milli>(now - prev_completion_).count();
        }
        prev_completion_ = now;
        has_prev_ = true;

        std::lock_guard<std::mutex> lock(mutex_);
        if (!pending_fences_.empty() && pending_fences_.front().frame == pending.frame) {
          pending_fences_.pop_front();
        }
        completed_intervals_[pending.frame] = interval;
        break;
      } else if (res == VK_TIMEOUT) {
        continue;
      } else {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!pending_fences_.empty() && pending_fences_.front().frame == pending.frame) {
          pending_fences_.pop_front();
        }
        break;
      }
    }
  }
}

bool VulkanCommandProcessor::InitializeGpuTimingResources() {
  const ui::vulkan::VulkanDevice* vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  VkDevice device = vulkan_device->device();
  VkPhysicalDevice physical_device = vulkan_device->physical_device();
  const auto& ifn = vulkan_device->vulkan_instance()->functions();

  VkPhysicalDeviceProperties props{};
  ifn.vkGetPhysicalDeviceProperties(physical_device, &props);
  timestamp_period_ns_ = props.limits.timestampPeriod;

  uint32_t qf_count = 0;
  ifn.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, nullptr);
  std::vector<VkQueueFamilyProperties> qf_props(qf_count);
  ifn.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, qf_props.data());
  uint32_t qf_index = vulkan_device->queue_family_graphics_compute();
  if (qf_index < qf_props.size()) {
    timestamp_valid_bits_ = qf_props[qf_index].timestampValidBits;
  } else {
    timestamp_valid_bits_ = 0;
  }

  REXGPU_INFO("Vulkan timestamps: valid_bits={}, period_ns={}", timestamp_valid_bits_,
              timestamp_period_ns_);

  VkQueryPoolCreateInfo pool_info{};
  pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  pool_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
  pool_info.queryCount = kTotalTimestampQueries;
  if (dfn.vkCreateQueryPool(device, &pool_info, nullptr, &timestamp_query_pool_) != VK_SUCCESS) {
    REXGPU_WARN("Failed to create Vulkan timestamp query pool");
    timestamp_query_pool_ = VK_NULL_HANDLE;
  }

  if (REXCVAR_GET(gpu_profile_transfer_timestamps) &&
      REXCVAR_GET(gpu_frame_audit_enable) && timestamp_valid_bits_) {
    pool_info.queryCount = kMaxTimestampFramesInFlight * kTransferQueriesPerFrame;
    if (dfn.vkCreateQueryPool(device, &pool_info, nullptr, &transfer_profile_pool_) != VK_SUCCESS)
      transfer_profile_pool_ = VK_NULL_HANDLE;
  }

  if (REXCVAR_GET(gpu_profile_scope_timestamps) && timestamp_valid_bits_) {
    pool_info.queryCount = kMaxTimestampFramesInFlight * kScopeQueriesPerFrame;
    if (dfn.vkCreateQueryPool(device, &pool_info, nullptr, &scope_profile_pool_) != VK_SUCCESS) {
      REXGPU_WARN("Failed to create Vulkan scope profile query pool");
      scope_profile_pool_ = VK_NULL_HANDLE;
    }
  }
  gpu_completion_tracker_.Start(device, &dfn);
  return true;
}

uint32_t VulkanCommandProcessor::BeginTransferProfile() {
  if (!transfer_profile_pool_ || !frame_open_ ||
      !ShouldProfileFrameDraws(frame_current_)) return UINT32_MAX;
  uint32_t slot = frame_current_ % kMaxTimestampFramesInFlight;
  uint32_t& count = transfer_profile_counts_[slot];
  if (count + 2 > kTransferQueriesPerFrame) return UINT32_MAX;
  uint32_t query = slot * kTransferQueriesPerFrame + count;
  count += 2;
  deferred_command_buffer_.CmdProfileTimestamp(transfer_profile_pool_, query);
  return query;
}

void VulkanCommandProcessor::EndTransferProfile(uint32_t query) {
  if (query != UINT32_MAX)
    deferred_command_buffer_.CmdProfileTimestamp(transfer_profile_pool_, query + 1);
}

uint32_t VulkanCommandProcessor::BeginScopeProfile(
    ScopeCategory category, uint32_t width, uint32_t height,
    const std::string& format,
    VkAttachmentLoadOp color_load, VkAttachmentStoreOp color_store,
    VkAttachmentLoadOp depth_load, VkAttachmentStoreOp depth_store,
    const std::string& description) {
  if (!REXCVAR_GET(gpu_profile_scope_timestamps) ||
      !REXCVAR_GET(gpu_frame_audit_enable) || !frame_open_ ||
      !ShouldProfileFrameDraws(frame_current_)) {
    return UINT32_MAX;
  }
  uint32_t slot = frame_current_ % kMaxTimestampFramesInFlight;
  auto& scopes = frame_scopes_[slot];
  if (scopes.size() >= kMaxScopesPerFrame) return UINT32_MAX;

  uint32_t& query_count = scope_profile_counts_[slot];
  uint32_t query_start = UINT32_MAX;
  if (scope_profile_pool_ && query_count + 2 <= kScopeQueriesPerFrame) {
    query_start = slot * kScopeQueriesPerFrame + query_count;
    query_count += 2;
    deferred_command_buffer_.CmdProfileTimestamp(scope_profile_pool_, query_start);
  }

  ScopeProfileRecord record{};
  record.category = category;
  record.width = width;
  record.height = height;
  record.format = format;
  record.draw_count = 0;
  record.color_load_op = color_load;
  record.color_store_op = color_store;
  record.depth_load_op = depth_load;
  record.depth_store_op = depth_store;
  record.query_start = query_start;
  record.query_end = (query_start != UINT32_MAX) ? query_start + 1 : UINT32_MAX;
  record.description = description;

  uint32_t scope_idx = uint32_t(scopes.size());
  scopes.push_back(std::move(record));
  // Resolve envelopes and compute spans must not replace the live render scope.
  if (category == ScopeCategory::kMainPass || category == ScopeCategory::kTransferPass)
    active_scope_index_ = scope_idx;
  return scope_idx;
}

void VulkanCommandProcessor::EndScopeProfile(uint32_t scope_index) {
  if (scope_index == UINT32_MAX) return;
  uint32_t slot = frame_current_ % kMaxTimestampFramesInFlight;
  auto& scopes = frame_scopes_[slot];
  if (scope_index < scopes.size() && !scopes[scope_index].ended) {
    scopes[scope_index].ended = true;
    if (scope_profile_pool_ && scopes[scope_index].query_end != UINT32_MAX) {
      deferred_command_buffer_.CmdProfileTimestamp(scope_profile_pool_, scopes[scope_index].query_end);
    }
  }
  if (active_scope_index_ == scope_index) {
    active_scope_index_ = UINT32_MAX;
  }
}

void VulkanCommandProcessor::NoteScopeDraw(const VulkanShader* pixel_shader) {
  if (active_scope_index_ != UINT32_MAX) {
    uint32_t slot = frame_current_ % kMaxTimestampFramesInFlight;
    auto& scopes = frame_scopes_[slot];
    if (active_scope_index_ < scopes.size()) {
      auto& scope = scopes[active_scope_index_];
      scope.draw_count++;
      if (!pixel_shader) {
        ++scope.pixel_shader_overflow_draw_count;
        return;
      }
      uint64_t hash = pixel_shader->ucode_data_hash();
      auto group = scope.pixel_shaders.find(hash);
      if (group == scope.pixel_shaders.end()) {
        if (scope.pixel_shaders.size() >= 64) {
          ++scope.pixel_shader_overflow_draw_count;
          return;
        }
        group = scope.pixel_shaders.emplace(hash, ScopeProfileRecord::PixelShaderGroup{
            0, pixel_shader && pixel_shader->writes_depth(),
            pixel_shader && pixel_shader->kills_pixels()}).first;
      }
      ++group->second.draw_count;
    }
  }
}

void VulkanCommandProcessor::RecordTransferElision() {
  frame_pass_stats_.elided_dead_transfers++;
}

void VulkanCommandProcessor::ShutdownGpuTimingResources() {
  gpu_completion_tracker_.Stop();

  const ui::vulkan::VulkanDevice* vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  VkDevice device = vulkan_device->device();

  if (scope_profile_pool_ != VK_NULL_HANDLE) {
    dfn.vkDestroyQueryPool(device, scope_profile_pool_, nullptr);
    scope_profile_pool_ = VK_NULL_HANDLE;
  }
  if (transfer_profile_pool_) {
    dfn.vkDestroyQueryPool(device, transfer_profile_pool_, nullptr);
    transfer_profile_pool_ = VK_NULL_HANDLE;
  }
  if (timestamp_query_pool_ != VK_NULL_HANDLE) {
    dfn.vkDestroyQueryPool(device, timestamp_query_pool_, nullptr);
    timestamp_query_pool_ = VK_NULL_HANDLE;
  }
}

void VulkanCommandProcessor::RecordBarrierTuple(
    BarrierRequester requester, bool is_image, VkAccessFlags src_access,
    VkAccessFlags dst_access, VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage,
    VkImageLayout old_layout, VkImageLayout new_layout) {
  BarrierTupleKey key{};
  key.requester = requester;
  key.is_image = is_image;
  key.src_access = src_access;
  key.dst_access = dst_access;
  key.src_stage = src_stage;
  key.dst_stage = dst_stage;
  key.old_layout = old_layout;
  key.new_layout = new_layout;
  barrier_tuple_frequency_map_[key]++;
}

void VulkanCommandProcessor::RecordBarrierBreakAttribution() {
  bool seen_requester[5] = {};

  for (size_t i = 0; i < pending_barriers_buffer_memory_barriers_.size(); ++i) {
    BarrierRequester req = i < pending_buffer_barrier_requesters_.size()
                               ? pending_buffer_barrier_requesters_[i]
                               : BarrierRequester::kOther;
    if (static_cast<size_t>(req) < 5) seen_requester[static_cast<size_t>(req)] = true;
    const auto& b = pending_barriers_buffer_memory_barriers_[i];
    RecordBarrierTuple(req, false, b.srcAccessMask, b.dstAccessMask,
                       current_pending_barrier_.src_stage_mask,
                       current_pending_barrier_.dst_stage_mask,
                       VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_UNDEFINED);
  }

  for (size_t i = 0; i < pending_barriers_image_memory_barriers_.size(); ++i) {
    BarrierRequester req = i < pending_image_barrier_requesters_.size()
                               ? pending_image_barrier_requesters_[i]
                               : BarrierRequester::kOther;
    if (static_cast<size_t>(req) < 5) seen_requester[static_cast<size_t>(req)] = true;
    const auto& img = pending_barriers_image_memory_barriers_[i];
    RecordBarrierTuple(req, true, img.srcAccessMask, img.dstAccessMask,
                       current_pending_barrier_.src_stage_mask,
                       current_pending_barrier_.dst_stage_mask,
                       img.oldLayout, img.newLayout);
  }

  if (seen_requester[static_cast<size_t>(BarrierRequester::kSharedMemory)]) current_frame_barrier_breaks_shmem_++;
  if (seen_requester[static_cast<size_t>(BarrierRequester::kTextureCache)]) current_frame_barrier_breaks_texture_++;
  if (seen_requester[static_cast<size_t>(BarrierRequester::kRenderTargetCache)]) current_frame_barrier_breaks_rt_++;
  if (seen_requester[static_cast<size_t>(BarrierRequester::kPrimitiveProcessor)]) current_frame_barrier_breaks_prim_++;
  if (seen_requester[static_cast<size_t>(BarrierRequester::kOther)]) current_frame_barrier_breaks_other_++;
}

static const char* RequesterName(VulkanCommandProcessor::BarrierRequester req) {
  switch (req) {
    case VulkanCommandProcessor::BarrierRequester::kSharedMemory: return "shared_memory";
    case VulkanCommandProcessor::BarrierRequester::kTextureCache: return "texture_cache";
    case VulkanCommandProcessor::BarrierRequester::kRenderTargetCache: return "render_target_cache";
    case VulkanCommandProcessor::BarrierRequester::kPrimitiveProcessor: return "primitive_processor";
    default: return "other";
  }
}

void VulkanCommandProcessor::LogTopBarrierTuples() {
  if (barrier_tuple_frequency_map_.empty()) return;

  std::vector<std::pair<BarrierTupleKey, uint64_t>> sorted(
      barrier_tuple_frequency_map_.begin(), barrier_tuple_frequency_map_.end());
  std::sort(sorted.begin(), sorted.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  constexpr VkAccessFlags kWriteAccess =
      VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
      VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT |
      VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_MEMORY_WRITE_BIT;

  REXGPU_INFO("=== TOP 10 BARRIER TUPLES CAUSING RENDER PASS BREAKS ({} unique tuples) ===",
              sorted.size());
  size_t count = std::min<size_t>(10, sorted.size());
  for (size_t i = 0; i < count; ++i) {
    const auto& key = sorted[i].first;
    uint64_t freq = sorted[i].second;
    bool read_to_read = ((key.src_access & kWriteAccess) == 0 && (key.dst_access & kWriteAccess) == 0);
    bool layout_widening = key.is_image && (key.old_layout == key.new_layout) && ((key.src_access & kWriteAccess) == 0);
    bool redundant = read_to_read || layout_widening;
    REXGPU_INFO(
        "#{}: count={} requester={} type={} srcStage=0x{:X} dstStage=0x{:X} "
        "srcAccess=0x{:X} dstAccess=0x{:X} oldLayout={} newLayout={} [REDUNDANT: {}{}{}]",
        i + 1, freq, RequesterName(key.requester), key.is_image ? "image" : "buffer",
        key.src_stage, key.dst_stage, key.src_access, key.dst_access,
        key.is_image ? std::to_string(key.old_layout) : "N/A",
        key.is_image ? std::to_string(key.new_layout) : "N/A",
        redundant ? "YES" : "NO",
        read_to_read ? " (read->read)" : "",
        layout_widening ? " (same-layout widening)" : "");
  }
  REXGPU_INFO("=========================================================================");
}

void VulkanCommandProcessor::LogScopeProfileAggregateSummary() {
  if (!accum_scope_frames_) return;
  const double frames = double(accum_scope_frames_);
  const double scope_ms = accum_scope_main_ms_ + accum_scope_transfer_ms_ +
      accum_scope_resolve_ms_ + accum_scope_compute_ms_;
  const bool timed = accum_scope_records_ && accum_scope_timed_records_ == accum_scope_records_ &&
      scope_ms > 0.0;
  REXGPU_INFO("SCOPE_SUMMARY frames={} timed={}/{} saturated_frames={} timing={} "
              "gpu_frame_ms={} gpu_complete_interval_ms={} completion_samples={}/{} swap_interval_ms={:.6f}",
      accum_scope_frames_, accum_scope_timed_records_, accum_scope_records_,
      accum_scope_saturated_frames_, timed ? "available" : "unavailable_partial_or_zero_spans",
      timestamp_valid_bits_ ? fmt::format("{:.6f}", accum_scope_gpu_frame_ms_ / frames) : "N/A",
      accum_scope_complete_samples_ ? fmt::format("{:.6f}", accum_scope_complete_interval_ms_ /
          double(accum_scope_complete_samples_)) : "N/A",
      accum_scope_complete_samples_, accum_scope_frames_, accum_scope_swap_interval_ms_ / frames);
  const char* names[] = {"Main", "Transfer", "Resolve(inclusive)", "Compute"};
  const double counts[] = {accum_scope_main_count_, accum_scope_transfer_count_,
      accum_scope_resolve_count_, accum_scope_compute_count_};
  const double times[] = {accum_scope_main_ms_, accum_scope_transfer_ms_,
      accum_scope_resolve_ms_, accum_scope_compute_ms_};
  for (size_t i = 0; i < 4; ++i) {
    REXGPU_INFO("SCOPE_CATEGORY category={} count_total={} count_per_frame={:.6f} ms_per_frame={}",
        names[i], counts[i], counts[i] / frames,
        timed ? fmt::format("{:.6f}", times[i] / frames) : "N/A");
  }
  // Resolve is an inclusive envelope, not additive to its nested transfer/compute work.
  std::vector<std::pair<std::string, ScopeArchetype>> sorted(scope_archetypes_.begin(), scope_archetypes_.end());
  std::sort(sorted.begin(), sorted.end(), [timed](const auto& a, const auto& b) {
    if (timed && a.second.total_gpu_time_ms != b.second.total_gpu_time_ms)
      return a.second.total_gpu_time_ms > b.second.total_gpu_time_ms;
    if (a.second.occurrence_count != b.second.occurrence_count)
      return a.second.occurrence_count > b.second.occurrence_count;
    if (a.second.total_draws != b.second.total_draws)
      return a.second.total_draws > b.second.total_draws;
    return a.first < b.first;
  });
  for (size_t i = 0; i < std::min<size_t>(15, sorted.size()); ++i) {
    const auto& s = sorted[i].second;
    REXGPU_INFO("SCOPE_TOP rank={} category={} size={}x{} format={} color_load/store={}/{} "
                "depth_load/store={}/{} count={} draws={} ms_per_frame={} description={}",
        i + 1, names[size_t(s.category)], s.width, s.height, s.format,
        int(s.color_load_op), int(s.color_store_op), int(s.depth_load_op), int(s.depth_store_op),
        s.occurrence_count, s.total_draws,
        timed ? fmt::format("{:.6f}", s.total_gpu_time_ms / frames) : "N/A", s.description);
  }
}

void VulkanCommandProcessor::FlushCompletedAuditRows(bool force) {
  if (pending_audit_rows_.empty()) return;

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  std::string csv_path = REXCVAR_GET(gpu_frame_audit_csv_path);
  std::ofstream csv;
  if (!csv_path.empty()) {
    csv.open(csv_path, std::ios::app);
  }

  while (!pending_audit_rows_.empty()) {
    const auto& row = pending_audit_rows_.front();
    uint64_t closing_sub = row.closing_submission;
    if (closing_sub > submission_completed_ && !force) {
      break;
    }

    if (transfer_profile_pool_ && row.transfer_profile_count) {
      std::vector<uint64_t> values(row.transfer_profile_count);
      VkResult result = dfn.vkGetQueryPoolResults(device, transfer_profile_pool_,
          (row.frame % kMaxTimestampFramesInFlight) * kTransferQueriesPerFrame,
          row.transfer_profile_count, values.size() * sizeof(uint64_t), values.data(),
          sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
      if (result == VK_NOT_READY && !force) break;
      if (result == VK_SUCCESS) {
        uint64_t mask = timestamp_valid_bits_ == 64 ? ~0ULL : ((1ULL << timestamp_valid_bits_) - 1);
        double total_ms = 0.0;
        for (size_t i = 0; i < values.size(); i += 2)
          total_ms += double((values[i+1] - values[i]) & mask) * timestamp_period_ns_ / 1e6;
        REXGPU_INFO("TRANSFER_PROFILE frame={} batches={} gpu_span_ms={:.6f} saturated={}",
            row.frame, values.size()/2, total_ms, values.size() == kTransferQueriesPerFrame);
      } else {
        REXGPU_WARN("TRANSFER_PROFILE frame={} unavailable result={}", row.frame, int(result));
      }
    }
    double gpu_submit_ms = 0.0;
    double gpu_frame_ms = 0.0;
    if (timestamp_query_pool_ != VK_NULL_HANDLE && timestamp_valid_bits_ > 0) {
      uint32_t slot = row.frame % kMaxTimestampFramesInFlight;
      uint32_t sub_count = row.submission_count;
      if (sub_count > 0) {
        uint32_t q_count = sub_count * 2;
        std::vector<uint64_t> timestamps(q_count, 0);
        VkQueryResultFlags flags = VK_QUERY_RESULT_64_BIT;
        if (force) {
          flags |= VK_QUERY_RESULT_WAIT_BIT;
        }
        VkResult res = dfn.vkGetQueryPoolResults(
            device, timestamp_query_pool_, slot * kMaxQueriesPerFrameSlot, q_count,
            sizeof(uint64_t) * q_count, timestamps.data(), sizeof(uint64_t),
            flags);
        if (res == VK_NOT_READY && !force) {
          // Query not available yet; wait for subsequent flush
          break;
        }
        if (res == VK_SUCCESS) {
          uint64_t mask = (timestamp_valid_bits_ == 64) ? ~0ULL : ((1ULL << timestamp_valid_bits_) - 1);
          for (uint32_t s = 0; s < sub_count; ++s) {
            uint64_t s_start = timestamps[s * 2];
            uint64_t s_end = timestamps[s * 2 + 1];
            if (s_start > 0 && s_end > 0 && s_start != ~0ULL && s_end != ~0ULL) {
              if (s_end >= s_start) {
                gpu_submit_ms += double(s_end - s_start) * timestamp_period_ns_ / 1e6;
              } else if (timestamp_valid_bits_ < 64) {
                gpu_submit_ms += double((mask - s_start) + s_end + 1) * timestamp_period_ns_ / 1e6;
              }
            }
          }
          uint64_t f_start = timestamps[0];
          uint64_t f_end = timestamps[(sub_count - 1) * 2 + 1];
          if (f_start > 0 && f_end > 0 && f_start != ~0ULL && f_end != ~0ULL) {
            if (f_end >= f_start) {
              gpu_frame_ms = double(f_end - f_start) * timestamp_period_ns_ / 1e6;
            } else if (timestamp_valid_bits_ < 64) {
              gpu_frame_ms = double((mask - f_start) + f_end + 1) * timestamp_period_ns_ / 1e6;
            } else {
              gpu_frame_ms = gpu_submit_ms;
            }
          } else {
            gpu_frame_ms = gpu_submit_ms;
          }
        }
      }
    }

    double gpu_complete_interval_ms = 0.0;
    const bool completion_available =
        gpu_completion_tracker_.GetCompletionInterval(row.frame, &gpu_complete_interval_ms);

    if (REXCVAR_GET(gpu_profile_scope_timestamps)) {
      ++accum_scope_frames_;
      accum_scope_gpu_frame_ms_ += gpu_frame_ms;
      if (completion_available) {
        accum_scope_complete_interval_ms_ += gpu_complete_interval_ms;
        ++accum_scope_complete_samples_;
      }
      accum_scope_swap_interval_ms_ += row.swap_interval_ms;
      accum_scope_saturated_frames_ += row.scopes.size() == kMaxScopesPerFrame;
      // Availability is queried without WAIT: unsupported or incomplete queries
      // must never block the audit or masquerade as zero-cost GPU work.
      std::vector<uint64_t> values(row.scopes.size() * 4, 0);
      VkResult result = VK_NOT_READY;
      if (scope_profile_pool_ && timestamp_valid_bits_ && !values.empty()) {
        result = dfn.vkGetQueryPoolResults(device, scope_profile_pool_,
            (row.frame % kMaxTimestampFramesInFlight) * kScopeQueriesPerFrame,
            uint32_t(row.scopes.size() * 2), values.size() * sizeof(uint64_t),
            values.data(), 2 * sizeof(uint64_t),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT);
      }
      double counts[4] = {}, times[4] = {};
      uint64_t timed = 0;
      for (size_t i = 0; i < row.scopes.size(); ++i) {
        const auto& scope = row.scopes[i];
        const bool readable = (result == VK_SUCCESS || result == VK_NOT_READY) &&
            scope.query_start != UINT32_MAX;
        const auto duration = ScopeProfileMilliseconds(timestamp_valid_bits_, timestamp_period_ns_,
            scope.ended && readable, values[i * 4], values[i * 4 + 1],
            values[i * 4 + 2], values[i * 4 + 3]);
        bool available = duration.has_value();
        double ms = duration.value_or(0.0);
        timed += available;
        size_t category = size_t(scope.category);
        ++counts[category];
        times[category] += ms;
        for (const auto& [shader_hash, group] : scope.pixel_shaders) {
          REXGPU_INFO("[RASTER_SCOPE] frame={} scope={} category={} width={} height={} "
                      "total_draws={} ps={:016X} writes_depth={} kills_pixels={} "
                      "ps_draws={} overflow_or_unattributed_draws={}",
                      row.frame, i, category, scope.width, scope.height, scope.draw_count,
                      shader_hash, uint32_t(group.writes_depth), uint32_t(group.kills_pixels),
                      group.draw_count, scope.pixel_shader_overflow_draw_count);
        }
        std::string key = fmt::format("{}:{}x{}:{}:{}:{}:{}:{}:{}",
            category, scope.width, scope.height, scope.format,
            int(scope.color_load_op), int(scope.color_store_op),
            int(scope.depth_load_op), int(scope.depth_store_op), scope.description);
        auto& archetype = scope_archetypes_[key];
        archetype.category = scope.category;
        archetype.width = scope.width;
        archetype.height = scope.height;
        archetype.format = scope.format;
        archetype.color_load_op = scope.color_load_op;
        archetype.color_store_op = scope.color_store_op;
        archetype.depth_load_op = scope.depth_load_op;
        archetype.depth_store_op = scope.depth_store_op;
        archetype.description = scope.description;
        archetype.total_gpu_time_ms += ms;
        archetype.total_draws += scope.draw_count;
        ++archetype.occurrence_count;
      }
      accum_scope_records_ += row.scopes.size();
      accum_scope_timed_records_ += timed;
      accum_scope_main_count_ += counts[0]; accum_scope_main_ms_ += times[0];
      accum_scope_transfer_count_ += counts[1]; accum_scope_transfer_ms_ += times[1];
      accum_scope_resolve_count_ += counts[2]; accum_scope_resolve_ms_ += times[2];
      accum_scope_compute_count_ += counts[3]; accum_scope_compute_ms_ += times[3];
      REXGPU_INFO("SCOPE_PROFILE frame={} main={} transfer={} resolve={} compute={} timed={}/{}",
          row.frame, counts[0], counts[1], counts[2], counts[3], timed, row.scopes.size());
    }

    uint64_t lag = frame_current_ > row.frame ? (frame_current_ - row.frame) : 0;
    REXGPU_INFO(
        "[FRAME_AUDIT_GPU] frame={} lag={} gpu_submit_ms={:.3f} gpu_frame_ms={:.3f} "
        "gpu_complete_interval_ms={:.3f}",
        row.frame, lag, gpu_submit_ms, gpu_frame_ms, gpu_complete_interval_ms);

    if (csv.is_open()) {
      if (csv.tellp() == std::streampos(0)) {
        csv << "frame,draws,wall_ms,total_draw_ms,prim_ms,rts_ms,pipe_ms,bind_ms,shmem_ms,"
               "bar_rp_ms,vk_draw_ms,fence_wait_ms,fence_waits,other_ms,"
               "resolve_attempts,resolve_success,resolve_empty,resolve_bytes,"
               "color_clears,depth_clears,passes,swap_interval_ms,main_passes,transfer_passes,"
               "main_render_area_pixels,main_full_area_pixels,bounded_main_passes,"
               "scissor_render_area_enabled,exec_submit_ms,submit_count,copy_resolve_ms,copy_count,"
               "swap_pre_ms,pre_wall_fence_ms,pre_wall_fence_count,break_transfer,break_framebuffer,break_depth,"
               "break_stencil,break_color,break_clear,break_area,break_barrier,break_force,break_submission,"
               "occlusion_wait_ms,occlusion_wait_count,memexport_wait_ms,memexport_wait_count,"
               "timeline_other_ms,ring_idle_ms,wait_mem_ms,wait_reg_ms,pm4_exclusive_ms,"
               "draw_exclusive_ms,copy_exclusive_ms,swap_exclusive_ms,exec_ms,queue_submit_ms,fence_in_wall_ms,"
               "wait_mem_count,wait_reg_count,wait_top_address,wait_top_is_memory,wait_top_polls,wait_address_overflow,"
               "interrupt_ms,swap_packet_ms,coherent_ms,pending_fns_ms,end_command_buffer_ms,"
               "reset_fence_ms,queue_acquire_ms,queue_acquire_count,"
               "gpu_submit_ms,gpu_frame_ms,gpu_complete_interval_ms,"
               "break_barrier_shmem,break_barrier_texture,break_barrier_rt,break_barrier_prim,break_barrier_other,"
               "shmem_hoist_fallbacks,shmem_hoist_successes,"
               "break_growth_only,break_growth_framebuffer,break_framebuffer_only,"
               "retained_depth_only_draws,transfer_guest_scope_reuses,dyn_ms,shader_translate_ms,"
               "sampler_prep_ms,texture_request_ms,sh_samp_ms\n";
      }
      double total_draw_ms = row.cpu_stats.time_draw_total.count() / 1e6;
      double prim_ms = row.cpu_stats.time_primitive_processing.count() / 1e6;
      double shader_translate_ms = row.cpu_stats.time_shader_translation.count() / 1e6;
      double sampler_prep_ms = row.cpu_stats.time_sampler_preparation.count() / 1e6;
      double texture_request_ms = row.cpu_stats.time_texture_request.count() / 1e6;
      double rts_ms = row.cpu_stats.time_render_target_update.count() / 1e6;
      double pipe_ms = row.cpu_stats.time_pipeline_configuration.count() / 1e6;
      double dyn_state_ms = row.cpu_stats.time_dynamic_state.count() / 1e6;
      double bind_ms = row.cpu_stats.time_binding_updates.count() / 1e6;
      double shmem_ms = row.cpu_stats.time_shared_memory_residency.count() / 1e6;
      double bar_rp_ms = row.cpu_stats.time_barriers_and_render_pass.count() / 1e6;
      double vk_draw_ms = row.cpu_stats.time_vk_draw.count() / 1e6;
      double fence_wait_ms = row.cpu_stats.time_fence_waits.count() / 1e6;
      // Keep this residual aligned with the text audit's named draw phases.
      // Opening-fence waits may be part of time_draw_total and therefore remain
      // in other_ms; fence_wait_ms is an overlapping diagnostic counter.
      double other_ms = total_draw_ms - (prim_ms + shader_translate_ms + sampler_prep_ms +
                                         texture_request_ms + rts_ms + pipe_ms + dyn_state_ms +
                                         bind_ms + shmem_ms + bar_rp_ms + vk_draw_ms);

      csv << row.frame << ',' << row.cpu_stats.draw_count << ','
          << fmt::format("{:.3f}", row.wall_ms) << ',' << fmt::format("{:.3f}", total_draw_ms) << ','
          << fmt::format("{:.3f}", prim_ms) << ',' << fmt::format("{:.3f}", rts_ms) << ','
          << fmt::format("{:.3f}", pipe_ms) << ',' << fmt::format("{:.3f}", bind_ms) << ','
          << fmt::format("{:.3f}", shmem_ms) << ',' << fmt::format("{:.3f}", bar_rp_ms) << ','
          << fmt::format("{:.3f}", vk_draw_ms) << ',' << fmt::format("{:.3f}", fence_wait_ms) << ','
          << row.cpu_stats.fence_wait_count << ','
          << fmt::format("{:.3f}", other_ms) << ','
          << row.work_counters.resolve_attempts << ','
          << row.work_counters.resolve_successes << ','
          << row.work_counters.resolve_empty_or_noop << ','
          << row.work_counters.resolve_bytes << ','
          << row.work_counters.color_clears << ','
          << row.work_counters.depth_clears << ','
          << row.pass_stats.pass_count << ','
          << (row.swap_interval_ms > 0.0
                  ? fmt::format("{:.3f}", row.swap_interval_ms) : std::string{}) << ','
          << row.pass_stats.main_pass_count << ','
          << row.pass_stats.transfer_pass_count << ','
          << row.pass_stats.main_render_area_pixels << ','
          << row.pass_stats.main_full_area_pixels << ','
          << row.pass_stats.bounded_main_pass_count << ','
          << uint32_t(REXCVAR_GET(vulkan_scissor_render_area)) << ','
          << fmt::format("{:.3f}", row.cpu_stats.time_exec_submit.count() / 1e6) << ','
          << row.cpu_stats.submit_count << ','
          << fmt::format("{:.3f}", row.cpu_stats.time_copy.count() / 1e6) << ','
          << row.cpu_stats.copy_count << ','
          << fmt::format("{:.3f}", row.cpu_stats.time_swap_pre.count() / 1e6) << ','
          << fmt::format("{:.3f}", row.cpu_stats.time_pre_wall_fence.count() / 1e6) << ','
          << row.cpu_stats.pre_wall_fence_count << ','
          << row.cpu_stats.break_transfer << ',' << row.cpu_stats.break_framebuffer << ','
          << row.cpu_stats.break_depth << ',' << row.cpu_stats.break_stencil << ','
          << row.cpu_stats.break_color << ',' << row.cpu_stats.break_clear << ','
          << row.cpu_stats.break_area << ',' << row.cpu_stats.break_barrier << ','
          << row.cpu_stats.break_force << ',' << row.cpu_stats.break_submission << ','
          << fmt::format("{:.3f}", row.cpu_stats.time_occlusion_wait.count() / 1e6) << ','
          << row.cpu_stats.occlusion_wait_count << ','
          << fmt::format("{:.3f}", row.cpu_stats.time_memexport_wait.count() / 1e6) << ','
          << row.cpu_stats.memexport_wait_count;

      auto append_duration = [&](size_t i) {
        csv << ',';
        if (REXCVAR_GET(gpu_frame_timeline_enable)) csv << fmt::format("{:.6f}", row.timeline_durations[i].count() / 1e6);
      };
      for (size_t i = 0; i < 11; ++i) append_duration(i);
      if (REXCVAR_GET(gpu_frame_timeline_enable)) {
        csv << ',' << row.timeline_wait_counts[0] << ',' << row.timeline_wait_counts[1]
            << ',' << uint32_t(row.timeline_top_key) << ',' << (row.timeline_top_key >> 32) << ',' << row.timeline_top_polls
            << ',' << row.timeline_wait_overflow;
      } else {
        csv << ",,,,,,";
      }
      for (size_t i = 11; i < row.timeline_durations.size(); ++i) append_duration(i);
      csv << ',';
      if (REXCVAR_GET(gpu_frame_timeline_enable)) csv << row.queue_acquire_count;
      csv << ',' << fmt::format("{:.3f}", gpu_submit_ms)
          << ',' << fmt::format("{:.3f}", gpu_frame_ms)
          << ',' << fmt::format("{:.3f}", gpu_complete_interval_ms)
          << ',' << row.break_barrier_shmem
          << ',' << row.break_barrier_texture
          << ',' << row.break_barrier_rt
          << ',' << row.break_barrier_prim
          << ',' << row.break_barrier_other
          << ',' << row.shmem_hoist_fallbacks
          << ',' << row.shmem_hoist_successes
          << ',' << row.cpu_stats.break_growth_only
          << ',' << row.cpu_stats.break_growth_framebuffer
          << ',' << row.cpu_stats.break_framebuffer_only
          << ',' << row.cpu_stats.retained_depth_only_draws
          << ',' << row.cpu_stats.transfer_guest_scope_reuses
          << ',' << fmt::format("{:.3f}", dyn_state_ms)
          << ',' << fmt::format("{:.3f}", shader_translate_ms)
          << ',' << fmt::format("{:.3f}", sampler_prep_ms)
          << ',' << fmt::format("{:.3f}", texture_request_ms)
          << ',' << fmt::format("{:.3f}", shader_translate_ms + sampler_prep_ms + texture_request_ms)
          << '\n';
      csv.flush();
    }

    pending_audit_rows_.pop_front();
  }
}

bool VulkanCommandProcessor::IsResolveChainAuditFrame() const {
  const uint64_t end = REXCVAR_GET(gpu_frame_audit_window_end);
  return REXCVAR_GET(gpu_resolve_chain_audit) && REXCVAR_GET(gpu_frame_audit_enable) &&
      frame_open_ && frame_current_ >= REXCVAR_GET(gpu_frame_audit_window_start) &&
      (!end || frame_current_ <= end);
}

bool VulkanCommandProcessor::IsGpuFrameAuditWindow() const {
  const uint64_t end = REXCVAR_GET(gpu_frame_audit_window_end);
  return REXCVAR_GET(gpu_frame_audit_enable) && frame_open_ &&
         frame_current_ >= REXCVAR_GET(gpu_frame_audit_window_start) &&
         (!end || frame_current_ <= end);
}

void VulkanCommandProcessor::RecordFrameAuditRow(uint64_t frame, double wall_ms) {
  double total_draw_ms = frame_cpu_profiling_stats_.time_draw_total.count() / 1e6;
  double prim_ms = frame_cpu_profiling_stats_.time_primitive_processing.count() / 1e6;
  double shader_translate_ms = frame_cpu_profiling_stats_.time_shader_translation.count() / 1e6;
  double sampler_prep_ms = frame_cpu_profiling_stats_.time_sampler_preparation.count() / 1e6;
  double texture_request_ms = frame_cpu_profiling_stats_.time_texture_request.count() / 1e6;
  double rts_ms = frame_cpu_profiling_stats_.time_render_target_update.count() / 1e6;
  double pipe_ms = frame_cpu_profiling_stats_.time_pipeline_configuration.count() / 1e6;
  double dyn_state_ms = frame_cpu_profiling_stats_.time_dynamic_state.count() / 1e6;
  double bind_ms = frame_cpu_profiling_stats_.time_binding_updates.count() / 1e6;
  double shmem_ms = frame_cpu_profiling_stats_.time_shared_memory_residency.count() / 1e6;
  double bar_rp_ms = frame_cpu_profiling_stats_.time_barriers_and_render_pass.count() / 1e6;
  double vk_draw_ms = frame_cpu_profiling_stats_.time_vk_draw.count() / 1e6;
  double fence_wait_ms = frame_cpu_profiling_stats_.time_fence_waits.count() / 1e6;
  const double shader_samp_ms = shader_translate_ms + sampler_prep_ms + texture_request_ms;
  // The named preparation and draw phases are disjoint. Opening-fence waits may
  // occur inside the draw timer, so other_ms can contain that wait; the
  // separate pre-wall counter and aggregate fence counter overlap this residual.
  double other_ms = total_draw_ms - (prim_ms + shader_translate_ms + sampler_prep_ms +
                                     texture_request_ms + rts_ms + pipe_ms + dyn_state_ms +
                                     bind_ms + shmem_ms + bar_rp_ms + vk_draw_ms);

  REXGPU_INFO(
      "[FRAME_AUDIT] frame={} draws={} wall_ms={:.3f} total_draw_ms={:.3f} "
      "prim_ms={:.3f} sh_samp_ms={:.3f} rts_ms={:.3f} pipe_ms={:.3f} dyn_ms={:.3f} "
      "bind_ms={:.3f} shmem_ms={:.3f} "
      "bar_rp_ms={:.3f} vk_draw_ms={:.3f} fence_wait_ms={:.3f} fence_waits={} "
      "other_ms={:.3f} "
      "resolves={}/{}/{} res_bytes={} color_clears={} depth_clears={} passes={} "
      "merged_passes={} elided_transfers={} shader_translate_ms={:.3f} sampler_prep_ms={:.3f} "
      "texture_request_ms={:.3f} pre_wall_fence_ms={:.3f} pre_wall_fence_count={}",
      frame, frame_cpu_profiling_stats_.draw_count, wall_ms, total_draw_ms,
      prim_ms, shader_samp_ms, rts_ms, pipe_ms, dyn_state_ms, bind_ms, shmem_ms,
      bar_rp_ms, vk_draw_ms, fence_wait_ms, frame_cpu_profiling_stats_.fence_wait_count, other_ms,
      frame_work_counters_.resolve_attempts, frame_work_counters_.resolve_successes,
      frame_work_counters_.resolve_empty_or_noop, frame_work_counters_.resolve_bytes,
      frame_work_counters_.color_clears, frame_work_counters_.depth_clears,
      frame_pass_stats_.pass_count, frame_pass_stats_.merged_guest_passes,
      frame_pass_stats_.elided_dead_transfers, shader_translate_ms, sampler_prep_ms,
      texture_request_ms, frame_cpu_profiling_stats_.time_pre_wall_fence.count() / 1e6,
      frame_cpu_profiling_stats_.pre_wall_fence_count);

  REXGPU_INFO("[WORK_PLACEMENT] frame={} render_scopes={} interrupted_gaps={} "
              "unattributed_gaps={} attribution=command_stream_not_metal",
              frame, work_purpose_audit_.render_scopes, work_purpose_audit_.interrupted_gaps,
              work_purpose_audit_.unattributed_gaps);
  for (const auto& [key, row] : work_purpose_audit_.rows) {
    REXGPU_INFO("[WORK_PURPOSE] frame={} purpose={} x={} y={} z={} bytes_each={} count={} "
                "prologue={} before_render={} between={} tail={}",
                frame, kWorkPurposeNames[size_t(key.purpose)], key.x, key.y, key.z, key.bytes,
                row.count, row.prologue, row.before_render, row.between, row.tail);
  }
  for (size_t i = 0; i < work_purpose_audit_.gaps_by_cause.size(); ++i) {
    if (work_purpose_audit_.gaps_by_cause[i])
      REXGPU_INFO("[WORK_SPLIT_CAUSE] frame={} purpose={} gaps={} co_causes_nonadditive=true",
                  frame, kWorkPurposeNames[i], work_purpose_audit_.gaps_by_cause[i]);
  }

  PendingFrameAuditRow pending{};
  pending.frame = frame;
  pending.closing_submission = GetCurrentSubmission() - 1;
  pending.submission_count = frame_submission_counts_[frame % kMaxTimestampFramesInFlight];
  pending.transfer_profile_count = transfer_profile_counts_[frame % kMaxTimestampFramesInFlight];
  pending.scopes = frame_scopes_[frame % kMaxTimestampFramesInFlight];
  pending.wall_ms = wall_ms;
  pending.cpu_stats = frame_cpu_profiling_stats_;
  pending.work_counters = frame_work_counters_;
  pending.pass_stats = frame_pass_stats_;
  pending.swap_interval_ms = frame_swap_interval_ms_;
  const auto& durations = GetFrameCpuTimeline().durations();
  for (size_t i = 0; i < std::min(durations.size(), pending.timeline_durations.size()); ++i) {
    pending.timeline_durations[i] = durations[i];
  }
  FrameCpuTimeline::PollEntry top{};
  for (const auto& entry : GetFrameCpuTimeline().wait_polls()) {
    if (entry.polls > top.polls) top = entry;
  }
  pending.timeline_top_key = top.key;
  pending.timeline_top_polls = top.polls;
  pending.timeline_wait_counts[0] = GetFrameCpuTimeline().wait_counts()[0];
  pending.timeline_wait_counts[1] = GetFrameCpuTimeline().wait_counts()[1];
  pending.timeline_wait_overflow = GetFrameCpuTimeline().wait_overflow();
  pending.queue_acquire_count = GetFrameCpuTimeline().queue_acquire_count();
  pending.break_barrier_shmem = current_frame_barrier_breaks_shmem_;
  pending.break_barrier_texture = current_frame_barrier_breaks_texture_;
  pending.break_barrier_rt = current_frame_barrier_breaks_rt_;
  pending.break_barrier_prim = current_frame_barrier_breaks_prim_;
  pending.break_barrier_other = current_frame_barrier_breaks_other_;
  pending.shmem_hoist_fallbacks = uint32_t(current_frame_shmem_hoist_fallbacks_);
  pending.shmem_hoist_successes = uint32_t(current_frame_shmem_hoist_successes_);

  pending_audit_rows_.push_back(pending);

  current_frame_barrier_breaks_shmem_ = 0;
  current_frame_barrier_breaks_texture_ = 0;
  current_frame_barrier_breaks_rt_ = 0;
  current_frame_barrier_breaks_prim_ = 0;
  current_frame_barrier_breaks_other_ = 0;
  current_frame_shmem_hoist_fallbacks_ = 0;
  current_frame_shmem_hoist_successes_ = 0;
}

void VulkanCommandProcessor::FlushRasterVariantAudit() {
  for (size_t i = 0; i < raster_variant_audit_.used; ++i) {
    const auto& bucket = raster_variant_audit_.buckets[i];
    const auto& d = bucket.decision;
    // Tri-state gate fields: 0=not evaluated, 1=false, 2=true. Alpha func,
    // guest MSAA and normalized depth fields use the guest register enums.
    REXGPU_INFO("[RASTER_VARIANT] frame={} ps={:016X} mod={:016X} draws={} depth_mode={} "
                "host_rt={} float24={} implicit_early={} writes_rt0={} coverage_initial={} "
                "coverage_final={} alpha_to_mask={} alpha_func={} compare_pass={} "
                "ref_le_128={} c102_ge_999={} tc_present={} fetch0_opaque={} "
                "guest_msaa={} depth_test={} depth_write={} depth_compare={}",
                frame_current_, d.shader_hash, d.modification, bucket.draws, d.depth_mode,
                uint32_t(d.host_render_targets), uint32_t(d.float24_conversion),
                uint32_t(d.implicit_early_z_write), uint32_t(d.writes_color_target0),
                uint32_t(d.coverage_alpha_initial), uint32_t(d.coverage_alpha_final),
                uint32_t(d.alpha_to_mask), d.alpha_func, uint32_t(d.alpha_compare_pass),
                uint32_t(d.alpha_ref_le_128), uint32_t(d.c102_w_ge_999),
                uint32_t(d.texture_cache_present), uint32_t(d.fetch0_opaque),
                d.guest_msaa, d.depth_test, d.depth_write, d.depth_compare);
  }
  for (size_t i = 0; i < raster_variant_audit_.alpha_pairs_used; ++i) {
    const auto& pair = raster_variant_audit_.alpha_pairs[i];
    REXGPU_INFO("[RASTER_ALPHA] frame={} ps={:016X} mod={:016X} validity={} "
                "alpha_ref_bits={:08X} c102_w_bits={:08X} draws={}",
                frame_current_, pair.shader_hash, pair.modification, pair.validity,
                pair.alpha_ref_bits, pair.c102_w_bits, pair.draws);
  }
  REXGPU_INFO("[RASTER_ALPHA_TOTAL] frame={} target_draws={} counted={} overflow={} invalid={}",
              frame_current_, raster_variant_audit_.target_draws,
              raster_variant_audit_.alpha_counted, raster_variant_audit_.alpha_overflow,
              raster_variant_audit_.alpha_invalid);
  REXGPU_INFO("[RASTER_VARIANT_TOTAL] frame={} target_draws={} counted={} overflow={} "
              "invalid={} buckets={} all_guest_draws={}", frame_current_,
              raster_variant_audit_.target_draws, raster_variant_audit_.counted,
              raster_variant_audit_.overflow, raster_variant_audit_.invalid,
              raster_variant_audit_.used, frame_cpu_profiling_stats_.draw_count);
  raster_variant_audit_.Clear();
}

bool VulkanCommandProcessor::IssueDraw(xenos::PrimitiveType prim_type, uint32_t index_count,
                                       IndexBufferInfo* index_buffer_info,
                                       bool major_mode_explicit) {
  FrameCpuTimeline::Scope audit_draw(GetFrameCpuTimeline(), FrameCpuTimeline::Bucket::kDraw);
  bool diagnostic_capture_active = false;
  bool diagnostic_vertex_output_capture_active = false;
  bool diagnostic_draw_recorded = false;
  uint32_t diagnostic_draw_ordinal = 0;
  uint64_t diagnostic_vertex_hash = 0;
  uint64_t diagnostic_pixel_hash = 0;
  const char* diagnostic_capture_error = nullptr;
  const char* diagnostic_selection_stage = "before_pixel_analysis";
  uint64_t diagnostic_effective_pixel_hash = 0;
  const uint64_t diagnostic_raw_vertex_hash =
      active_vertex_shader() ? active_vertex_shader()->ucode_data_hash() : 0;
  const uint64_t diagnostic_raw_pixel_hash =
      active_pixel_shader() ? active_pixel_shader()->ucode_data_hash() : 0;
  bool diagnostic_log_candidate = false;
  auto report_diagnostic_outcome = [&](std::string_view status, uint32_t count, bool overflow,
                                       std::string_view error) {
    LogFragmentDiagnosticOutcome(frame_current_, diagnostic_vertex_hash, diagnostic_pixel_hash,
                                 diagnostic_draw_ordinal, status, count, overflow, error);
  };
  auto finish_diagnostic_capture = [&]() {
    if (diagnostic_log_candidate) {
      LogFragmentDiagnostic(fmt::format(
          "Fragment diagnostic candidate: frame={} VS={:016X} PS={:016X} effective_PS={:016X} "
          "draw={} expected_PS={:016X} frame_match={} hash_match={} storage_ready={} "
          "latch_consumed={} stage={}",
          frame_current_, diagnostic_raw_vertex_hash, diagnostic_raw_pixel_hash,
          diagnostic_effective_pixel_hash, diagnostic_storage_draw_log_.ordinal(),
          diagnostics::GetFragmentDiagnosticPixelHash(diagnostic_storage_shader_target_),
          frame_current_ == diagnostic_storage_frame_,
          diagnostic_effective_pixel_hash ==
              diagnostics::GetFragmentDiagnosticPixelHash(diagnostic_storage_shader_target_),
          diagnostic_storage_buffer_ != nullptr, diagnostic_storage_attempt_consumed_,
          diagnostic_selection_stage));
    }
    if (!diagnostic_capture_active)
      return;
    uint64_t submission = submission_open_ ? GetCurrentSubmission() : GetCurrentSubmission() - 1;
    bool completed = !submission_open_ || EndSubmission(false);
    if (completed && submission > submission_completed_) {
      CheckSubmissionFenceAndDeviceLoss(submission);
      completed = !device_lost_ && submission_completed_ >= submission;
    }
    if (!completed || device_lost_) {
      report_diagnostic_outcome("failed", 0, false, "submission_not_completed");
      diagnostic_capture_active = false;
      return;
    }
    bool overflow = false;
    uint32_t captured_count = 0;
    VulkanDiagnosticStorageBuffer::Header diagnostic_header{};
    const bool valid = diagnostic_storage_buffer_->ReadCompleted(
        true, diagnostic_storage_records_, overflow, &captured_count, &diagnostic_header,
        !(diagnostics::QuadPartnerMarkerEnabled() &&
          diagnostics::QuadPartnerOmitsEntryCounters(diagnostics::QuadPartnerDiagnosticRevision())));
    LogFragmentDiagnostic(
        fmt::format("Fragment diagnostic readback: frame={} draw={} valid={} count={} overflow={} "
                    "header_valid={} recording_enabled={} capacity={} abi={}",
                    frame_current_, diagnostic_draw_ordinal, valid, captured_count, overflow,
                    diagnostic_header.valid, diagnostic_header.recording_enabled,
                    diagnostic_header.capacity, diagnostic_header.abi_version));
    if (diagnostic_vertex_output_capture_active && valid && captured_count) {
      std::vector<VulkanDiagnosticStorageBuffer::Record> vertex_output_records;
      vertex_output_records.reserve(captured_count);
      for (uint32_t i = 0; i < captured_count; ++i) {
        const auto& record = diagnostic_storage_records_[i];
        if (record.fetch_slot == UINT32_C(0xFFFFFFFE)) {
          vertex_output_records.push_back(record);
        }
      }
      char path[512];
      const int path_length = std::snprintf(
          path, sizeof(path),
          "/Users/Kushal/Projects/cosmicR/out/face-shoulder-eye-audit-20261023/"
          "frame-%llu-session-%llu-draw-%u-post-host-clip.bin",
          static_cast<unsigned long long>(frame_current_),
          static_cast<unsigned long long>(diagnostic_storage_session_), diagnostic_draw_ordinal);
      FILE* output = path_length > 0 && size_t(path_length) < sizeof(path)
                         ? std::fopen(path, "wb")
                         : nullptr;
      const size_t bytes = output ? std::fwrite(vertex_output_records.data(),
                                                sizeof(VulkanDiagnosticStorageBuffer::Record),
                                                vertex_output_records.size(), output)
                                  : 0;
      const int close_status = output ? std::fclose(output) : -1;
      const bool saved = bytes == vertex_output_records.size() && close_status == 0;
      uint32_t finite_positions = 0;
      for (const auto& record : vertex_output_records) {
        bool finite = true;
        for (uint32_t component = 0; component < 4; ++component) {
          finite &= std::isfinite(rex::memory::Reinterpret<float>(record.output_rgba[component]));
        }
        finite_positions += finite;
      }
      LogFragmentDiagnostic(fmt::format(
          "Vertex output diagnostic readback: frame={} draw={} valid={} records={} overflow={} "
          "fragment_records={} finite_clip_positions={} record_size={} status={} path='{}'",
          frame_current_, diagnostic_draw_ordinal, valid, vertex_output_records.size(), overflow,
          captured_count - vertex_output_records.size(),
          finite_positions, sizeof(VulkanDiagnosticStorageBuffer::Record),
          saved ? "saved" : "save_failed", saved ? path : ""));
      if (diagnostic_storage_combined_vertex_output_mode_) {
        char combined_path[512];
        const int combined_path_length = std::snprintf(
            combined_path, sizeof(combined_path),
            "/Users/Kushal/Projects/cosmicR/out/face-shoulder-eye-audit-20261023/"
            "frame-%llu-session-%llu-draw-%u-combined-records.bin",
            static_cast<unsigned long long>(frame_current_),
            static_cast<unsigned long long>(diagnostic_storage_session_), diagnostic_draw_ordinal);
        FILE* combined_output = combined_path_length > 0 &&
                                        size_t(combined_path_length) < sizeof(combined_path)
                                    ? std::fopen(combined_path, "wb")
                                    : nullptr;
        const size_t combined_bytes = combined_output
                                          ? std::fwrite(diagnostic_storage_records_.data(),
                                                        sizeof(VulkanDiagnosticStorageBuffer::Record),
                                                        captured_count, combined_output)
                                          : 0;
        const int combined_close_status = combined_output ? std::fclose(combined_output) : -1;
        LogFragmentDiagnostic(fmt::format(
            "Combined vertex/fragment diagnostic readback: frame={} draw={} records={} "
            "vertex_records={} fragment_records={} status={} path='{}'",
            frame_current_, diagnostic_draw_ordinal, captured_count, vertex_output_records.size(),
            captured_count - vertex_output_records.size(),
            combined_bytes == captured_count && combined_close_status == 0 ? "saved" : "save_failed",
            combined_bytes == captured_count && combined_close_status == 0 ? combined_path : ""));
      }
    }
    if (!diagnostic_vertex_output_capture_active && diagnostic_storage_census_mode_) {
      const char* roi_name = diagnostic_storage_local_overwrite_census_mode_ ? "local_roi" : "nose_roi";
      LogFragmentDiagnostic(fmt::format(
          "Fragment diagnostic census: frame={} draw={} VS={:016X} PS={:016X} "
          "entry={} entry_{}={} entry_bounds={},{}..{},{} eligible={} eligible_{}={} "
          "recording_enabled={} abi={}",
          frame_current_, diagnostic_draw_ordinal, diagnostic_vertex_hash, diagnostic_pixel_hash,
          diagnostic_header.entry_invocation_count, roi_name, diagnostic_header.entry_nose_roi_count,
          diagnostic_header.entry_min_x, diagnostic_header.entry_min_y,
          diagnostic_header.entry_max_x, diagnostic_header.entry_max_y,
          diagnostic_header.eligible_invocation_count, roi_name,
          diagnostic_header.eligible_nose_roi_count, diagnostic_header.recording_enabled,
          diagnostic_header.abi_version));
    }
    if (!diagnostic_vertex_output_capture_active && !diagnostic_storage_census_mode_ && valid &&
        captured_count) {
      for (uint32_t record_index = 0; record_index < captured_count; ++record_index) {
        const VulkanDiagnosticStorageBuffer::Record& record =
            diagnostic_storage_records_[record_index];
        float depth_z = 0.0f;
        std::memcpy(&depth_z, &record.draw_id, sizeof(float));
        const uint64_t record_shader_hash = (uint64_t(record.shader_hash_hi) << 32) | record.shader_hash_lo;
        if (record_shader_hash == diagnostics::kFragmentDiagnosticDeferredPixelHash &&
            (record.fetch_slot & 0xF0000000u) == 0xF0000000u) {
          if ((record.fetch_slot & 0xFFu) == 0xFFu) {
            continue;
          }
          LogFragmentDiagnostic(fmt::format(
              "Fragment diagnostic deferred register: frame={} draw={} invocation={} "
              "instr={} register=r{} xy={},{} value_bits={:08X},{:08X},{:08X},{:08X}",
              frame_current_, diagnostic_draw_ordinal, record.frame_id / 3,
              record.stage_marker, record.fetch_slot & 0xFF, record.fragment_x,
              record.fragment_y, record.rgba[0], record.rgba[1], record.rgba[2],
              record.rgba[3]));
          continue;
        }
        LogFragmentDiagnostic(
            fmt::format("Fragment diagnostic sample: frame={} draw={} PS={:016X} record={}/{} invocation={} "
                        "slot={} stage={} xy={},{} z={:.6f} uv_bits={:08X},{:08X} "
                        "rgba_bits={:08X},{:08X},{:08X},{:08X}",
                        frame_current_, diagnostic_draw_ordinal, record_shader_hash, record_index, captured_count,
                        record_index / 3, record.fetch_slot, record.stage_marker, record.fragment_x,
                        record.fragment_y, depth_z, record.uv[0], record.uv[1], record.rgba[0],
                        record.rgba[1], record.rgba[2], record.rgba[3]));
        if (record_shader_hash == diagnostics::kFragmentDiagnosticGBufferPixelHash &&
            (record.stage_marker & 0x200) != 0) {
          const uint32_t* varying = record_index % 3 == 0 ? record.output_rgba : record.rgba;
          LogFragmentDiagnostic(fmt::format(
              "Fragment diagnostic interpolator: frame={} draw={} invocation={} register={} "
              "xy={},{} value_bits={:08X},{:08X},{:08X},{:08X}",
              frame_current_, diagnostic_draw_ordinal, record_index / 3,
              record.stage_marker & 0x1FF, record.fragment_x, record.fragment_y, varying[0],
              varying[1], varying[2], varying[3]));
        }
        if (record_shader_hash == diagnostics::kFragmentDiagnosticGBufferPixelHash &&
            record.fetch_slot != UINT32_MAX && (record.fetch_slot & 0x80000000) != 0) {
          LogFragmentDiagnostic(fmt::format(
              "Fragment diagnostic interpolator: frame={} draw={} invocation={} register={} "
              "xy={},{} value_bits={:08X},{:08X},{:08X},{:08X}",
              frame_current_, diagnostic_draw_ordinal, record_index / 3,
              record.fetch_slot & 0x7FFFFFFF, record.fragment_x, record.fragment_y, record.rgba[0],
              record.rgba[1], record.rgba[2], record.rgba[3]));
        }
        if (record_shader_hash == diagnostics::kFragmentDiagnosticGBufferPixelHash &&
            (record.stage_marker & 0x40000000) != 0) {
          LogFragmentDiagnostic(fmt::format(
              "Fragment diagnostic interpolator: frame={} draw={} invocation={} register={} "
              "xy={},{} value_bits={:08X},{:08X},{:08X},{:08X}",
              frame_current_, diagnostic_draw_ordinal, record_index / 3,
              record.stage_marker & 0x1FF, record.fragment_x, record.fragment_y,
              record.output_rgba[0], record.output_rgba[1], record.output_rgba[2],
              record.output_rgba[3]));
        }
        if (record_shader_hash == diagnostics::kFragmentDiagnosticGBufferPixelHash &&
            record_index % 3 == 2) {
          std::string trace_values;
          for (uint32_t value = 0; value < 12; ++value) {
            const auto& v = record.vertex_fetch_inputs[value];
            trace_values += fmt::format("{}{:08X},{:08X},{:08X},{:08X}",
                                        value ? ";" : "", v[0], v[1], v[2], v[3]);
          }
          LogFragmentDiagnostic(fmt::format(
              "Fragment diagnostic G-buffer tf4 stages: frame={} draw={} invocation={} xy={},{} "
              "vectors[0..11]={}",
              frame_current_, diagnostic_draw_ordinal, record_index / 3, record.fragment_x,
              record.fragment_y, trace_values));
        }
        if (diagnostic_storage_local_candidate_output_mode_ &&
            (record.fetch_slot == 14 ||
             (record.fetch_slot == UINT32_MAX && record_index % 3 == 2))) {
          LogFragmentDiagnostic(fmt::format(
              "Fragment diagnostic shader-output: frame={} draw={} PS={:016X} invocation={} xy={},{} z={:.6f} "
              "rgba_bits={:08X},{:08X},{:08X},{:08X}",
              frame_current_, diagnostic_draw_ordinal, record_shader_hash, record_index / 3, record.fragment_x,
              record.fragment_y, depth_z, record.output_rgba[0], record.output_rgba[1],
              record.output_rgba[2], record.output_rgba[3]));
        }
        if (record_shader_hash == diagnostics::kFragmentDiagnosticDeferredPixelHash &&
            record_index % 3 < 2) {
          const uint32_t* input = record.output_rgba;
          LogFragmentDiagnostic(fmt::format(
              "Fragment diagnostic deferred input: frame={} draw={} invocation={} xy={},{} "
              "kind={} value_bits={:08X},{:08X},{:08X},{:08X}",
              frame_current_, diagnostic_draw_ordinal, record_index / 3, record.fragment_x,
              record.fragment_y, record_index % 3 == 0 ? "guest_r0" : "interpolator0",
              input[0], input[1], input[2], input[3]));
        }
        if (diagnostic_storage_local_candidate_output_mode_ &&
            record_shader_hash == diagnostics::kFragmentDiagnosticPixelHash &&
            (record.fetch_slot == 0 || record.fetch_slot == 2)) {
          LogFragmentDiagnostic(fmt::format(
              "Fragment diagnostic gradients: frame={} draw={} invocation={} slot={} xy={},{} "
              "gradient_bits={:08X},{:08X},{:08X},{:08X}",
              frame_current_, diagnostic_draw_ordinal, record_index / 3,
              record.fetch_slot == 0 ? 0 : 14, record.fragment_x, record.fragment_y,
              record.output_rgba[0], record.output_rgba[1], record.output_rgba[2],
              record.output_rgba[3]));
        }
      }
    }
    const bool disabled = diagnostic_storage_buffer_->SetRecordingEnabled(false);
    const bool capture_succeeded = diagnostic_draw_recorded && valid && disabled &&
                                   !diagnostic_capture_error &&
                                   ((diagnostic_storage_census_mode_ &&
                                     !diagnostic_vertex_output_capture_active) ||
                                    captured_count != 0);
    report_diagnostic_outcome(
        capture_succeeded ? "complete" : "failed", captured_count, overflow,
        diagnostic_capture_error    ? diagnostic_capture_error
        : !disabled                 ? "disable_failed"
        : !diagnostic_draw_recorded ? "draw_not_recorded"
        : !valid                    ? (overflow ? "capacity_overflow" : "invalid_readback")
        : !captured_count
            ? (diagnostic_storage_census_mode_ && !diagnostic_vertex_output_capture_active
                   ? "census_only"
                   : "no_tf14_fetch_recorded")
            : "none");
    if (diagnostic_storage_local_overwrite_census_mode_ &&
        diagnostic_pixel_hash == diagnostics::kFragmentDiagnosticPixelHash && valid &&
        diagnostic_header.eligible_nose_roi_count) {
      diagnostic_storage_local_overwrite_tracking_started_ = true;
      LogFragmentDiagnostic(fmt::format(
          "Fragment diagnostic local-overwrite tracking started: frame={} draw={} "
          "eligible_local_roi={}",
          frame_current_, diagnostic_draw_ordinal, diagnostic_header.eligible_nose_roi_count));
    }
    diagnostic_capture_active = false;
    diagnostic_vertex_output_capture_active = false;
  };
  struct DiagnosticScopeGuard {
    decltype(finish_diagnostic_capture)& fn;
    bool active;
    ~DiagnosticScopeGuard() {
      if (REX_UNLIKELY(active)) {
        fn();
      }
    }
  } diagnostic_cleanup{finish_diagnostic_capture,
                        diagnostic_storage_enabled_ || diagnostic_snapshot_enabled_};
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  const RegisterFile& regs = *register_file_;
  (void)index_buffer_info;
  uint32_t provenance_draw_id = 0;
  auto draw_fail = [&](const char* stage) {
    diagnostic_selection_stage = stage;
    auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
    REXGPU_ERROR(
        "Vulkan IssueDraw failed at {} "
        "(prim_type={}, index_count={}, source_select={}, major_mode={}, explicit_major={}, "
        "path_select={}, tess_mode={}, edram_mode={})",
        stage, uint32_t(prim_type), index_count, uint32_t(vgt_draw_initiator.source_select),
        uint32_t(vgt_draw_initiator.major_mode), uint32_t(major_mode_explicit),
        uint32_t(regs.Get<reg::VGT_OUTPUT_PATH_CNTL>().path_select),
        uint32_t(regs.Get<reg::VGT_HOS_CNTL>().tess_mode),
        uint32_t(regs.Get<reg::RB_MODECONTROL>().edram_mode));
    if (diagnostic_capture_active && !diagnostic_draw_recorded && !diagnostic_capture_error) {
      diagnostic_capture_error = stage;
    }
    if (provenance_draw_id) {
      REXGPU_INFO("GPU provenance frame {} draw {} status=failed stage={}", frame_current_,
                  provenance_draw_id, stage);
    }
    return false;
  };

  xenos::EdramMode edram_mode = regs.Get<reg::RB_MODECONTROL>().edram_mode;
  if (edram_mode == xenos::EdramMode::kCopy) {
    diagnostic_selection_stage = "copy_not_draw";
    // Special copy handling. Preserve the identity of the last actual draw for
    // the resolve diagnostic, but never count this copy command as a draw.
    return IssueCopy();
  }

  if (!diagnostic_snapshot_gate_latched_) {
    diagnostic_snapshot_gate_latched_ = true;
    std::ifstream snapshot_gate(diagnostics::kRenderTargetSnapshotEnablePath);
    if (snapshot_gate.good()) {
      const std::string content((std::istreambuf_iterator<char>(snapshot_gate)), {});
      diagnostic_snapshot_enabled_ = diagnostics::ParseRenderTargetSnapshotMode(content);
      if (diagnostic_snapshot_enabled_) {
        diagnostic_snapshot_request_ =
            diagnostics::ParseRenderTargetSnapshotRequest(content);
        diagnostic_snapshot_enabled_ = diagnostic_snapshot_request_.valid();
      }
    }
    if (diagnostic_snapshot_enabled_) {
      const char* snapshot_output = diagnostic_snapshot_request_.ben_pixel_history
                                        ? diagnostics::kBenPixelHistorySnapshotOutputPath
                                        : diagnostic_snapshot_request_.compact_ben
                                              ? diagnostics::kBenOcclusionSnapshotOutputPath
                                              : diagnostics::kRenderTargetSnapshotOutputPath;
      REXGPU_INFO(
          "Render target snapshot gate: frame={} draw={} roi={},{} {}x{} color={} depth={} "
          "after_each_draw={} storage_cap=per-request output='{}'",
          diagnostic_snapshot_request_.frame, diagnostic_snapshot_request_.draw_ordinal,
          diagnostic_snapshot_request_.x, diagnostic_snapshot_request_.y,
          diagnostic_snapshot_request_.width, diagnostic_snapshot_request_.height,
          diagnostic_snapshot_request_.color, diagnostic_snapshot_request_.depth,
          diagnostic_snapshot_request_.after_each_draw, snapshot_output);
    }
  }
  if (diagnostic_snapshot_enabled_ &&
      frame_current_ == diagnostic_snapshot_request_.frame) {
    if (diagnostic_snapshot_ordinal_frame_ != frame_current_) {
      diagnostic_snapshot_ordinal_frame_ = frame_current_;
      diagnostic_snapshot_draw_ordinal_ = 0;
      diagnostic_snapshot_head_draw_seen_ = false;
    }
    ++diagnostic_snapshot_draw_ordinal_;
  }

  diagnostics::FragmentDiagnosticDrawLog::FrameSummary completed_draw_frame;
  if (diagnostic_storage_enabled_) {
    diagnostic_log_candidate = diagnostic_storage_draw_log_.Observe(
        frame_current_, diagnostic_raw_pixel_hash, &completed_draw_frame);
    if (completed_draw_frame.available) {
      LogFragmentDiagnostic(fmt::format(
          "Fragment diagnostic draw-log summary: frame={} observed_draws={} logged_entries={} "
          "truncated={}",
          completed_draw_frame.frame, completed_draw_frame.observed_draws,
          completed_draw_frame.logged_entries, completed_draw_frame.truncated));
    }
  }
  if (diagnostic_storage_enabled_ && frame_current_ == diagnostic_storage_frame_) {
    diagnostic_storage_last_vertex_hash_ = diagnostic_raw_vertex_hash;
    diagnostic_storage_last_pixel_hash_ = diagnostic_raw_pixel_hash;
    diagnostic_storage_last_draw_ordinal_ = diagnostic_storage_draw_log_.ordinal();
    diagnostic_draw_ordinal = diagnostic_storage_last_draw_ordinal_;
  }

  if (IsGpuProvenanceFrame(frame_current_)) {
    provenance_draw_id = ++gpu_provenance_draw_index_;
  }

  bool profile_cpu = ShouldProfileFrameDraws(frame_current_);
  auto t_draw_start = profile_cpu ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
  struct DrawProfileTimer {
    FrameCpuProfilingStats* stats;
    std::chrono::steady_clock::time_point start;
    ~DrawProfileTimer() {
      if (stats) {
        stats->time_draw_total += std::chrono::steady_clock::now() - start;
      }
    }
  } draw_profile_timer{profile_cpu ? &frame_cpu_profiling_stats_ : nullptr, t_draw_start};

  bool surface_pitch_is_zero = regs.Get<reg::RB_SURFACE_INFO>().surface_pitch == 0;

  const ui::vulkan::VulkanDevice::Properties& device_properties = GetVulkanDevice()->properties();

  memexport_ranges_.clear();

  // Vertex shader analysis.
  auto vertex_shader = static_cast<VulkanShader*>(active_vertex_shader());
  if (!vertex_shader) {
    // Always need a vertex shader.
    return draw_fail("missing_vertex_shader");
  }
  pipeline_cache_->AnalyzeShaderUcode(*vertex_shader);
  bool memexport_used_vertex = vertex_shader->memexport_eM_written() != 0;
  if (memexport_used_vertex) {
    if (!device_properties.vertexPipelineStoresAndAtomics) {
      REXGPU_ERROR(
          "Vertex shader memexport draw encountered without "
          "vertexPipelineStoresAndAtomics support");
      return false;
    }
    draw_util::AddMemExportRanges(regs, *vertex_shader, memexport_ranges_);
  }

  // Pixel shader analysis.
  bool primitive_polygonal = draw_util::IsPrimitivePolygonal(regs);
  bool is_rasterization_done = draw_util::IsRasterizationPotentiallyDone(regs, primitive_polygonal);
  if (surface_pitch_is_zero && is_rasterization_done) {
    diagnostic_selection_stage = "zero_surface_pitch";
    // Doesn't actually draw.
    // Unlikely that zero would even really be legal though.
    return true;
  }
  VulkanShader* pixel_shader = nullptr;
  if (is_rasterization_done) {
    // See xenos::EdramMode for explanation why the pixel shader is only used
    // when it's kColorDepth here.
    if (edram_mode == xenos::EdramMode::kColorDepth) {
      pixel_shader = static_cast<VulkanShader*>(active_pixel_shader());
      if (pixel_shader) {
        pipeline_cache_->AnalyzeShaderUcode(*pixel_shader);
        if (!draw_util::IsPixelShaderNeededWithRasterization(*pixel_shader, regs)) {
          pixel_shader = nullptr;
        }
      }
    }
  } else {
    // Disabling pixel shader for this case is also required by the pipeline
    // cache.
    if (!memexport_used_vertex) {
      diagnostic_selection_stage = "no_rasterization_or_memexport";
      // This draw has no effect.
      return true;
    }
  }


  diagnostic_selection_stage = "predicate_evaluated";
  diagnostic_effective_pixel_hash = pixel_shader ? pixel_shader->ucode_data_hash() : 0;
  if (REX_UNLIKELY(diagnostic_storage_enabled_)) {
    const uint64_t diagnostic_target_hash =
        diagnostics::GetFragmentDiagnosticPixelHash(diagnostic_storage_shader_target_);
    if (!diagnostic_storage_first_pixel_hash_logged_ &&
        diagnostic_effective_pixel_hash == diagnostic_target_hash) {
      diagnostic_storage_first_pixel_hash_logged_ = true;
      LogFragmentDiagnostic(
          fmt::format("Fragment diagnostic first_target_PS: frame={} VS={:016X} PS={:016X} "
                      "configured_frame={} recommended_frame={}",
                      frame_current_, vertex_shader->ucode_data_hash(),
                      diagnostic_effective_pixel_hash, diagnostic_storage_frame_, frame_current_));
    }
    if (diagnostic_storage_vertex_input_dump_mode_ &&
        frame_current_ == diagnostics::kVertexInputDiagnosticFirstFrame && diagnostic_log_candidate) {
      LogFragmentDiagnostic(fmt::format(
          "Vertex input diagnostic draw: frame={} ordinal={} VS={:016X} PS={:016X} "
          "prim={} indices={} indexed={} index_base=0x{:08X} index_bytes={} index_format={} "
          "index_endian={}",
          frame_current_, diagnostic_storage_draw_log_.ordinal(), vertex_shader->ucode_data_hash(),
          diagnostic_effective_pixel_hash, uint32_t(prim_type), index_count,
          index_buffer_info && index_buffer_info->length, index_buffer_info ? index_buffer_info->guest_base : 0,
          index_buffer_info ? index_buffer_info->length : 0,
          index_buffer_info ? uint32_t(index_buffer_info->format) : 0,
          index_buffer_info ? uint32_t(index_buffer_info->endianness) : 0));
      const auto& fetch_bitmap = vertex_shader->constant_register_map().vertex_fetch_bitmap;
      for (uint32_t bitmap_index = 0; bitmap_index < rex::countof(fetch_bitmap); ++bitmap_index) {
        uint32_t bits = fetch_bitmap[bitmap_index];
        uint32_t bit_index;
        while (rex::bit_scan_forward(bits, &bit_index)) {
          bits &= ~(uint32_t(1) << bit_index);
          const uint32_t fetch_index = bitmap_index * 32 + bit_index;
          const auto fetch = regs.GetVertexFetch(fetch_index);
          LogFragmentDiagnostic(fmt::format(
              "Vertex input diagnostic fetch: frame={} draw_ordinal={} fetch={} type={} "
              "address=0x{:08X} bytes={} endian={} dword0={:08X} dword1={:08X}",
              frame_current_, diagnostic_storage_draw_log_.ordinal(), fetch_index,
              uint32_t(fetch.type), fetch.address << 2, uint64_t(fetch.size) << 2,
              uint32_t(fetch.endian), fetch.dword_0, fetch.dword_1));
        }
      }
    }
    const bool candidate_shader_match =
        (diagnostic_storage_local_candidate_output_mode_ ||
         diagnostic_storage_local_candidate_triples_mode_)
            ? (diagnostic_storage_shader_target_ == diagnostics::FragmentDiagnosticShaderTarget::kHead
                   ? ((diagnostic_effective_pixel_hash == UINT64_C(0x79A3448304FA0D22) &&
                       vertex_shader &&
                       vertex_shader->ucode_data_hash() == UINT64_C(0x4197798218650D43) &&
                       index_count == 22914 && index_buffer_info &&
                       index_buffer_info->guest_base == UINT32_C(0x1B237EC0)) ||
                      diagnostic_effective_pixel_hash == UINT64_C(0x2D4C010D3D75DEA5) ||
                      diagnostic_effective_pixel_hash == UINT64_C(0x3EB44C3A6D2D267B))
                   : diagnostic_storage_shader_target_ ==
                             diagnostics::FragmentDiagnosticShaderTarget::kProducerConsumer
                         ? (diagnostic_effective_pixel_hash ==
                                diagnostics::kFragmentDiagnosticGBufferPixelHash ||
                            diagnostic_effective_pixel_hash ==
                                diagnostics::kFragmentDiagnosticDeferredPixelHash)
                   : diagnostic_effective_pixel_hash == diagnostic_target_hash)
            : diagnostics::IsFragmentDiagnosticTarget(frame_current_, diagnostic_effective_pixel_hash,
                                                      diagnostic_storage_frame_,
                                                      diagnostic_storage_shader_target_);
    const bool expected_shader_capture =
        frame_current_ == diagnostic_storage_frame_ && candidate_shader_match;
    const bool later_local_overwrite_capture =
        diagnostic_storage_local_overwrite_census_mode_ &&
        frame_current_ == diagnostic_storage_frame_ &&
        diagnostic_storage_local_overwrite_tracking_started_;
    if ((!diagnostic_storage_vertex_input_dump_mode_ ||
         diagnostic_storage_combined_vertex_output_mode_) &&
        diagnostic_storage_buffer_ && pixel_shader &&
        (expected_shader_capture || later_local_overwrite_capture)) {
      const bool local_candidate_mode = diagnostic_storage_local_candidate_triples_mode_ ||
                                       diagnostic_storage_local_candidate_output_mode_;
      const bool local_roi_mode = local_candidate_mode || diagnostic_storage_local_overwrite_census_mode_;
      const bool multi_draw_diagnostic_mode = diagnostic_storage_census_mode_ || local_candidate_mode;
      const bool census_limit_reached =
          multi_draw_diagnostic_mode && diagnostic_storage_census_draw_count_ >=
                                            diagnostics::kFragmentDiagnosticMaxCensusDraws;
      if (census_limit_reached) {
        diagnostic_selection_stage = "census_draw_limit";
        if (!diagnostic_storage_census_limit_logged_) {
          diagnostic_storage_census_limit_logged_ = true;
          LogFragmentDiagnostic(
              fmt::format("Fragment diagnostic multi-draw capture stopped at limit: frame={} max_draws={}",
                          frame_current_, diagnostics::kFragmentDiagnosticMaxCensusDraws));
        }
      } else if (!diagnostic_storage_attempt_consumed_ || multi_draw_diagnostic_mode) {
        diagnostic_storage_attempt_consumed_ = true;
        if (multi_draw_diagnostic_mode)
          ++diagnostic_storage_census_draw_count_;
        diagnostic_vertex_hash = vertex_shader->ucode_data_hash();
        diagnostic_pixel_hash = pixel_shader->ucode_data_hash();
        diagnostic_selection_stage = "selected";
        const uint64_t prior_submission =
            submission_open_ ? GetCurrentSubmission() : GetCurrentSubmission() - 1;
        bool previous_work_completed = !submission_open_ || EndSubmission(false);
        if (previous_work_completed && prior_submission > submission_completed_) {
          CheckSubmissionFenceAndDeviceLoss(prior_submission);
          previous_work_completed = !device_lost_ && submission_completed_ >= prior_submission;
        }
        if (!previous_work_completed || device_lost_) {
          report_diagnostic_outcome("failed", 0, false, "prior_submission_failed");
        } else if (!diagnostic_storage_buffer_->Clear() ||
                   !diagnostic_storage_buffer_->SetRecordingEnabled(
                       true, !diagnostic_storage_census_mode_ || local_candidate_mode,
                       local_roi_mode, diagnostic_storage_local_candidate_output_mode_)) {
          report_diagnostic_outcome("failed", 0, false, "reset_or_enable_failed");
        } else {
          diagnostic_capture_active = true;
          if (!BeginSubmission(true)) {
            diagnostic_capture_error = "begin_submission_failed";
            return draw_fail("diagnostic_begin_submission");
          }
        }
      }
    }
  }

  bool memexport_used_pixel = pixel_shader && (pixel_shader->memexport_eM_written() != 0);
  if (memexport_used_pixel) {
    if (!device_properties.fragmentStoresAndAtomics) {
      REXGPU_ERROR(
          "Pixel shader memexport draw encountered without "
          "fragmentStoresAndAtomics support");
      if (diagnostic_capture_active) {
        diagnostic_capture_error = "fragment_stores_and_atomics_unsupported";
      }
      return false;
    }
    draw_util::AddMemExportRanges(regs, *pixel_shader, memexport_ranges_);
  }
  draw_util::Scissor vertex_diagnostic_scissor{};
  if (REX_UNLIKELY(diagnostic_storage_vertex_input_dump_mode_)) {
    draw_util::GetScissor(regs, vertex_diagnostic_scissor);
    const uint32_t ben_vertex_output_target =
        diagnostics::GetBenOcclusionVertexOutputDiagnosticTarget(
            frame_current_, vertex_diagnostic_scissor.extent[1], diagnostic_raw_vertex_hash,
            diagnostic_effective_pixel_hash, index_count,
            regs.Get<reg::RB_DEPTHCONTROL>().value);
    const bool legacy_vertex_input_target =
        !diagnostic_storage_ben_occlusion_mode_ &&
        diagnostics::IsVertexInputDiagnosticHeadDraw(
            frame_current_, diagnostic_storage_draw_log_.ordinal(),
            vertex_diagnostic_scissor.extent[1], diagnostic_raw_vertex_hash,
            diagnostic_effective_pixel_hash);
    if (diagnostic_storage_buffer_ && pixel_shader &&
        (diagnostic_storage_ben_occlusion_mode_ ? ben_vertex_output_target != 0
                                                 : legacy_vertex_input_target)) {
    if (diagnostic_storage_ben_occlusion_mode_ && ben_vertex_output_target && index_buffer_info &&
        index_buffer_info->length && index_buffer_info->length <= 16 * 1024 * 1024 && memory_) {
      const uint8_t* index_bytes =
          memory_->TranslatePhysical<const uint8_t*>(index_buffer_info->guest_base);
      if (index_bytes) {
        const std::string index_path = fmt::format(
            "/Users/Kushal/Projects/cosmicR/out/face-shoulder-eye-audit-20261023/"
            "frame-{}-session-{}-draw-{}-index.bin",
            frame_current_, diagnostic_storage_session_, diagnostic_storage_draw_log_.ordinal());
        std::ofstream index_output(index_path, std::ios::binary);
        if (index_output.good()) {
          index_output.write(reinterpret_cast<const char*>(index_bytes),
                             std::streamsize(index_buffer_info->length));
        }
        LogFragmentDiagnostic(fmt::format(
            "Ben occlusion index stream: frame={} draw={} address=0x{:08X} count={} bytes={} "
            "format={} endian={} status={}",
            frame_current_, diagnostic_storage_draw_log_.ordinal(), index_buffer_info->guest_base,
            index_count, index_buffer_info->length, uint32_t(index_buffer_info->format),
            uint32_t(index_buffer_info->endianness),
            index_output.good() ? "saved" : "save_failed"));
      }
    }
    // [DEBUG-face-stage] Preserve the exact same-draw tf0/tf2/tf14 descriptors and guest
    // resource bytes for the frame-6000 fragment capture. This runs only after the existing
    // shader/scissor selector has identified the lower-strip head draw; it does not modify
    // resource contents or guest rendering state.
    if (!diagnostic_storage_ben_occlusion_mode_ &&
        frame_current_ == diagnostics::kVertexInputDiagnosticSnapshotFrame) {
      static bool face_stage_gbuffer_fetch_dumped = false;
      static bool face_stage_head_fetch_dumped = false;
      const bool is_gbuffer_fetch_dump =
          diagnostic_raw_pixel_hash == diagnostics::kGBufferVertexDiagnosticPixelHash;
      bool& fetch_dumped = is_gbuffer_fetch_dump ? face_stage_gbuffer_fetch_dumped
                                                 : face_stage_head_fetch_dumped;
      if (!fetch_dumped && memory_ && register_file_) {
        fetch_dumped = true;
        const std::string dump_prefix = fmt::format(
            "/Users/Kushal/Projects/cosmicR/out/face-shoulder-eye-audit-20261023/"
            "frame-6000-session-{}-draw-{}",
            static_cast<unsigned long long>(diagnostic_storage_session_),
            diagnostic_storage_draw_log_.ordinal());
        std::ofstream manifest(dump_prefix + "-fetch-resources.txt");
        const std::vector<uint32_t> resource_slots =
            is_gbuffer_fetch_dump ? std::vector<uint32_t>{4u}
                                  : std::vector<uint32_t>{0u, 2u, 14u};
        for (uint32_t slot : resource_slots) {
          const xenos::xe_gpu_texture_fetch_t fetch = register_file_->GetTextureFetch(slot);
          uint32_t width_minus_1 = 0, height_minus_1 = 0, depth_minus_1 = 0;
          uint32_t base_page = 0, mip_page = 0, mip_min = 0, mip_max = 0;
          texture_util::GetSubresourcesFromFetchConstant(
              fetch, &width_minus_1, &height_minus_1, &depth_minus_1, &base_page, &mip_page,
              &mip_min, &mip_max);
          const texture_util::TextureGuestLayout guest_layout =
              texture_util::GetGuestTextureLayout(
                  fetch.dimension, fetch.pitch, width_minus_1 + 1, height_minus_1 + 1,
                  depth_minus_1 + 1, fetch.tiled, fetch.format, fetch.packed_mips,
                  base_page != 0, mip_max);
          const uint32_t base_size = guest_layout.base.level_data_extent_bytes;
          const uint32_t mip_size = guest_layout.mips_total_extent_bytes;
          if (manifest.good()) {
            manifest << "slot=" << slot << " dimension=" << uint32_t(fetch.dimension)
                     << " tiled=" << uint32_t(fetch.tiled)
                     << " format=" << uint32_t(fetch.format)
                     << " endian=" << uint32_t(fetch.endianness)
                     << " pitch=" << fetch.pitch << " width=" << width_minus_1 + 1
                     << " height=" << height_minus_1 + 1 << " depth=" << depth_minus_1 + 1
                     << " base_page=0x" << std::hex << base_page << " mip_page=0x" << mip_page
                     << std::dec << " mip_min=" << mip_min << " mip_max=" << mip_max
                     << " base_size=" << base_size << " mip_size=" << mip_size
                     << " fetch_words=" << std::hex << fetch.dword_0 << ',' << fetch.dword_1
                     << ',' << fetch.dword_2 << ',' << fetch.dword_3 << ',' << fetch.dword_4
                     << ',' << fetch.dword_5 << std::dec << '\n';
          }
          constexpr uint32_t kMaxFaceStageTextureDumpBytes = 16 * 1024 * 1024;
          auto dump_resource = [&](uint32_t page, uint32_t size, const char* suffix) {
            if (!page || !size || size > kMaxFaceStageTextureDumpBytes) {
              REXGPU_INFO(
                  "[DEBUG-face-stage] Skipping tf{} {} resource dump: page=0x{:05X} size={}",
                  slot, suffix, page, size);
              return;
            }
            const uint32_t address = page << 12;
            const uint8_t* source = memory_->TranslatePhysical<const uint8_t*>(address);
            if (!source) {
              REXGPU_INFO(
                  "[DEBUG-face-stage] Missing tf{} {} resource mapping at 0x{:08X}", slot,
                  suffix, address);
              return;
            }
            std::ofstream output(dump_prefix + fmt::format("-tf{}-{}.bin", slot, suffix),
                                 std::ios::binary);
            if (output.good()) {
              output.write(reinterpret_cast<const char*>(source), size);
              REXGPU_INFO(
                  "[DEBUG-face-stage] Captured tf{} {} resource frame={} draw={} address=0x{:08X} "
                  "bytes={}",
                  slot, suffix, frame_current_, diagnostic_storage_draw_log_.ordinal(), address,
                  size);
            }
          };
          dump_resource(base_page, base_size, "base");
          dump_resource(mip_page, mip_size, "mips");
        }
      }
    }
    const uint32_t frame_slot =
        diagnostics::GetVertexInputDiagnosticFrameSlot(frame_current_);
    if (diagnostic_storage_ben_occlusion_mode_ ? ben_vertex_output_target != 0
                                               : !diagnostic_vertex_output_frame_captured_[frame_slot]) {
      const bool legacy_target_vertex_output =
          (diagnostic_raw_vertex_hash == diagnostics::kVertexInputDiagnosticVertexHash &&
           diagnostic_effective_pixel_hash == diagnostics::kVertexInputDiagnosticPixelHash) ||
          (diagnostic_raw_vertex_hash == diagnostics::kGBufferVertexDiagnosticVertexHash &&
           diagnostic_effective_pixel_hash == diagnostics::kGBufferVertexDiagnosticPixelHash);
      const bool target_vertex_output = diagnostic_storage_ben_occlusion_mode_
                                            ? ben_vertex_output_target != 0
                                            : legacy_target_vertex_output;
      if (diagnostic_storage_combined_vertex_output_mode_) {
        if (diagnostic_capture_active && target_vertex_output) {
          diagnostic_vertex_output_frame_captured_[frame_slot] = true;
          diagnostic_vertex_output_capture_active = true;
          diagnostic_vertex_hash = diagnostic_raw_vertex_hash;
          diagnostic_pixel_hash = diagnostic_effective_pixel_hash;
          diagnostic_draw_ordinal = diagnostic_storage_draw_log_.ordinal();
          LogFragmentDiagnostic(fmt::format(
              "Combined vertex/fragment diagnostic armed: frame={} draw={} VS={:016X} PS={:016X} "
              "index_count={} guest_index_base=0x{:08X} guest_index_bytes={}",
              frame_current_, diagnostic_draw_ordinal, diagnostic_vertex_hash,
              diagnostic_pixel_hash, index_count,
              index_buffer_info ? index_buffer_info->guest_base : 0,
              index_buffer_info ? index_buffer_info->length : 0));
        }
      } else {
      const uint64_t prior_submission =
          submission_open_ ? GetCurrentSubmission() : GetCurrentSubmission() - 1;
      bool prior_work_completed = !submission_open_ || EndSubmission(false);
      if (prior_work_completed && prior_submission > submission_completed_) {
        CheckSubmissionFenceAndDeviceLoss(prior_submission);
        prior_work_completed = !device_lost_ && submission_completed_ >= prior_submission;
      }
      if (!prior_work_completed || device_lost_) {
        diagnostic_capture_error = "prior_submission_failed_before_vertex_capture";
      } else if (!diagnostic_storage_buffer_->Clear() ||
                 !diagnostic_storage_buffer_->SetRecordingEnabled(true, false, false, false)) {
        diagnostic_capture_error = "vertex_capture_buffer_reset_failed";
      } else {
        if (!diagnostic_storage_ben_occlusion_mode_) {
          diagnostic_vertex_output_frame_captured_[frame_slot] = true;
        }
        diagnostic_vertex_output_capture_active = true;
        diagnostic_capture_active = true;
        diagnostic_vertex_hash = diagnostic_raw_vertex_hash;
        diagnostic_pixel_hash = diagnostic_effective_pixel_hash;
        diagnostic_draw_ordinal = diagnostic_storage_draw_log_.ordinal();
        LogFragmentDiagnostic(fmt::format(
            "Vertex output diagnostic armed: frame={} draw={} VS={:016X} PS={:016X} "
            "base_vertex=0 first_index=0 instances=1 index_count={} guest_index_base=0x{:08X} "
            "guest_index_bytes={} mode=post_host_viewport_clip",
            frame_current_, diagnostic_draw_ordinal, diagnostic_vertex_hash,
            diagnostic_pixel_hash, index_count,
            index_buffer_info ? index_buffer_info->guest_base : 0,
            index_buffer_info ? index_buffer_info->length : 0));
      }
      }
    }
    }
  }

  const uint64_t draw_vs_hash = vertex_shader ? vertex_shader->ucode_data_hash() : 0;
  const uint64_t draw_ps_hash = pixel_shader ? pixel_shader->ucode_data_hash() : 0;
  const bool is_ultimatrix_draw =
      (draw_vs_hash == UINT64_C(0x8A449D0796A656D7) || draw_ps_hash == UINT64_C(0x6E6F06115807B602));
  const bool is_torso_draw =
      (draw_vs_hash == UINT64_C(0x046523BA1EB2717E) || draw_ps_hash == UINT64_C(0x2D4C010D3D75DEA5));

  const bool is_glow_draw =
      (draw_vs_hash == UINT64_C(0x7A0841365E75A823) || draw_ps_hash == UINT64_C(0x7ABDFEFBCADB4BF9));

  reg::RB_DEPTHCONTROL normalized_depth_control = draw_util::GetNormalizedDepthControl(regs);
  bool ultimatrix_compat_override_applied = false;

  uint32_t ps_param_gen_pos = UINT32_MAX;
  uint32_t interpolator_mask =
      pixel_shader ? (vertex_shader->writes_interpolators() &
                      pixel_shader->GetInterpolatorInputMask(regs.Get<reg::SQ_PROGRAM_CNTL>(),
                                                             regs.Get<reg::SQ_CONTEXT_MISC>(),
                                                             ps_param_gen_pos))
                   : 0;

  PrimitiveProcessor::ProcessingResult primitive_processing_result;
  SpirvShaderTranslator::Modification vertex_shader_modification;
  SpirvShaderTranslator::Modification pixel_shader_modification;
  RasterVariantDecision raster_variant_decision;
  const bool raster_variant_active = RasterVariantAuditActive(frame_current_) && pixel_shader &&
                                    IsRasterVariantTarget(pixel_shader->ucode_data_hash());
  VulkanShader::VulkanTranslation* vertex_shader_translation;
  VulkanShader::VulkanTranslation* pixel_shader_translation;
  bool memexport_writes_possible = memexport_used_vertex || memexport_used_pixel;

  // Two iterations because a submission (even the current one - in which case
  // it needs to be ended, and a new one must be started) may need to be awaited
  // in case of a sampler count overflow, and if that happens, all subsystem
  // updates done previously must be performed again because the updates done
  // before the awaiting may be referencing objects destroyed by
  // CompletedSubmissionUpdated.
  for (uint32_t i = 0; i < 2; ++i) {
    if (!BeginSubmission(true)) {
      return draw_fail("begin_submission");
    }

    // Process primitives.
    auto t_prim_start = profile_cpu ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
    if (!primitive_processor_->Process(primitive_processing_result)) {
      return draw_fail("primitive_processing");
    }
    if (profile_cpu) {
      frame_cpu_profiling_stats_.time_primitive_processing +=
          std::chrono::steady_clock::now() - t_prim_start;
    }
    if (!primitive_processing_result.host_draw_vertex_count) {
      // Nothing to draw. A diagnostic candidate selected above may still
      // produce no host draw command, so preserve the reason in its outcome.
      if (diagnostic_capture_active) {
        diagnostic_capture_error = "zero_host_draw_vertices";
      }
      return true;
    }
    if (primitive_processing_result.host_primitive_type == xenos::PrimitiveType::kTriangleFan) {
      // Vulkan uses the same fan-to-list conversion policy as D3D12.
      REXGPU_ERROR(
          "PrimitiveProcessor returned triangle fan for Vulkan draw; expected "
          "triangle list conversion for D3D12 parity");
      assert_always();
      return false;
    }
    // Tessellation and rectangle expansion variants for rasterization are
    // produced by the primitive processor and are handled by the Vulkan
    // pipeline cache.
    Shader::HostVertexShaderType host_vertex_shader_type =
        primitive_processing_result.host_vertex_shader_type;
    if (host_vertex_shader_type != Shader::HostVertexShaderType::kVertex &&
        host_vertex_shader_type != Shader::HostVertexShaderType::kPointListAsTriangleStrip &&
        host_vertex_shader_type != Shader::HostVertexShaderType::kRectangleListAsTriangleStrip &&
        !Shader::IsHostVertexShaderTypeDomain(host_vertex_shader_type)) {
      REXGPU_ERROR("Unsupported Vulkan host vertex shader type {}",
                   uint32_t(host_vertex_shader_type));
      return false;
    }

    // Shader modifications and translations form one measured phase. This
    // phase starts after primitive processing and ends before sampler work, so
    // it does not overlap the other draw-accounting buckets.
    auto t_shader_translation_start = profile_cpu ? std::chrono::steady_clock::now()
                                                  : std::chrono::steady_clock::time_point{};
    vertex_shader_modification = pipeline_cache_->GetCurrentVertexShaderModification(
        *vertex_shader, primitive_processing_result.host_vertex_shader_type, interpolator_mask,
        ps_param_gen_pos != UINT32_MAX);
    if (diagnostic_vertex_output_capture_active) {
      vertex_shader_modification.vertex.diagnostic_storage_version = 1;
    }
    pixel_shader_modification = pixel_shader ? pipeline_cache_->GetCurrentPixelShaderModification(
                                                   *pixel_shader, interpolator_mask,
                                                   ps_param_gen_pos, normalized_depth_control,
                                                   raster_variant_active ? &raster_variant_decision : nullptr)
                                             : SpirvShaderTranslator::Modification(0);
    if (diagnostic_capture_active && pixel_shader && i == 0) {
      const auto& target_texture_bindings = pixel_shader->texture_bindings();
      const auto fetch14_binding_count = std::count_if(
          target_texture_bindings.begin(), target_texture_bindings.end(),
          [](const Shader::TextureBinding& binding) { return binding.fetch_constant == 14; });
      const uint32_t diagnostic_storage_version =
          pixel_shader_modification.pixel.diagnostic_storage_version;
      LogFragmentDiagnostic(fmt::format(
          "Fragment diagnostic pipeline: frame={} draw={} PS={:016X} diagnostic_storage_version={} "
          "fetch14_bindings={} host_vertex_count={} index_count={}",
          frame_current_, diagnostic_draw_ordinal, diagnostic_pixel_hash,
          diagnostic_storage_version, fetch14_binding_count,
          primitive_processing_result.host_draw_vertex_count, index_count));
    }

    // Translate the shaders now to obtain the sampler bindings.
    vertex_shader_translation = static_cast<VulkanShader::VulkanTranslation*>(
        vertex_shader->GetOrCreateTranslation(vertex_shader_modification.value));
    pixel_shader_translation =
        pixel_shader ? static_cast<VulkanShader::VulkanTranslation*>(
                           pixel_shader->GetOrCreateTranslation(pixel_shader_modification.value))
                     : nullptr;
    if (!pipeline_cache_->EnsureShadersTranslated(vertex_shader_translation,
                                                  pixel_shader_translation)) {
      return draw_fail("shader_translation");
    }
    if (profile_cpu) {
      frame_cpu_profiling_stats_.time_shader_translation +=
          std::chrono::steady_clock::now() - t_shader_translation_start;
    }

    // Obtain the samplers. Note that the bindings don't depend on the shader
    // modification, so if on the second iteration of this loop it becomes
    // different for some reason (like a race condition with the guest in index
    // buffer processing in the primitive processor resulting in different host
    // vertex shader types), the bindings will stay the same.
    // TODO(Triang3l): Sampler caching and reuse for adjacent draws within one
    // submission.
    uint32_t samplers_overflowed_count = 0;
    auto t_sampler_prep_start = profile_cpu ? std::chrono::steady_clock::now()
                                            : std::chrono::steady_clock::time_point{};
    for (uint32_t j = 0; j < 2; ++j) {
      std::vector<std::pair<VulkanTextureCache::SamplerParameters, VkSampler>>& shader_samplers =
          j ? current_samplers_pixel_ : current_samplers_vertex_;
      if (!i) {
        shader_samplers.clear();
      }
      const VulkanShader* shader = j ? pixel_shader : vertex_shader;
      if (!shader) {
        continue;
      }
      const std::vector<VulkanShader::SamplerBinding>& shader_sampler_bindings =
          shader->GetSamplerBindingsAfterTranslation();
      if (!i) {
        shader_samplers.reserve(shader_sampler_bindings.size());
        for (const VulkanShader::SamplerBinding& shader_sampler_binding : shader_sampler_bindings) {
          const bool sampler_override_target =
              j && diagnostic_storage_enabled_ && diagnostics::IsFragmentDiagnosticTf14FilterTarget(
                                                      diagnostic_storage_tf14_filter_mode_,
                                                      frame_current_, shader->ucode_data_hash(),
                                                      shader_sampler_binding.fetch_constant,
                                                      diagnostic_storage_frame_);
          const VulkanTextureCache::SamplerParameters sampler_parameters_before =
              texture_cache_->GetSamplerParameters(shader_sampler_binding);
          const bool anisotropy_disabled =
              sampler_parameters_before.aniso_filter == xenos::AnisoFilter::kDisabled;
          const bool force_min_mag_linear =
              sampler_override_target &&
              diagnostic_storage_tf14_filter_mode_ ==
                  diagnostics::FragmentDiagnosticTf14FilterMode::kLinear &&
              anisotropy_disabled;
          VulkanTextureCache::SamplerParameters sampler_parameters =
              force_min_mag_linear
                  ? texture_cache_->GetSamplerParameters(shader_sampler_binding, true)
                  : sampler_parameters_before;
          const bool tf0_mip_override_target =
              j && diagnostic_storage_enabled_ &&
              diagnostics::IsFragmentDiagnosticTf0MipFilterTarget(
                  diagnostic_storage_tf0_mip_filter_mode_, frame_current_,
                  shader->ucode_data_hash(), shader_sampler_binding.fetch_constant,
                  diagnostic_storage_frame_);
          if (tf0_mip_override_target) {
            const bool target_linear =
                diagnostic_storage_tf0_mip_filter_mode_ ==
                diagnostics::FragmentDiagnosticTf0MipFilterMode::kLinear;
            const bool mip_filterable =
                !sampler_parameters_before.mip_base_map &&
                (sampler_parameters_before.min_linear ||
                 sampler_parameters_before.mag_linear ||
                 sampler_parameters_before.mip_linear);
            bool tf0_mip_override_applied = false;
            const char* tf0_mip_override_reason = "none";
            if (!mip_filterable) {
              tf0_mip_override_reason = "mip_filter_unsupported";
            } else {
              const uint32_t mip_before = sampler_parameters.mip_linear;
              sampler_parameters.mip_linear = target_linear ? 1 : 0;
              tf0_mip_override_applied = sampler_parameters.mip_linear != mip_before;
              if (!tf0_mip_override_applied) {
                tf0_mip_override_reason = target_linear ? "already_linear" : "already_point";
              }
            }
            LogFragmentDiagnostic(fmt::format(
                "Fragment diagnostic tf0 mip override: frame={} draw={} PS={:016X} fetch=0 "
                "mip_before={} mip_after={} applied={} reason={}",
                frame_current_, diagnostic_draw_ordinal, shader->ucode_data_hash(),
                uint32_t(sampler_parameters_before.mip_linear),
                uint32_t(sampler_parameters.mip_linear), tf0_mip_override_applied,
                tf0_mip_override_reason));
          }
          bool sampler_override_applied = false;
          const char* sampler_override_reason = "none";
          if (sampler_override_target) {
            if (diagnostic_storage_tf14_filter_mode_ ==
                diagnostics::FragmentDiagnosticTf14FilterMode::kPoint) {
              if (anisotropy_disabled) {
                sampler_parameters.min_linear = 0;
                sampler_parameters.mag_linear = 0;
                sampler_override_applied = sampler_parameters_before.min_linear ||
                                           sampler_parameters_before.mag_linear;
                if (!sampler_override_applied) {
                  sampler_override_reason = "already_point";
                }
              } else {
                sampler_override_reason = "anisotropy_enabled";
              }
            } else if (diagnostic_storage_tf14_filter_mode_ ==
                       diagnostics::FragmentDiagnosticTf14FilterMode::kLinear) {
              sampler_override_applied =
                  force_min_mag_linear &&
                  (sampler_parameters.min_linear != sampler_parameters_before.min_linear ||
                   sampler_parameters.mag_linear != sampler_parameters_before.mag_linear);
              if (!anisotropy_disabled) {
                sampler_override_reason = "anisotropy_enabled";
              } else if (!sampler_override_applied && !sampler_parameters.min_linear &&
                         !sampler_parameters.mag_linear) {
                sampler_override_reason = "linear_filter_unsupported";
              } else if (!sampler_override_applied) {
                sampler_override_reason = "already_linear";
              }
            }
            LogFragmentDiagnostic(fmt::format(
                "Fragment diagnostic sampler override: frame={} draw={} PS={:016X} fetch=14 "
                "min_before={} mag_before={} min_after={} mag_after={} mip_linear={} "
                "aniso_filter={} applied={} reason={}",
                frame_current_, diagnostic_draw_ordinal, shader->ucode_data_hash(),
                sampler_parameters_before.min_linear ? "linear" : "point",
                sampler_parameters_before.mag_linear ? "linear" : "point",
                sampler_parameters.min_linear ? "linear" : "point",
                sampler_parameters.mag_linear ? "linear" : "point",
                uint32_t(sampler_parameters.mip_linear),
                uint32_t(sampler_parameters.aniso_filter), sampler_override_applied,
                sampler_override_reason));
          }
          shader_samplers.emplace_back(sampler_parameters, VK_NULL_HANDLE);
        }
      }
      for (std::pair<VulkanTextureCache::SamplerParameters, VkSampler>& shader_sampler_pair :
           shader_samplers) {
        // UseSampler calls are needed even on the second iteration in case the
        // submission was broken (and thus the last usage submission indices for
        // the used samplers need to be updated) due to an overflow within one
        // submission. Though sampler overflow is a very rare situation overall.
        bool sampler_overflowed;
        VkSampler shader_sampler =
            texture_cache_->UseSampler(shader_sampler_pair.first, sampler_overflowed);
        shader_sampler_pair.second = shader_sampler;
        if (shader_sampler == VK_NULL_HANDLE) {
          if (!sampler_overflowed || i) {
            // If !sampler_overflowed, just failed to create a sampler for some
            // reason.
            // If i == 1, an overflow has happened twice, can't recover from it
            // anymore (would enter an infinite loop otherwise if the number of
            // attempts was not limited to 2). Possibly too many unique samplers
            // in one draw, or failed to await submission completion.
            return draw_fail("sampler_acquisition");
          }
          ++samplers_overflowed_count;
        }
      }
    }
    if (profile_cpu) {
      frame_cpu_profiling_stats_.time_sampler_preparation +=
          std::chrono::steady_clock::now() - t_sampler_prep_start;
    }
    if (!samplers_overflowed_count) {
      break;
    }
    assert_zero(i);
    // Free space for as many samplers as how many haven't been allocated
    // successfully - obtain the submission index that needs to be awaited to
    // reuse `samplers_overflowed_count` slots. This must be done after all the
    // UseSampler calls, not inside the loop calling UseSampler, because earlier
    // UseSampler calls may "mark for deletion" some samplers that later
    // UseSampler calls in the loop may actually demand.
    uint64_t sampler_overflow_await_submission =
        texture_cache_->GetSubmissionToAwaitOnSamplerOverflow(samplers_overflowed_count);
    assert_true(sampler_overflow_await_submission <= GetCurrentSubmission());
    CheckSubmissionFenceAndDeviceLoss(sampler_overflow_await_submission);
  }

  uint32_t normalized_color_mask =
      pixel_shader ? draw_util::GetNormalizedColorMask(regs, pixel_shader->writes_color_targets())
                   : 0;

  // Update the textures before most other work in the submission because
  // samplers depend on this (and in case of sampler overflow in a submission,
  // submissions must be split) - may perform dispatches and copying.
  uint32_t used_texture_mask =
      vertex_shader->GetUsedTextureMaskAfterTranslation() |
      (pixel_shader != nullptr ? pixel_shader->GetUsedTextureMaskAfterTranslation() : 0);
  auto t_texture_request_start = profile_cpu ? std::chrono::steady_clock::now()
                                             : std::chrono::steady_clock::time_point{};
  texture_cache_->RequestTextures(used_texture_mask);
  if (profile_cpu) {
    frame_cpu_profiling_stats_.time_texture_request +=
        std::chrono::steady_clock::now() - t_texture_request_start;
  }

  const VulkanPipelineCache::PipelineLayoutProvider* pipeline_layout_provider;
  // Set up the render targets - this may perform dispatches and draws.
  auto t_rts_start = profile_cpu ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
  if (!render_target_cache_->Update(is_rasterization_done, normalized_depth_control,
                                    normalized_color_mask, *vertex_shader)) {
    return draw_fail("render_target_update");
  }
  if (REXCVAR_GET(gpu_frame_audit_enable) &&
      REXCVAR_QUERY(bool, render_target_retain_color_on_depth_only) &&
      render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets &&
      is_rasterization_done &&
      (normalized_depth_control.z_enable || normalized_depth_control.stencil_enable) &&
      !normalized_color_mask) {
    const uint32_t bound = render_target_cache_->last_update_render_pass_key().depth_and_color_used;
    if ((bound & 1) && (bound >> 1)) {
      ++frame_cpu_profiling_stats_.retained_depth_only_draws;
    }
  }
  if (profile_cpu) {
    frame_cpu_profiling_stats_.time_render_target_update +=
        std::chrono::steady_clock::now() - t_rts_start;
  }

  if (provenance_draw_id &&
      (provenance_draw_id <= 10 || provenance_draw_id % 100 == 0 || provenance_draw_id == 1000)) {
    std::array<uint32_t, 1 + xenos::kMaxColorRenderTargets> depth_and_color_formats{};
    static_assert(std::tuple_size_v<decltype(depth_and_color_formats)> ==
                  1 + xenos::kMaxColorRenderTargets);
    bool host_render_targets =
        render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets;
    uint32_t bound_targets =
        host_render_targets
            ? render_target_cache_->GetLastUpdateBoundRenderTargets(depth_and_color_formats.data())
            : render_target_cache_->last_update_render_pass_key().depth_and_color_used;
    auto render_pass_key = render_target_cache_->last_update_render_pass_key();
    REXGPU_INFO(
        "GPU provenance frame {} draw {} targets path={} mask=0x{:X} depth_format=0x{:X} "
        "render_pass_key=0x{:X} framebuffer={:p} edram_buffer={:p}",
        frame_current_, provenance_draw_id, host_render_targets ? "host" : "fsi", bound_targets,
        depth_and_color_formats[0], render_pass_key.key,
        static_cast<const void*>(render_target_cache_->last_update_framebuffer()),
        static_cast<void*>(render_target_cache_->edram_buffer()));
    if (host_render_targets) {
      for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
        REXGPU_INFO("GPU provenance frame {} draw {} color{} format=0x{:X}", frame_current_,
                    provenance_draw_id, i, depth_and_color_formats[1 + i]);
      }
    }
    if (host_render_targets) {
      VkRenderingAttachmentInfo colors[xenos::kMaxColorRenderTargets]{};
      VkRenderingAttachmentInfo depth{};
      VkRenderingAttachmentInfo stencil{};
      uint32_t color_count = 0;
      render_target_cache_->GetLastUpdateRenderingAttachments(colors, &color_count, &depth,
                                                              &stencil);
      for (uint32_t i = 0; i < color_count; ++i) {
        if (colors[i].imageView != VK_NULL_HANDLE) {
          REXGPU_INFO("GPU provenance frame {} draw {} color{} image_view={:p}", frame_current_,
                      provenance_draw_id, i, static_cast<void*>(colors[i].imageView));
        }
      }
      if (depth.imageView != VK_NULL_HANDLE) {
        REXGPU_INFO("GPU provenance frame {} draw {} depth image_view={:p}", frame_current_,
                    provenance_draw_id, static_cast<void*>(depth.imageView));
      }
    }
  }

  // Create the pipeline (for this, need the render pass from the render target
  // cache), translating the shaders - doing this now to obtain the used
  // textures.
  auto t_pipe_start = profile_cpu ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
  VkPipeline pipeline;
  void* pipeline_handle = nullptr;
  if (!pipeline_cache_->ConfigurePipeline(vertex_shader_translation, pixel_shader_translation,
                                          primitive_processing_result, normalized_depth_control,
                                          normalized_color_mask,
                                          render_target_cache_->last_update_render_pass_key(),
                                          pipeline, pipeline_layout_provider, &pipeline_handle)) {
    return draw_fail("configure_pipeline");
  }
  bool pipeline_is_placeholder = false;
  // Reload the current handle state to observe async hot-swap completions that
  // may happen between pipeline configuration and binding.
  pipeline_cache_->GetPipelineAndLayoutByHandle(pipeline_handle, pipeline, pipeline_layout_provider,
                                                &pipeline_is_placeholder);
  if (profile_cpu) {
    frame_cpu_profiling_stats_.time_pipeline_configuration +=
        std::chrono::steady_clock::now() - t_pipe_start;
  }

  if (REXCVAR_GET(async_shader_compilation) && pipeline_is_placeholder) {
    if (diagnostic_capture_active) {
      diagnostic_capture_error = "async_placeholder_not_executed";
    }
    frame_used_async_placeholder_pipeline_ = true;
    if (provenance_draw_id) {
      REXGPU_INFO("GPU provenance frame {} draw {} status=async_placeholder_not_executed",
                  frame_current_, provenance_draw_id);
    }
    return true;
  }
  if (pipeline == VK_NULL_HANDLE || pipeline_layout_provider == nullptr) {
    return draw_fail("pipeline_lookup");
  }
  auto t_dyn_start = profile_cpu ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
  if (current_guest_graphics_pipeline_ != pipeline) {
    deferred_command_buffer_.CmdVkBindPipeline(VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
    current_guest_graphics_pipeline_ = pipeline;
    current_external_graphics_pipeline_ = VK_NULL_HANDLE;
  }

  // Update the graphics pipeline, and if the new graphics pipeline has a
  // different layout, invalidate incompatible descriptor sets before updating
  // current_guest_graphics_pipeline_layout_.
  auto pipeline_layout = static_cast<const PipelineLayout*>(pipeline_layout_provider);
  if (current_guest_graphics_pipeline_layout_ != pipeline_layout) {
    if (current_guest_graphics_pipeline_layout_) {
      // Keep descriptor set layouts for which the new pipeline layout is
      // compatible with the previous one (pipeline layouts are compatible for
      // set N if set layouts 0 through N are compatible).
      uint32_t descriptor_sets_kept = uint32_t(SpirvShaderTranslator::kDescriptorSetCount);
      if (current_guest_graphics_pipeline_layout_->descriptor_set_layout_textures_vertex_ref() !=
          pipeline_layout->descriptor_set_layout_textures_vertex_ref()) {
        descriptor_sets_kept = std::min(
            descriptor_sets_kept, uint32_t(SpirvShaderTranslator::kDescriptorSetTexturesVertex));
        bound_textures_vertex_count_ = UINT32_MAX;
        bound_samplers_vertex_count_ = UINT32_MAX;
        current_graphics_descriptor_set_values_up_to_date_ &=
            ~(UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex);
      }
      if (current_guest_graphics_pipeline_layout_->descriptor_set_layout_textures_pixel_ref() !=
          pipeline_layout->descriptor_set_layout_textures_pixel_ref()) {
        descriptor_sets_kept = std::min(
            descriptor_sets_kept, uint32_t(SpirvShaderTranslator::kDescriptorSetTexturesPixel));
        bound_textures_pixel_count_ = UINT32_MAX;
        bound_samplers_pixel_count_ = UINT32_MAX;
        current_graphics_descriptor_set_values_up_to_date_ &=
            ~(UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel);
      }
      current_graphics_descriptor_sets_bound_up_to_date_ &=
          (UINT32_C(1) << descriptor_sets_kept) - 1;
    } else {
      // No or unknown pipeline layout previously bound - all bindings are in an
      // indeterminate state.
      current_graphics_descriptor_sets_bound_up_to_date_ = 0;
      bound_textures_vertex_count_ = UINT32_MAX;
      bound_samplers_vertex_count_ = UINT32_MAX;
      bound_textures_pixel_count_ = UINT32_MAX;
      bound_samplers_pixel_count_ = UINT32_MAX;
      current_graphics_descriptor_set_values_up_to_date_ &=
          ~((UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex) |
            (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel));
    }
    current_guest_graphics_pipeline_layout_ = pipeline_layout;
  }

  bool host_render_targets_used =
      render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets;
  uint32_t draw_resolution_scale_x = texture_cache_->draw_resolution_scale_x();
  uint32_t draw_resolution_scale_y = texture_cache_->draw_resolution_scale_y();

  // Get dynamic rasterizer state.
  draw_util::ViewportInfo viewport_info;
  // Just handling maxViewportDimensions is enough - viewportBoundsRange[1] must
  // be at least 2 * max(maxViewportDimensions[0...1]) - 1, and
  // maxViewportDimensions must be greater than or equal to the size of the
  // largest possible framebuffer attachment (if the viewport has positive
  // offset and is between maxViewportDimensions and viewportBoundsRange[1],
  // GetHostViewportInfo will adjust ndc_scale/ndc_offset to clamp it, and the
  // clamped range will be outside the largest possible framebuffer anyway.
  // FIXME(Triang3l): Possibly handle maxViewportDimensions and
  // viewportBoundsRange separately because when using fragment shader
  // interlocks, framebuffers are not used, while the range may be wider than
  // dimensions? Though viewport bigger than 4096 - the smallest possible
  // maximum dimension (which is below the 8192 texture size limit on the Xbox
  // 360) - and with offset, is probably a situation that never happens in real
  // life. Or even disregard the viewport bounds range in the fragment shader
  // interlocks case completely - apply the viewport and the scissor offset
  // directly to pixel address and to things like ps_param_gen.
  draw_util::GetHostViewportInfo(
      regs, draw_resolution_scale_x, draw_resolution_scale_y, false,
      device_properties.maxViewportDimensions[0], device_properties.maxViewportDimensions[1], true,
      normalized_depth_control,
      host_render_targets_used && render_target_cache_->depth_float24_convert_in_pixel_shader(),
      host_render_targets_used, pixel_shader && pixel_shader->writes_depth(), viewport_info);

  if (REX_UNLIKELY(frame_current_ == 6000)) {
    const auto rpk = render_target_cache_->last_update_render_pass_key();
    const bool depth_attachment_bound = (rpk.depth_and_color_used & 1) != 0;
    const reg::RB_DEPTHCONTROL raw_dc = regs.Get<reg::RB_DEPTHCONTROL>();

    bool vk_depth_test_enable = false;
    bool vk_depth_write_enable = false;
    uint32_t vk_depth_compare_op = 0;
    if (depth_attachment_bound) {
      if (normalized_depth_control.z_enable) {
        vk_depth_write_enable = (normalized_depth_control.z_write_enable != 0);
        vk_depth_compare_op = uint32_t(normalized_depth_control.zfunc);
      } else {
        vk_depth_compare_op = uint32_t(xenos::CompareFunction::kAlways);
      }
      if (vk_depth_write_enable || vk_depth_compare_op != uint32_t(xenos::CompareFunction::kAlways)) {
        vk_depth_test_enable = true;
      }
    }

    auto get_tf_str = [&](uint32_t slot) -> std::string {
      const auto tf = regs.GetTextureFetch(slot);
      if (tf.type == xenos::FetchConstantType::kTexture) {
        return fmt::format("tf{}=[0x{:08X} fmt={} {}x{}]", slot,
                           uint32_t(tf.base_address << 12), uint32_t(tf.format),
                           uint32_t(tf.size_2d.width + 1), uint32_t(tf.size_2d.height + 1));
      }
      return fmt::format("tf{}=none", slot);
    };

    LogFragmentDiagnostic(fmt::format(
        "DRAW_CENSUS: frame={} draw={} VS={:016X} PS={:016X} idx={} "
        "raw_dc=0x{:08X}(z_en={} z_wr={} zfunc={}) "
        "norm_dc=0x{:08X}(z_en={} z_wr={} zfunc={}) "
        "depth_bound={} vk_zTest={} vk_zWr={} vk_zOp={} "
        "blend=0x{:08X} Z=[{:.4f}, {:.4f}] {} {} {} {}",
        frame_current_, diagnostic_draw_ordinal,
        draw_vs_hash, draw_ps_hash, index_count,
        raw_dc.value, uint32_t(raw_dc.z_enable), uint32_t(raw_dc.z_write_enable), uint32_t(raw_dc.zfunc),
        normalized_depth_control.value, uint32_t(normalized_depth_control.z_enable),
        uint32_t(normalized_depth_control.z_write_enable), uint32_t(normalized_depth_control.zfunc),
        depth_attachment_bound, vk_depth_test_enable, vk_depth_write_enable, vk_depth_compare_op,
        regs[XE_GPU_REG_RB_BLENDCONTROL0],
        viewport_info.z_min, viewport_info.z_max,
        get_tf_str(0), get_tf_str(1), get_tf_str(2), get_tf_str(3)));
  }

  if (REX_UNLIKELY(frame_current_ == 6000 && (is_ultimatrix_draw || is_torso_draw || is_glow_draw))) {
    const auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
    const float po_scale = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE);
    const float po_offset = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET);
    const auto rpk = render_target_cache_->last_update_render_pass_key();
    const bool depth_attachment_bound = (rpk.depth_and_color_used & 1) != 0;
    const auto depth_info = regs.Get<reg::RB_DEPTH_INFO>();
    const auto* framebuffer = render_target_cache_->last_update_framebuffer();
    const auto clip_control = regs.Get<reg::PA_CL_CLIP_CNTL>();
    const auto color_control = regs.Get<reg::RB_COLORCONTROL>();
    const auto pixel_modification =
        SpirvShaderTranslator::Modification(pixel_shader_translation->modification());
    const reg::RB_DEPTHCONTROL raw_dc = regs.Get<reg::RB_DEPTHCONTROL>();
    const auto tf0 = regs.GetTextureFetch(0);

    LogFragmentDiagnostic(fmt::format(
        "BEN_DEPTH_AUDIT: frame={} draw_idx={} role={} VS={:016X} PS={:016X} idx={} "
        "raw_dc=0x{:08X}(z_en={} z_wr={} zfunc={}) "
        "norm_dc=0x{:08X}(z_en={} z_wr={} zfunc={}) override={} "
        "depth_bound={} depth_guest_base=0x{:08X} depth_fmt={} "
        "render_pass_key=0x{:X} framebuffer={:p} ps_writes_depth={} ps_kills={} "
        "depth_mode={} clip_disable={} viewport={}x{} z=[{:.4f},{:.4f}] "
        "color_control=0x{:08X} alpha_test={} alpha_to_mask={} blend=0x{:08X} "
        "po_en={} po_scale={:.4f} po_offset={:.4f} "
        "pipeline_discard={} vs_export_mode={} output_path={} "
        "tf0=[0x{:08X} fmt={}]",
        frame_current_, gpu_provenance_draw_index_,
        is_glow_draw ? "GLOW" : (is_ultimatrix_draw ? "ULTIMATRIX" : "TORSO"),
        draw_vs_hash, draw_ps_hash, index_count,
        raw_dc.value, uint32_t(raw_dc.z_enable), uint32_t(raw_dc.z_write_enable), uint32_t(raw_dc.zfunc),
        normalized_depth_control.value, uint32_t(normalized_depth_control.z_enable),
        uint32_t(normalized_depth_control.z_write_enable), uint32_t(normalized_depth_control.zfunc),
        ultimatrix_compat_override_applied,
        depth_attachment_bound, uint32_t(depth_info.depth_base << 12),
        uint32_t(depth_info.depth_format), rpk.key,
        framebuffer ? static_cast<void*>(framebuffer->framebuffer) : nullptr,
        pixel_shader && pixel_shader->writes_depth(),
        pixel_shader && pixel_shader->kills_pixels(),
        uint32_t(pixel_modification.pixel.depth_stencil_mode),
        uint32_t(clip_control.clip_disable), viewport_info.xy_extent[0],
        viewport_info.xy_extent[1], viewport_info.z_min, viewport_info.z_max,
        color_control.value, uint32_t(color_control.alpha_test_enable),
        uint32_t(color_control.alpha_to_mask_enable),
        regs[XE_GPU_REG_RB_BLENDCONTROL0],
        uint32_t(pa_su_sc_mode_cntl.poly_offset_front_enable), po_scale, po_offset,
        !is_rasterization_done,
        uint32_t(regs.Get<reg::SQ_PROGRAM_CNTL>().vs_export_mode),
        uint32_t(regs.Get<reg::VGT_OUTPUT_PATH_CNTL>().path_select),
        uint32_t(tf0.base_address << 12), uint32_t(tf0.format)));
    render_target_cache_->LogLastUpdateDepthAttachmentState(frame_current_,
                                                            gpu_provenance_draw_index_);

    // Sample vertex positions and compute transformed NDC / screen coordinates
    auto get_vs_const = [&](uint32_t idx) -> std::array<float, 4> {
      std::array<float, 4> result{};
      for (int c = 0; c < 4; ++c) {
        uint32_t bits = regs[XE_GPU_REG_SHADER_CONSTANT_000_X + idx * 4 + c];
        std::memcpy(&result[c], &bits, sizeof(float));
      }
      return result;
    };
    const auto c8 = get_vs_const(8);
    const auto c9 = get_vs_const(9);
    const auto c10 = get_vs_const(10);
    const auto c11 = get_vs_const(11);

    xenos::xe_gpu_vertex_fetch_t vfetch{};
    int found_fi = -1;
    if (regs.GetVertexFetch(95).type == xenos::FetchConstantType::kVertex && regs.GetVertexFetch(95).address != 0) {
      vfetch = regs.GetVertexFetch(95);
      found_fi = 95;
    } else {
      for (int fi = 95; fi >= 0; --fi) {
        auto candidate = regs.GetVertexFetch(fi);
        if (candidate.type == xenos::FetchConstantType::kVertex && candidate.address != 0) {
          vfetch = candidate;
          found_fi = fi;
          break;
        }
      }
    }

    uint32_t stride_dwords = 0;
    if (is_torso_draw) stride_dwords = 11;
    else if (is_ultimatrix_draw) stride_dwords = 5;
    else if (is_glow_draw) stride_dwords = 6;

    if (vfetch.type == xenos::FetchConstantType::kVertex && vfetch.address != 0 && stride_dwords > 0) {
      const uint32_t physical_address = vfetch.address << 2;
      const uint32_t byte_count = vfetch.size << 2;
      const uint32_t* raw_words = memory_->TranslatePhysical<const uint32_t*>(physical_address);
      if (raw_words && byte_count >= stride_dwords * 4) {
        const uint32_t max_vert_in_buffer = byte_count / (stride_dwords * 4);
        float min_sx = 9999.0f, max_sx = -9999.0f;
        float min_sy = 9999.0f, max_sy = -9999.0f;
        float min_z = 9999.0f, max_z = -9999.0f;
        float roi_min_z = 9999.0f, roi_max_z = -9999.0f;
        uint32_t roi_count = 0;
        uint32_t tested_count = 0;

        auto test_vertex = [&](uint32_t vi) {
          if (vi >= max_vert_in_buffer) return;
          tested_count++;
          uint32_t bx = xenos::GpuSwap(raw_words[vi * stride_dwords + 0], vfetch.endian);
          uint32_t by = xenos::GpuSwap(raw_words[vi * stride_dwords + 1], vfetch.endian);
          uint32_t bz = xenos::GpuSwap(raw_words[vi * stride_dwords + 2], vfetch.endian);
          float x, y, z;
          std::memcpy(&x, &bx, sizeof(float));
          std::memcpy(&y, &by, sizeof(float));
          std::memcpy(&z, &bz, sizeof(float));

          float cx = x * c8[0] + y * c9[0] + z * c10[0] + c11[0];
          float cy = x * c8[1] + y * c9[1] + z * c10[1] + c11[1];
          float cz = x * c8[2] + y * c9[2] + z * c10[2] + c11[2];
          float cw = x * c8[3] + y * c9[3] + z * c10[3] + c11[3];
          if (cw > 0.0001f) {
            float sx = (cx / cw * 0.5f + 0.5f) * 1280.0f;
            float sy = (1.0f - (cy / cw * 0.5f + 0.5f)) * 720.0f;
            float ndc_z = cz / cw;
            min_sx = std::min(min_sx, sx); max_sx = std::max(max_sx, sx);
            min_sy = std::min(min_sy, sy); max_sy = std::max(max_sy, sy);
            min_z = std::min(min_z, ndc_z); max_z = std::max(max_z, ndc_z);
            if (sx >= 550.0f && sx <= 750.0f && sy >= 400.0f && sy <= 550.0f) {
              roi_min_z = std::min(roi_min_z, ndc_z);
              roi_max_z = std::max(roi_max_z, ndc_z);
              roi_count++;
            }
          }
        };

        if (index_buffer_info && index_buffer_info->length && index_count > 0) {
          const uint8_t* index_data = memory_->TranslatePhysical<const uint8_t*>(index_buffer_info->guest_base);
          if (index_data) {
            for (uint32_t ii = 0; ii < index_count; ++ii) {
              uint32_t vi = 0;
              if (index_buffer_info->format == xenos::IndexFormat::kInt16) {
                if (ii * 2 + 1 < index_buffer_info->length) {
                  uint16_t raw_idx = reinterpret_cast<const uint16_t*>(index_data)[ii];
                  vi = xenos::GpuSwap(raw_idx, index_buffer_info->endianness);
                }
              } else {
                if (ii * 4 + 3 < index_buffer_info->length) {
                  uint32_t raw_idx = reinterpret_cast<const uint32_t*>(index_data)[ii];
                  vi = xenos::GpuSwap(raw_idx, index_buffer_info->endianness);
                }
              }
              test_vertex(vi);
            }
          }
        } else {
          uint32_t total_verts = std::min(max_vert_in_buffer, 1500u);
          for (uint32_t vi = 0; vi < total_verts; ++vi) {
            test_vertex(vi);
          }
        }

        LogFragmentDiagnostic(fmt::format(
            "BEN_NDC_DEPTH: frame={} draw_idx={} role={} VS={:016X} verts={} "
            "sx=[{:.1f},{:.1f}] sy=[{:.1f},{:.1f}] all_z=[{:.5f},{:.5f}] roi_cnt={} roi_z=[{:.5f},{:.5f}] fi={} "
            "c8=[{:.4f},{:.4f},{:.4f},{:.4f}] c11=[{:.4f},{:.4f},{:.4f},{:.4f}]",
            frame_current_, gpu_provenance_draw_index_,
            is_glow_draw ? "GLOW" : (is_ultimatrix_draw ? "ULTIMATRIX" : "TORSO"),
            draw_vs_hash, tested_count,
            min_sx, max_sx, min_sy, max_sy, min_z, max_z,
            roi_count, roi_min_z, roi_max_z, found_fi,
            c8[0], c8[1], c8[2], c8[3], c11[0], c11[1], c11[2], c11[3]));
      }
    }
  }

  if (REX_UNLIKELY(diagnostic_storage_enabled_)) {
    const bool is_ben_mesh =
        (draw_ps_hash == UINT64_C(0x79A3448304FA0D22) ||
         draw_ps_hash == UINT64_C(0x6E6F06115807B602) ||
         draw_ps_hash == UINT64_C(0x2D4C010D3D75DEA5) ||
         draw_ps_hash == UINT64_C(0x3EB44C3A6D2D267B));
    if (is_ben_mesh) {
      s_current_ben_delta.char_draw_count++;
      const auto tf0_fetch = regs.GetTextureFetch(0);
      const auto tf2_fetch = regs.GetTextureFetch(2);
      s_current_ben_delta.texture_hash ^= (uint64_t(tf0_fetch.base_address) << 32) ^ uint64_t(tf2_fetch.base_address);
      s_current_ben_delta.blend_hash ^= regs[XE_GPU_REG_RB_BLENDCONTROL0];
      const auto c102_val = regs[XE_GPU_REG_SHADER_CONSTANT_256_X + 102 * 4];
      const auto c41_val = regs[XE_GPU_REG_SHADER_CONSTANT_256_X + 41 * 4];
      s_current_ben_delta.constant_hash ^= (uint64_t(c102_val) << 32) ^ uint64_t(c41_val);

      if (draw_ps_hash == UINT64_C(0x6E6F06115807B602)) {
        if (index_count == 24) s_current_ben_delta.submesh_counts[0]++;
        else if (index_count == 48) s_current_ben_delta.submesh_counts[1]++;
        else if (index_count == 108) s_current_ben_delta.submesh_counts[2]++;
        else if (index_count == 183) s_current_ben_delta.submesh_counts[3]++;
      } else if (draw_ps_hash == UINT64_C(0x2D4C010D3D75DEA5)) {
        if (index_count == 432) s_current_ben_delta.submesh_counts[4]++;
        else if (index_count == 576) s_current_ben_delta.submesh_counts[5]++;
        else if (index_count == 1344) s_current_ben_delta.submesh_counts[6]++;
      } else if (draw_ps_hash == UINT64_C(0x79A3448304FA0D22)) {
        s_current_ben_delta.submesh_counts[7]++;
      }
    }
  }

  if (REX_UNLIKELY(frame_current_ >= 5312 && frame_current_ <= 5322 && pixel_shader && index_count > 10)) {
    const auto tf0_dbg = regs.GetTextureFetch(0);
    LogFragmentDiagnostic(fmt::format(
        "BEN_FLICKER_TRACE: frame={} draw={} VS={:016X} PS={:016X} idx={} raw_dc=0x{:08X} tf0=[0x{:08X} fmt={}]",
        frame_current_, gpu_provenance_draw_index_, draw_vs_hash, draw_ps_hash, index_count,
        regs.Get<reg::RB_DEPTHCONTROL>().value, uint32_t(tf0_dbg.base_address << 12), uint32_t(tf0_dbg.format)));
  }

  const bool comprehensive_candidate_state =
      (diagnostic_storage_overwrite_census_mode_ ||
       diagnostic_storage_local_overwrite_census_mode_ ||
       diagnostic_storage_local_candidate_output_mode_ ||
       diagnostic_storage_vertex_input_dump_mode_) &&
      diagnostic_log_candidate && frame_current_ == diagnostic_storage_frame_;
  if (REX_UNLIKELY((diagnostic_capture_active &&
       (diagnostic_storage_census_mode_ || diagnostic_storage_local_candidate_triples_mode_ ||
        diagnostic_storage_local_candidate_output_mode_ ||
        diagnostic_storage_vertex_input_dump_mode_)) ||
      comprehensive_candidate_state)) {
    draw_util::Scissor scissor_info{};
    draw_util::GetScissor(regs, scissor_info);
    const auto* framebuffer =
        host_render_targets_used ? render_target_cache_->last_update_framebuffer() : nullptr;
    const uint32_t framebuffer_width = framebuffer ? framebuffer->host_extent.width : 0;
    const uint32_t framebuffer_height = framebuffer ? framebuffer->host_extent.height : 0;
    const uint32_t surface_pitch_tiles = regs.Get<reg::RB_SURFACE_INFO>().surface_pitch;
    const uint32_t color0 = regs[reg::RB_COLOR_INFO::rt_register_indices[0]];
    const uint32_t color1 = regs[reg::RB_COLOR_INFO::rt_register_indices[1]];
    const uint32_t color2 = regs[reg::RB_COLOR_INFO::rt_register_indices[2]];
    const uint32_t color3 = regs[reg::RB_COLOR_INFO::rt_register_indices[3]];
    LogFragmentDiagnostic(fmt::format(
        "Fragment diagnostic draw-state: frame={} draw={} source={} VS={:016X} PS={:016X} "
        "viewport_offset={},{} viewport_extent={},{} scissor_offset={},{} scissor_extent={},{} "
        "target_path={} framebuffer_extent={},{} surface_pitch_tiles={} scale={},{} "
        "color_info={:08X},{:08X},{:08X},{:08X} depth_info={:08X} depth_control={:08X} "
        "blend_control={:08X} color_control={:08X} mode_control={:08X} surface_info={:08X}",
        frame_current_, diagnostic_draw_ordinal,
        comprehensive_candidate_state ? "candidate" : "selected", diagnostic_raw_vertex_hash,
        diagnostic_raw_pixel_hash, viewport_info.xy_offset[0], viewport_info.xy_offset[1],
        viewport_info.xy_extent[0], viewport_info.xy_extent[1], scissor_info.offset[0],
        scissor_info.offset[1], scissor_info.extent[0], scissor_info.extent[1],
        host_render_targets_used ? "host" : "edram_interlock", framebuffer_width,
        framebuffer_height, surface_pitch_tiles, draw_resolution_scale_x, draw_resolution_scale_y,
        color0, color1, color2, color3, regs[XE_GPU_REG_RB_DEPTH_INFO],
        regs[XE_GPU_REG_RB_DEPTHCONTROL], regs[XE_GPU_REG_RB_BLENDCONTROL0],
        regs[XE_GPU_REG_RB_COLORCONTROL], regs[XE_GPU_REG_RB_MODECONTROL],
        regs[XE_GPU_REG_RB_SURFACE_INFO]));
    auto get_ps_const = [&](uint32_t idx) -> std::array<float, 4> {
      std::array<float, 4> result{};
      for (int c = 0; c < 4; ++c) {
        uint32_t bits = regs[XE_GPU_REG_SHADER_CONSTANT_256_X + idx * 4 + c];
        std::memcpy(&result[c], &bits, sizeof(float));
      }
      return result;
    };
    const auto c102 = get_ps_const(102);
    const auto c41 = get_ps_const(41);
    const auto c101 = get_ps_const(101);
    const auto c59 = get_ps_const(59);
    LogFragmentDiagnostic(fmt::format(
        "Fragment diagnostic constants: frame={} draw={} PS={:016X} "
        "c102={:.6f},{:.6f},{:.6f},{:.6f} c41={:.6f},{:.6f},{:.6f},{:.6f} "
        "c101={:.6f},{:.6f},{:.6f},{:.6f} c59={:.6f},{:.6f},{:.6f},{:.6f}",
        frame_current_, diagnostic_draw_ordinal, diagnostic_raw_pixel_hash,
        c102[0], c102[1], c102[2], c102[3],
        c41[0], c41[1], c41[2], c41[3],
        c101[0], c101[1], c101[2], c101[3],
        c59[0], c59[1], c59[2], c59[3]));
    if (diagnostic_raw_pixel_hash == UINT64_C(0xED58716070696DA4) ||
        diagnostic_raw_pixel_hash == UINT64_C(0x3CC2A52F0A5CB063)) {
      auto get_ps_const_bits = [&](uint32_t idx) -> std::array<uint32_t, 4> {
        std::array<uint32_t, 4> result{};
        for (int c = 0; c < 4; ++c) {
          result[c] = regs[XE_GPU_REG_SHADER_CONSTANT_256_X + idx * 4 + c];
        }
        return result;
      };
      const auto c0 = get_ps_const_bits(0);
      const auto c1 = get_ps_const_bits(1);
      const auto c2 = get_ps_const_bits(2);
      const auto c27 = get_ps_const_bits(27);
      const auto c140 = get_ps_const_bits(140);
      const auto c141 = get_ps_const_bits(141);
      const auto c142 = get_ps_const_bits(142);
      const auto c254 = get_ps_const_bits(254);
      const auto c255 = get_ps_const_bits(255);
      LogFragmentDiagnostic(fmt::format(
          "Fragment diagnostic shader-math constants: frame={} draw={} PS={:016X} "
          "c0={:08X},{:08X},{:08X},{:08X} c1={:08X},{:08X},{:08X},{:08X} "
          "c2={:08X},{:08X},{:08X},{:08X} c27={:08X},{:08X},{:08X},{:08X} "
          "c140={:08X},{:08X},{:08X},{:08X} c141={:08X},{:08X},{:08X},{:08X} "
          "c142={:08X},{:08X},{:08X},{:08X} c254={:08X},{:08X},{:08X},{:08X} "
          "c255={:08X},{:08X},{:08X},{:08X}",
          frame_current_, diagnostic_draw_ordinal, diagnostic_raw_pixel_hash,
          c0[0], c0[1], c0[2], c0[3], c1[0], c1[1], c1[2], c1[3], c2[0], c2[1], c2[2],
          c2[3], c27[0], c27[1], c27[2], c27[3], c140[0], c140[1], c140[2], c140[3],
          c141[0], c141[1], c141[2], c141[3], c142[0], c142[1], c142[2], c142[3],
          c254[0], c254[1], c254[2], c254[3], c255[0], c255[1], c255[2], c255[3]));
    }
    if (diagnostic_raw_pixel_hash == diagnostics::kFragmentDiagnosticDeferredPixelHash) {
      const auto program_control = regs.Get<reg::SQ_PROGRAM_CNTL>();
      const auto context_misc = regs.Get<reg::SQ_CONTEXT_MISC>();
      LogFragmentDiagnostic(fmt::format(
          "Fragment diagnostic deferred input mapping: frame={} draw={} "
          "program_param_gen={} context_param_gen_pos={} resolved_param_gen_pos={} "
          "pixel_interpolator_mask={:08X} vertex_output_interpolator_mask={:08X}",
          frame_current_, diagnostic_draw_ordinal, uint32_t(program_control.param_gen),
          uint32_t(context_misc.param_gen_pos), ps_param_gen_pos, interpolator_mask,
          vertex_shader ? vertex_shader->writes_interpolators() : 0));
    }
  }

  const bool ben_watch_shader_pair =
      (draw_vs_hash == UINT64_C(0xD5F82E8E0317197D) &&
       draw_ps_hash == UINT64_C(0x18156850FF8D8BB1)) ||
      (draw_vs_hash == UINT64_C(0x7FF3708774FB5864) &&
       draw_ps_hash == UINT64_C(0xAC8BA94BD9146611));
  if (REX_UNLIKELY(diagnostic_snapshot_enabled_ && frame_current_ == 10000 && ben_watch_shader_pair)) {
    float native_bias_constant, native_bias_slope;
    draw_util::GetPreferredFacePolygonOffset(regs, primitive_polygonal, native_bias_slope,
                                             native_bias_constant);
    native_bias_constant *=
        regs.Get<reg::RB_DEPTH_INFO>().depth_format == xenos::DepthRenderTargetFormat::kD24S8
            ? draw_util::kD3D10PolygonOffsetFactorUnorm24
            : draw_util::kD3D10PolygonOffsetFactorFloat24;
    native_bias_slope *= xenos::kPolygonOffsetScaleSubpixelUnit *
                         float(std::max(draw_resolution_scale_x, draw_resolution_scale_y));
    draw_util::Scissor watch_scissor{};
    draw_util::GetScissor(regs, watch_scissor);
    REXGPU_WARN(
        "BEN_WATCH_DEPTH_STATE frame={} VS={:016X} PS={:016X} index_count={} "
        "raw_depth_control={:08X} raw_depth_info={:08X} scissor={},{} {}x{} "
        "poly_mode={:08X} raw_front=({:.9g},{:.9g}) raw_back=({:.9g},{:.9g}) "
        "guest_native_bias=({:.9g},{:.9g}) custom_eye_bias=0",
        frame_current_, draw_vs_hash, draw_ps_hash, index_count,
        regs[reg::RB_DEPTHCONTROL::register_index], regs[reg::RB_DEPTH_INFO::register_index],
        watch_scissor.offset[0], watch_scissor.offset[1], watch_scissor.extent[0],
        watch_scissor.extent[1], regs[reg::PA_SU_SC_MODE_CNTL::register_index],
        regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE),
        regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET),
        regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_SCALE),
        regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_OFFSET), native_bias_constant,
        native_bias_slope);
  }

  // Update dynamic graphics pipeline state.
  UpdateDynamicState(viewport_info, primitive_polygonal, normalized_depth_control);

  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();

  // Whether to load the guest 32-bit (usually big-endian) vertex index
  // indirectly in the vertex shader if full 32-bit indices are not supported by
  // the host.
  bool shader_32bit_index_dma =
      !device_properties.fullDrawIndexUint32 &&
      primitive_processing_result.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA &&
      vgt_draw_initiator.index_size == xenos::IndexFormat::kInt32 &&
      primitive_processing_result.host_vertex_shader_type == Shader::HostVertexShaderType::kVertex;
  if (!device_properties.fullDrawIndexUint32 &&
      primitive_processing_result.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA &&
      vgt_draw_initiator.index_size == xenos::IndexFormat::kInt32 &&
      primitive_processing_result.host_vertex_shader_type !=
          Shader::HostVertexShaderType::kVertex) {
    // PrimitiveProcessor is expected to pre-convert this case to
    // ProcessedIndexBufferType::kHostConverted with host-endian 24-bit
    // indices. Keep drawing, but report if that pre-conversion didn't happen.
    static bool non_vertex_32bit_guest_dma_no_full_uint32_logged = false;
    if (!non_vertex_32bit_guest_dma_no_full_uint32_logged) {
      REXGPU_WARN(
          "Vulkan draw has host vertex shader type {} with guest DMA 32-bit "
          "indices on a device without fullDrawIndexUint32; expected "
          "PrimitiveProcessor to return host-converted indices",
          uint32_t(primitive_processing_result.host_vertex_shader_type));
      non_vertex_32bit_guest_dma_no_full_uint32_logged = true;
    }
  }

  // Update system constants before uploading them.
  UpdateSystemConstantValues(primitive_polygonal, primitive_processing_result,
                             shader_32bit_index_dma, 0, viewport_info, used_texture_mask,
                             normalized_depth_control, normalized_color_mask);
  if (profile_cpu) {
    frame_cpu_profiling_stats_.time_dynamic_state +=
        std::chrono::steady_clock::now() - t_dyn_start;
  }

  // Update uniform buffers and descriptor sets after binding the pipeline with
  // the new layout.
  auto t_bind_start = profile_cpu ? std::chrono::steady_clock::now()
                                  : std::chrono::steady_clock::time_point{};
  if (!UpdateBindings(vertex_shader, pixel_shader)) {
    return draw_fail("update_bindings");
  }
  if (profile_cpu) {
    frame_cpu_profiling_stats_.time_binding_updates +=
        std::chrono::steady_clock::now() - t_bind_start;
  }

  // Ensure vertex buffers are resident.
  auto t_shmem_start = profile_cpu ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
  const Shader::ConstantRegisterMap& constant_map_vertex = vertex_shader->constant_register_map();
  for (uint32_t i = 0; i < rex::countof(constant_map_vertex.vertex_fetch_bitmap); ++i) {
    uint32_t vfetch_bits_remaining = constant_map_vertex.vertex_fetch_bitmap[i];
    uint32_t j;
    while (rex::bit_scan_forward(vfetch_bits_remaining, &j)) {
      vfetch_bits_remaining &= ~(uint32_t(1) << j);
      uint32_t vfetch_index = i * 32 + j;
      xenos::xe_gpu_vertex_fetch_t vfetch_constant = regs.GetVertexFetch(vfetch_index);
      switch (vfetch_constant.type) {
        case xenos::FetchConstantType::kVertex:
          break;
        case xenos::FetchConstantType::kInvalidVertex:
          if (REXCVAR_GET(gpu_allow_invalid_fetch_constants)) {
            break;
          }
          REXGPU_WARN(
              "Vertex fetch constant {} ({:08X} {:08X}) has \"invalid\" type! "
              "This "
              "is incorrect behavior, but you can try bypassing this by "
              "launching Xenia with --gpu_allow_invalid_fetch_constants=true.",
              vfetch_index, vfetch_constant.dword_0, vfetch_constant.dword_1);
          return false;
        default:
          REXGPU_WARN("Vertex fetch constant {} ({:08X} {:08X}) is completely invalid!",
                      vfetch_index, vfetch_constant.dword_0, vfetch_constant.dword_1);
          return false;
      }
      if (!shared_memory_->RequestRange(vfetch_constant.address << 2, vfetch_constant.size << 2)) {
        REXGPU_ERROR(
            "Failed to request vertex buffer at 0x{:08X} (size {}) in the shared "
            "memory",
            vfetch_constant.address << 2, vfetch_constant.size << 2);
        return false;
      }
    }
  }

  if (REX_UNLIKELY(diagnostic_storage_vertex_input_dump_mode_) && !diagnostic_storage_ben_occlusion_mode_ &&
      pixel_shader &&
      diagnostics::IsVertexInputDiagnosticHeadDraw(
          frame_current_, diagnostic_storage_draw_log_.ordinal(),
          vertex_diagnostic_scissor.extent[1],
          vertex_shader->ucode_data_hash(), pixel_shader->ucode_data_hash())) {
    const uint32_t dump_serial = ++diagnostic_vertex_input_dump_serial_;
    auto save_input = [&](const char* kind, uint32_t input_index, uint32_t physical_address,
                          uint64_t byte_count, uint32_t endian) {
      if (!byte_count) {
        LogFragmentDiagnostic(fmt::format(
            "Vertex input diagnostic file: frame={} serial={} kind={} index={} address=0x{:08X} "
            "bytes=0 endian={} status=refused_empty",
            frame_current_, dump_serial, kind, input_index, physical_address, endian));
        return;
      }
      if (byte_count > diagnostics::kVertexInputDiagnosticMaxBytes) {
        LogFragmentDiagnostic(fmt::format(
            "Vertex input diagnostic file: frame={} serial={} kind={} index={} address=0x{:08X} "
            "bytes={} endian={} status=refused_over_cap cap={}",
            frame_current_, dump_serial, kind, input_index, physical_address, byte_count, endian,
            diagnostics::kVertexInputDiagnosticMaxBytes));
        return;
      }
      if (!diagnostics::IsVertexInputDiagnosticRangeValid(physical_address, byte_count)) {
        LogFragmentDiagnostic(fmt::format(
            "Vertex input diagnostic file: frame={} serial={} kind={} index={} address=0x{:08X} "
            "bytes={} endian={} status=refused_range_overflow",
            frame_current_, dump_serial, kind, input_index, physical_address, byte_count, endian));
        return;
      }
      const uint8_t* input_data = memory_->TranslatePhysical<const uint8_t*>(physical_address);
      if (!input_data) {
        LogFragmentDiagnostic(fmt::format(
            "Vertex input diagnostic file: frame={} serial={} kind={} index={} address=0x{:08X} "
            "bytes={} endian={} status=refused_unmapped",
            frame_current_, dump_serial, kind, input_index, physical_address, byte_count, endian));
        return;
      }
      char path[512];
      const int path_length = std::snprintf(
          path, sizeof(path),
          "/Users/Kushal/Projects/cosmicR/out/face-shoulder-eye-audit-20261023/" 
          "frame-%llu-session-%llu-draw-%u-%s-%u-0x%08X.bin",
          static_cast<unsigned long long>(frame_current_),
          static_cast<unsigned long long>(diagnostic_storage_session_), dump_serial, kind, input_index,
          physical_address);
      if (path_length <= 0 || size_t(path_length) >= sizeof(path)) {
        LogFragmentDiagnostic(fmt::format(
            "Vertex input diagnostic file: frame={} serial={} kind={} index={} status=refused_path",
            frame_current_, dump_serial, kind, input_index));
        return;
      }
      FILE* output = std::fopen(path, "wb");
      if (!output) {
        LogFragmentDiagnostic(fmt::format(
            "Vertex input diagnostic file: frame={} serial={} kind={} index={} status=open_failed",
            frame_current_, dump_serial, kind, input_index));
        return;
      }
      const size_t written = std::fwrite(input_data, 1, size_t(byte_count), output);
      const int close_status = std::fclose(output);
      const bool saved = written == size_t(byte_count) && close_status == 0;
      LogFragmentDiagnostic(fmt::format(
          "Vertex input diagnostic file: frame={} serial={} kind={} index={} address=0x{:08X} "
          "bytes={} written={} endian={} status={} path='{}'",
          frame_current_, dump_serial, kind, input_index, physical_address, byte_count, written,
          endian, saved ? "saved" : "short_write", path));
    };

    const auto& fetch_bitmap = vertex_shader->constant_register_map().vertex_fetch_bitmap;
    for (uint32_t bitmap_index = 0; bitmap_index < rex::countof(fetch_bitmap); ++bitmap_index) {
      uint32_t bits = fetch_bitmap[bitmap_index];
      uint32_t bit_index;
      while (rex::bit_scan_forward(bits, &bit_index)) {
        bits &= ~(uint32_t(1) << bit_index);
        const uint32_t fetch_index = bitmap_index * 32 + bit_index;
        const auto fetch = regs.GetVertexFetch(fetch_index);
        if (fetch.type != xenos::FetchConstantType::kVertex) {
          LogFragmentDiagnostic(fmt::format(
              "Vertex input diagnostic fetch: frame={} serial={} fetch={} status=not_vertex type={}",
              frame_current_, dump_serial, fetch_index, uint32_t(fetch.type)));
          continue;
        }
        save_input("fetch", fetch_index, fetch.address << 2, uint64_t(fetch.size) << 2,
                   uint32_t(fetch.endian));
      }
    }
    if (index_buffer_info && index_buffer_info->length) {
      save_input("index", 0, index_buffer_info->guest_base, index_buffer_info->length,
                 uint32_t(index_buffer_info->endianness));
    } else {
      LogFragmentDiagnostic(fmt::format(
          "Vertex input diagnostic index: frame={} serial={} status=unavailable",
          frame_current_, dump_serial));
    }

    constexpr uint32_t kVertexConstantDwordCount = 256 * 4;
    const uint32_t* const vertex_constants =
        &regs.values[XE_GPU_REG_SHADER_CONSTANT_000_X];
    char constants_path[512];
    const int constants_path_length = std::snprintf(
        constants_path, sizeof(constants_path),
        "/Users/Kushal/Projects/cosmicR/out/face-shoulder-eye-audit-20261023/"
        "frame-%llu-session-%llu-serial-%u-vs-c0-c255.bin",
        static_cast<unsigned long long>(frame_current_),
        static_cast<unsigned long long>(diagnostic_storage_session_), dump_serial);
    FILE* constants_output = constants_path_length > 0 &&
                                     size_t(constants_path_length) < sizeof(constants_path)
                                 ? std::fopen(constants_path, "wb")
                                 : nullptr;
    const size_t constants_written = constants_output
                                         ? std::fwrite(vertex_constants, sizeof(uint32_t),
                                                       kVertexConstantDwordCount, constants_output)
                                         : 0;
    const int constants_close_status = constants_output ? std::fclose(constants_output) : -1;
    const bool constants_saved = constants_written == kVertexConstantDwordCount &&
                                 constants_close_status == 0;
    LogFragmentDiagnostic(fmt::format(
        "Vertex constant diagnostic file: frame={} serial={} registers=c0..c255 bytes={} "
        "status={} path='{}'",
        frame_current_, dump_serial, kVertexConstantDwordCount * sizeof(uint32_t),
        constants_saved ? "saved" : "save_failed", constants_saved ? constants_path : ""));
    LogFragmentDiagnostic(fmt::format(
        "Vertex raster state: frame={} serial={} draw={} base_vertex=0 first_index=0 "
        "instances=1 index_count={} PA_SU_SC_MODE_CNTL={:08X} PA_CL_CLIP_CNTL={:08X} "
        "PA_CL_VTE_CNTL={:08X} PA_SC_WINDOW_OFFSET={:08X} RB_DEPTHCONTROL={:08X} "
        "RB_DEPTH_INFO={:08X} RB_COLORCONTROL={:08X} RB_COLOR_INFO={:08X} "
        "RB_STENCILREFMASK={:08X} RB_BLENDCONTROL0={:08X} RB_ALPHA_REF={:08X}",
        frame_current_, dump_serial, diagnostic_storage_draw_log_.ordinal(), index_count,
        regs[reg::PA_SU_SC_MODE_CNTL::register_index], regs[reg::PA_CL_CLIP_CNTL::register_index],
        regs[reg::PA_CL_VTE_CNTL::register_index], regs[reg::PA_SC_WINDOW_OFFSET::register_index],
        regs[reg::RB_DEPTHCONTROL::register_index], regs[reg::RB_DEPTH_INFO::register_index],
        regs[reg::RB_COLORCONTROL::register_index], regs[reg::RB_COLOR_INFO::register_index],
        regs[reg::RB_STENCILREFMASK::register_index], regs[reg::RB_BLENDCONTROL::register_index],
        regs[XE_GPU_REG_RB_ALPHA_REF]));
  }


  // Synchronize the memory pages backing memory scatter export streams, and
  // calculate the range that includes the streams for the buffer barrier.
  uint32_t memexport_extent_start = UINT32_MAX, memexport_extent_end = 0;
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    uint32_t memexport_range_base_bytes = memexport_range.base_address_dwords << 2;
    if (!shared_memory_->RequestRange(memexport_range_base_bytes, memexport_range.size_bytes)) {
      REXGPU_ERROR(
          "Failed to request memexport stream at 0x{:08X} (size {}) in the "
          "shared memory",
          memexport_range_base_bytes, memexport_range.size_bytes);
      return false;
    }
    memexport_extent_start = std::min(memexport_extent_start, memexport_range_base_bytes);
    memexport_extent_end =
        std::max(memexport_extent_end, memexport_range_base_bytes + memexport_range.size_bytes);
  }
  if (memexport_writes_possible && memexport_ranges_.empty()) {
    if (!shared_memory_->RequestRange(0, SharedMemory::kBufferSize)) {
      REXGPU_ERROR(
          "Failed to request full shared memory residency for unresolved "
          "memexport destinations");
      return false;
    }
  }
  if (profile_cpu) {
    frame_cpu_profiling_stats_.time_shared_memory_residency +=
        std::chrono::steady_clock::now() - t_shmem_start;
  }

  ScratchBufferAcquisition guest_dma_index_scratch_buffer;
  if (primitive_processing_result.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA &&
      !shader_32bit_index_dma && memexport_writes_possible) {
    // Align with D3D12 behavior where memexporting draws using guest DMA indices
    // read a snapshot of indices to avoid read/write aliasing through shared
    // memory.
    VkDeviceSize guest_dma_index_size =
        VkDeviceSize(primitive_processing_result.host_draw_vertex_count) *
        (primitive_processing_result.host_index_format == xenos::IndexFormat::kInt16
             ? sizeof(uint16_t)
             : sizeof(uint32_t));
    if (guest_dma_index_size) {
      guest_dma_index_scratch_buffer = AcquireScratchGpuBuffer(
          guest_dma_index_size, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
      if (guest_dma_index_scratch_buffer.buffer() == VK_NULL_HANDLE) {
        return draw_fail("guest_dma_index_scratch_buffer");
      }

      shared_memory_->Use(VulkanSharedMemory::Usage::kRead);
      SubmitBarriers(true);

      VkBufferCopy guest_dma_index_copy_region = {};
      guest_dma_index_copy_region.srcOffset = primitive_processing_result.guest_index_base;
      guest_dma_index_copy_region.dstOffset = 0;
      guest_dma_index_copy_region.size = guest_dma_index_size;
      deferred_command_buffer_.CmdVkCopyBuffer(shared_memory_->buffer(),
                                               guest_dma_index_scratch_buffer.buffer(), 1,
                                               &guest_dma_index_copy_region);
      PushBufferMemoryBarrier(
          guest_dma_index_scratch_buffer.buffer(), 0, guest_dma_index_size,
          guest_dma_index_scratch_buffer.GetStageMask(), VK_PIPELINE_STAGE_VERTEX_INPUT_BIT,
          guest_dma_index_scratch_buffer.GetAccessMask(), VK_ACCESS_INDEX_READ_BIT);
      guest_dma_index_scratch_buffer.SetStageMask(VK_PIPELINE_STAGE_VERTEX_INPUT_BIT);
      guest_dma_index_scratch_buffer.SetAccessMask(VK_ACCESS_INDEX_READ_BIT);
    }
  }

  // Insert the shared memory barrier if needed.
  // TODO(Triang3l): Find some PM4 command that can be used for indication of
  // when memexports should be awaited instead of inserting the barrier in Use
  // every time if memory export was done in the previous draw?
  if (memexport_extent_start < memexport_extent_end) {
    shared_memory_->Use(
        VulkanSharedMemory::Usage::kGuestDrawReadWrite,
        std::make_pair(memexport_extent_start, memexport_extent_end - memexport_extent_start));
  } else if (memexport_writes_possible) {
    // Stream constants can be invalid or dynamic, making the exact destination
    // unknown. Keep synchronization conservative similarly to D3D12 UAV
    // handling for memexport-capable draws.
    shared_memory_->Use(VulkanSharedMemory::Usage::kGuestDrawReadWrite,
                        std::make_pair(0u, SharedMemory::kBufferSize));
  } else {
    shared_memory_->Use(VulkanSharedMemory::Usage::kRead);
  }

  // After all commands that may dispatch, copy or insert barriers, submit
  // the barriers (may end the render pass), and (re)enter the render pass
  // before drawing.
  auto t_bar_start = profile_cpu ? std::chrono::steady_clock::now()
                                 : std::chrono::steady_clock::time_point{};
  SubmitBarriersAndEnterRenderTargetCacheRenderPass(
      render_target_cache_->last_update_render_pass(),
      render_target_cache_->last_update_framebuffer());
  if (profile_cpu) {
    frame_cpu_profiling_stats_.time_barriers_and_render_pass +=
        std::chrono::steady_clock::now() - t_bar_start;
  }

  bool diagnostic_snapshot_target_draw = false;
  uint32_t diagnostic_ben_candidate_kind = 0;
  uint32_t diagnostic_ben_capture_origin_y = 0;
  uint32_t diagnostic_ben_output_y_offset = 0;
  auto record_diagnostic_snapshot_boundary = [&](diagnostics::RenderTargetSnapshotPoint point) {
    diagnostics::RenderTargetSnapshotBoundary boundary{};
    boundary.frame = frame_current_;
    boundary.draw_ordinal = diagnostic_snapshot_draw_ordinal_;
    boundary.point = uint32_t(point);
    if (diagnostic_snapshot_request_.compact_ben) {
      boundary.capture_flags = diagnostic_ben_candidate_kind << 16;
    }
    boundary.vertex_shader_hash = vertex_shader->ucode_data_hash();
    boundary.pixel_shader_hash = pixel_shader ? pixel_shader->ucode_data_hash() : 0;
    boundary.depth_control = regs[reg::RB_DEPTHCONTROL::register_index];
    boundary.color_control = regs[reg::RB_COLORCONTROL::register_index];
    boundary.stencil_ref_mask = regs[reg::RB_STENCILREFMASK::register_index];
    boundary.blend_control_0 = regs[reg::RB_BLENDCONTROL::register_index];
    boundary.color_info_0 = regs[reg::RB_COLOR_INFO::rt_register_indices[0]];
    boundary.depth_info = regs[reg::RB_DEPTH_INFO::register_index];
    boundary.color_write_mask = normalized_color_mask;
    boundary.coverage_control = regs[reg::PA_SU_SC_MODE_CNTL::register_index];
    boundary.alpha_ref = regs[XE_GPU_REG_RB_ALPHA_REF];
    draw_util::Scissor snapshot_scissor{};
    draw_util::GetScissor(regs, snapshot_scissor);
    boundary.scissor_x = uint32_t(snapshot_scissor.offset[0]);
    boundary.scissor_y = uint32_t(snapshot_scissor.offset[1]);
    boundary.scissor_width = snapshot_scissor.extent[0];
    boundary.scissor_height = snapshot_scissor.extent[1];
    diagnostics::BenOcclusionSnapshotBoundaryMetadata ben_metadata{};
    ben_metadata.viewport_x = float(viewport_info.xy_offset[0]);
    ben_metadata.viewport_y = float(viewport_info.xy_offset[1]);
    ben_metadata.viewport_width = viewport_info.xy_extent[0];
    ben_metadata.viewport_height = viewport_info.xy_extent[1];
    ben_metadata.viewport_min_depth = viewport_info.z_min;
    ben_metadata.viewport_max_depth = viewport_info.z_max;
    ben_metadata.index_count = index_count;
    ben_metadata.candidate_kind = diagnostic_ben_candidate_kind;
    ben_metadata.capture_origin_y = diagnostic_ben_capture_origin_y;
    ben_metadata.output_y_offset = diagnostic_ben_output_y_offset;
    ben_metadata.eyeball_depth_bias_active = 0;
    const bool captured = render_target_cache_->CaptureDiagnosticSnapshot(
        diagnostic_snapshot_request_, boundary, ben_metadata);
    if (!captured) {
      REXGPU_WARN("Render target snapshot boundary unavailable: frame={} draw={} point={}",
                  frame_current_, diagnostic_snapshot_draw_ordinal_, uint32_t(point));
    }
    return captured;
  };
  if (REX_UNLIKELY(diagnostic_snapshot_enabled_)) {
    bool diagnostic_snapshot_shader_match = false;
    switch (diagnostic_snapshot_request_.target) {
      case diagnostics::RenderTargetSnapshotTarget::kQuadPartner:
        diagnostic_snapshot_shader_match =
            diagnostics::IsQuadPartnerMarkerShader(diagnostic_raw_pixel_hash);
        break;
      case diagnostics::RenderTargetSnapshotTarget::kHeadShader:
        diagnostic_snapshot_shader_match =
            diagnostic_raw_vertex_hash == diagnostics::kRenderTargetSnapshotHeadVertexHash &&
            diagnostic_raw_pixel_hash == diagnostics::kRenderTargetSnapshotHeadPixelHash;
        break;
      case diagnostics::RenderTargetSnapshotTarget::kGBufferHeadShader:
        diagnostic_snapshot_shader_match =
            diagnostic_raw_vertex_hash == diagnostics::kRenderTargetSnapshotGBufferVertexHash &&
            diagnostic_raw_pixel_hash == diagnostics::kRenderTargetSnapshotGBufferPixelHash;
        break;
      case diagnostics::RenderTargetSnapshotTarget::kDeferredLighting:
        diagnostic_snapshot_shader_match =
            diagnostic_raw_vertex_hash == diagnostics::kRenderTargetSnapshotDeferredVertexHash &&
            diagnostic_raw_pixel_hash == diagnostics::kRenderTargetSnapshotDeferredPixelHash;
        break;
      case diagnostics::RenderTargetSnapshotTarget::kBenCuff: {
        if (diagnostic_snapshot_request_.compact_ben) {
          if (diagnostic_snapshot_request_.ben_pixel_history) {
            diagnostic_ben_candidate_kind = diagnostics::kBenOcclusionCandidateAllDraw;
            diagnostic_snapshot_shader_match = true;
          } else {
            bool is_hud_glow = false;
            if (diagnostic_raw_vertex_hash == UINT64_C(0x7A0841365E75A823) &&
                diagnostic_raw_pixel_hash == UINT64_C(0x7ABDFEFBCADB4BF9)) {
              float c11_z = 0.0f;
              float c11_w = 0.0f;
              std::memcpy(&c11_z,
                          &regs[XE_GPU_REG_SHADER_CONSTANT_000_X + 11 * 4 + 2],
                          sizeof(c11_z));
              std::memcpy(&c11_w,
                          &regs[XE_GPU_REG_SHADER_CONSTANT_000_X + 11 * 4 + 3],
                          sizeof(c11_w));
              is_hud_glow = c11_z == 0.0f && c11_w == 1.0f;
            }
            diagnostic_ben_candidate_kind = diagnostics::GetBenOcclusionCandidateKind(
                diagnostic_raw_vertex_hash, diagnostic_raw_pixel_hash, is_hud_glow);
            diagnostic_snapshot_shader_match = diagnostic_ben_candidate_kind != 0;
          }
        } else {
          diagnostic_snapshot_shader_match =
              diagnostic_raw_vertex_hash == UINT64_C(0x046523BA1EB2717E) &&
              diagnostic_raw_pixel_hash == UINT64_C(0x2D4C010D3D75DEA5) && index_count == 1836 &&
              viewport_info.xy_extent[1] == 336;
        }
      } break;
    }
    const bool diagnostic_snapshot_requested_draw =
        diagnostic_snapshot_request_.draw_ordinal
            ? diagnostic_snapshot_request_.matches(frame_current_,
                                                   diagnostic_snapshot_draw_ordinal_)
            : diagnostic_snapshot_shader_match;
    bool diagnostic_ben_roi_intersects_scissor = true;
    if (diagnostic_snapshot_request_.compact_ben && diagnostic_snapshot_shader_match) {
      draw_util::Scissor ben_scissor{};
      draw_util::GetScissor(regs, ben_scissor);
      const int64_t scissor_left = ben_scissor.offset[0];
      const int64_t scissor_top = ben_scissor.offset[1];
      const int64_t scissor_right = scissor_left + ben_scissor.extent[0];
      const int64_t scissor_bottom = scissor_top + ben_scissor.extent[1];
      if (diagnostic_snapshot_request_.ben_pixel_history) {
        if (ben_scissor.extent[1] == 336) {
          diagnostic_ben_capture_origin_y = 64;
          diagnostic_ben_output_y_offset = 384;
        } else if (ben_scissor.extent[1] == 720) {
          diagnostic_ben_capture_origin_y = 448;
          diagnostic_ben_output_y_offset = 0;
        } else {
          diagnostic_ben_roi_intersects_scissor = false;
        }
      }
      const int64_t roi_left = diagnostic_snapshot_request_.x;
      const int64_t roi_top = diagnostic_snapshot_request_.ben_pixel_history
                                  ? diagnostic_ben_capture_origin_y
                                  : diagnostic_snapshot_request_.y;
      const int64_t roi_right = roi_left + diagnostic_snapshot_request_.width;
      const int64_t roi_bottom = roi_top + diagnostic_snapshot_request_.height;
      const bool expected_ben_scissor_height = diagnostic_snapshot_request_.ben_pixel_history ||
          (diagnostic_ben_candidate_kind == diagnostics::kBenOcclusionCandidateGlow
               ? (ben_scissor.extent[1] == 336 || ben_scissor.extent[1] == 720)
               : ben_scissor.extent[1] == 336);
      diagnostic_ben_roi_intersects_scissor = diagnostic_ben_roi_intersects_scissor &&
          expected_ben_scissor_height && scissor_left < roi_right && roi_left < scissor_right &&
          scissor_top < roi_bottom && roi_top < scissor_bottom;
    }
    diagnostic_snapshot_target_draw =
        (diagnostic_snapshot_request_.compact_ben || !diagnostic_snapshot_head_draw_seen_ ||
         diagnostic_snapshot_request_.target == diagnostics::RenderTargetSnapshotTarget::kQuadPartner) &&
        diagnostic_snapshot_requested_draw &&
        diagnostic_ben_roi_intersects_scissor &&
        frame_current_ == diagnostic_snapshot_request_.frame;
    if (diagnostic_snapshot_target_draw) {
      record_diagnostic_snapshot_boundary(diagnostics::RenderTargetSnapshotPoint::kBeforeHeadDraw);
    }
  }

  // Draw.
  auto t_vkdraw_start = profile_cpu ? std::chrono::steady_clock::now()
                                    : std::chrono::steady_clock::time_point{};
  if (primitive_processing_result.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kNone ||
      shader_32bit_index_dma) {
    NoteScopeDraw(pixel_shader);
    deferred_command_buffer_.CmdVkDraw(primitive_processing_result.host_draw_vertex_count, 1, 0, 0);
  } else {
    std::pair<VkBuffer, VkDeviceSize> index_buffer;
    switch (primitive_processing_result.index_buffer_type) {
      case PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA:
        if (guest_dma_index_scratch_buffer.buffer() != VK_NULL_HANDLE) {
          index_buffer.first = guest_dma_index_scratch_buffer.buffer();
          index_buffer.second = 0;
        } else {
          index_buffer.first = shared_memory_->buffer();
          index_buffer.second = primitive_processing_result.guest_index_base;
        }
        break;
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted:
        index_buffer = primitive_processor_->GetConvertedIndexBuffer(
            primitive_processing_result.host_index_buffer_handle);
        break;
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForAuto:
      case PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA:
        index_buffer = primitive_processor_->GetBuiltinIndexBuffer(
            primitive_processing_result.host_index_buffer_handle);
        break;
      default:
        assert_unhandled_case(primitive_processing_result.index_buffer_type);
        return draw_fail("unexpected_index_buffer_type");
    }
    VkIndexType index_type =
        primitive_processing_result.host_index_format == xenos::IndexFormat::kInt16
            ? VK_INDEX_TYPE_UINT16
            : VK_INDEX_TYPE_UINT32;
    if (bound_index_buffer_ != index_buffer.first ||
        bound_index_buffer_offset_ != index_buffer.second ||
        bound_index_type_ != index_type) {
      deferred_command_buffer_.CmdVkBindIndexBuffer(index_buffer.first, index_buffer.second,
                                                    index_type);
      bound_index_buffer_ = index_buffer.first;
      bound_index_buffer_offset_ = index_buffer.second;
      bound_index_type_ = index_type;
    }
    NoteScopeDraw(pixel_shader);
    deferred_command_buffer_.CmdVkDrawIndexed(primitive_processing_result.host_draw_vertex_count, 1,
                                              0, 0, 0);
  }
  if (raster_variant_active) {
    // Both indexed and non-indexed guest draw commands have been recorded.
    // Failed/skipped/retried attempts never reach this consumer seam.
    raster_variant_audit_.Record(pixel_shader->ucode_data_hash(),
                                  pixel_shader_modification.value, raster_variant_decision);
  }
  if (pixel_shader_modification.pixel.depth_stencil_mode ==
      SpirvShaderTranslator::Modification::DepthStencilMode::kEarlyHint) {
    ++current_frame_early_z_draws_;
  }
  if (diagnostic_capture_active) {
    diagnostic_draw_recorded = true;
  }
  if (diagnostic_snapshot_target_draw) {
    const bool captured =
        record_diagnostic_snapshot_boundary(diagnostics::RenderTargetSnapshotPoint::kAfterHeadDraw);
    diagnostic_snapshot_head_draw_seen_ |= captured;
  } else if (diagnostic_snapshot_enabled_ && !diagnostic_snapshot_request_.compact_ben &&
             diagnostic_snapshot_head_draw_seen_ &&
             diagnostic_snapshot_request_.after_each_draw &&
             frame_current_ == diagnostic_snapshot_request_.frame) {
    record_diagnostic_snapshot_boundary(
        diagnostics::RenderTargetSnapshotPoint::kAfterSubsequentDraw);
  }
  if (profile_cpu) {
    frame_cpu_profiling_stats_.time_vk_draw +=
        std::chrono::steady_clock::now() - t_vkdraw_start;
  }

  // Invalidate textures in memexported memory and watch for changes.
  if (!memexport_ranges_.empty()) {
    for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
      TraceWaitDestination("memexport", frame_current_, memexport_range.base_address_dwords << 2,
                           memexport_range.size_bytes);
      shared_memory_->RangeWrittenByGpu(memexport_range.base_address_dwords << 2,
                                        memexport_range.size_bytes);
    }
  } else if (memexport_writes_possible) {
    // Stream constants can be invalid or dynamic, so exact destinations may be
    // unknown. Keep invalidation conservative in this case.
    TraceWaitDestination("memexport_unknown", frame_current_, 0, SharedMemory::kBufferSize);
    shared_memory_->RangeWrittenByGpu(0, SharedMemory::kBufferSize);
  }

  if (IsReadbackMemexportEnabled(REXCVAR_GET(vulkan_readback_memexport)) &&
      !memexport_ranges_.empty()) {
    uint32_t memexport_total_size = 0;
    for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
      memexport_total_size += memexport_range.size_bytes;
    }
    if (memexport_total_size) {
      if (REXCVAR_GET(readback_memexport_fast)) {
        IssueDraw_MemexportReadbackFastPath(memexport_total_size);
      } else {
        IssueDraw_MemexportReadbackFullPath(memexport_total_size);
      }
    }
  }

  if (provenance_draw_id &&
      (provenance_draw_id <= 10 || provenance_draw_id % 100 == 0 || provenance_draw_id == 1000)) {
    REXGPU_INFO("GPU provenance frame {} draw {} status=commands_recorded", frame_current_,
                provenance_draw_id);
  }
  if (current_dynamic_rendering_has_depth_) {
    frame_pass_stats_.draws_with_depth++;
  } else {
    frame_pass_stats_.draws_color_only++;
  }
  if (memexport_used_vertex) {
    frame_pass_stats_.draws_memexport++;
  }
  if (profile_cpu) frame_cpu_profiling_stats_.draw_count++;
  return true;
}

bool VulkanCommandProcessor::IssueDraw_MemexportReadbackFullPath(uint32_t total_size) {
  if (!total_size || memexport_ranges_.empty()) {
    return true;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  VkBuffer readback_buffer = VK_NULL_HANDLE;
  VkDeviceMemory readback_memory = VK_NULL_HANDLE;
  uint32_t readback_memory_type = UINT32_MAX;
  VkDeviceSize readback_memory_size = 0;
  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
          vulkan_device, total_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          ui::vulkan::util::MemoryPurpose::kReadback, readback_buffer, readback_memory,
          &readback_memory_type, &readback_memory_size)) {
    REXGPU_ERROR("Failed to create a Vulkan memexport readback buffer");
    return true;
  }

  shared_memory_->Use(VulkanSharedMemory::Usage::kRead);
  SubmitBarriers(true);

  uint32_t readback_offset = 0;
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    VkBufferCopy readback_region = {};
    readback_region.srcOffset = memexport_range.base_address_dwords << 2;
    readback_region.dstOffset = readback_offset;
    readback_region.size = memexport_range.size_bytes;
    deferred_command_buffer_.CmdVkCopyBuffer(shared_memory_->buffer(), readback_buffer, 1,
                                             &readback_region);
    readback_offset += memexport_range.size_bytes;
  }
  PushBufferMemoryBarrier(readback_buffer, 0, total_size, VK_PIPELINE_STAGE_TRANSFER_BIT,
                          VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                          VK_ACCESS_HOST_READ_BIT);

  const bool audit_wait = REXCVAR_GET(gpu_frame_audit_enable);
  auto audit_wait_start = audit_wait ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  bool wait_completed = AwaitAllQueueOperationsCompletion();
  if (audit_wait) {
    frame_cpu_profiling_stats_.time_memexport_wait += std::chrono::steady_clock::now() - audit_wait_start;
    ++frame_cpu_profiling_stats_.memexport_wait_count;
  }
  if (!wait_completed) {
    REXGPU_ERROR("Failed to await completion of Vulkan memexport readback");
    dfn.vkDestroyBuffer(device, readback_buffer, nullptr);
    dfn.vkFreeMemory(device, readback_memory, nullptr);
    return true;
  }

  void* readback_mapping = nullptr;
  if (dfn.vkMapMemory(device, readback_memory, 0, VK_WHOLE_SIZE, 0, &readback_mapping) !=
      VK_SUCCESS) {
    REXGPU_ERROR("Failed to map a Vulkan memexport readback buffer");
    dfn.vkDestroyBuffer(device, readback_buffer, nullptr);
    dfn.vkFreeMemory(device, readback_memory, nullptr);
    return true;
  }

  if (!(vulkan_device->memory_types().host_coherent & (uint32_t(1) << readback_memory_type))) {
    VkMappedMemoryRange readback_memory_range = {};
    readback_memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    readback_memory_range.memory = readback_memory;
    readback_memory_range.offset = 0;
    readback_memory_range.size = std::min(
        rex::round_up(VkDeviceSize(total_size), vulkan_device->properties().nonCoherentAtomSize),
        readback_memory_size);
    dfn.vkInvalidateMappedMemoryRanges(device, 1, &readback_memory_range);
  }

  const uint8_t* readback_bytes = reinterpret_cast<const uint8_t*>(readback_mapping);
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    std::memcpy(memory_->TranslatePhysical(memexport_range.base_address_dwords << 2),
                readback_bytes, memexport_range.size_bytes);
    readback_bytes += memexport_range.size_bytes;
  }
  dfn.vkUnmapMemory(device, readback_memory);
  dfn.vkDestroyBuffer(device, readback_buffer, nullptr);
  dfn.vkFreeMemory(device, readback_memory, nullptr);
  return true;
}

bool VulkanCommandProcessor::IssueDraw_MemexportReadbackFastPath(uint32_t total_size) {
  if (!total_size || memexport_ranges_.empty()) {
    return true;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  const uint64_t readback_key =
      MakeMemexportReadbackKey(memexport_ranges_.front().base_address_dwords, total_size);
  ReadbackBuffer& readback = memexport_readback_buffers_[readback_key];
  readback.last_used_frame = frame_current_;

  auto ensure_readback_slot = [&](uint32_t index, uint32_t size) -> bool {
    if (readback.buffers[index] != VK_NULL_HANDLE && readback.memories[index] != VK_NULL_HANDLE &&
        readback.mapped_data[index] != nullptr && size <= readback.sizes[index]) {
      return true;
    }

    VkBuffer new_buffer = VK_NULL_HANDLE;
    VkDeviceMemory new_memory = VK_NULL_HANDLE;
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
            vulkan_device, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            ui::vulkan::util::MemoryPurpose::kReadback, new_buffer, new_memory)) {
      return false;
    }

    void* new_mapping = nullptr;
    if (dfn.vkMapMemory(device, new_memory, 0, VK_WHOLE_SIZE, 0, &new_mapping) != VK_SUCCESS) {
      dfn.vkDestroyBuffer(device, new_buffer, nullptr);
      dfn.vkFreeMemory(device, new_memory, nullptr);
      return false;
    }

    if (readback.buffers[index] != VK_NULL_HANDLE || readback.memories[index] != VK_NULL_HANDLE) {
      if (!AwaitAllQueueOperationsCompletion()) {
        dfn.vkUnmapMemory(device, new_memory);
        dfn.vkDestroyBuffer(device, new_buffer, nullptr);
        dfn.vkFreeMemory(device, new_memory, nullptr);
        return false;
      }
      if (readback.mapped_data[index] != nullptr && readback.memories[index] != VK_NULL_HANDLE) {
        dfn.vkUnmapMemory(device, readback.memories[index]);
      }
      if (readback.buffers[index] != VK_NULL_HANDLE) {
        dfn.vkDestroyBuffer(device, readback.buffers[index], nullptr);
      }
      if (readback.memories[index] != VK_NULL_HANDLE) {
        dfn.vkFreeMemory(device, readback.memories[index], nullptr);
      }
    }

    readback.buffers[index] = new_buffer;
    readback.memories[index] = new_memory;
    readback.mapped_data[index] = new_mapping;
    readback.sizes[index] = size;
    readback.submission_written[index] = 0;
    readback.written_size[index] = 0;
    return true;
  };

  const uint32_t write_index = readback.current_index;
  const uint32_t read_index = 1 - write_index;
  const uint32_t readback_size = AlignReadbackBufferSize(total_size);
  if (!ensure_readback_slot(write_index, readback_size)) {
    return IssueDraw_MemexportReadbackFullPath(total_size);
  }

  shared_memory_->Use(VulkanSharedMemory::Usage::kRead);
  SubmitBarriers(true);

  uint32_t readback_offset = 0;
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    VkBufferCopy readback_region = {};
    readback_region.srcOffset = memexport_range.base_address_dwords << 2;
    readback_region.dstOffset = readback_offset;
    readback_region.size = memexport_range.size_bytes;
    deferred_command_buffer_.CmdVkCopyBuffer(shared_memory_->buffer(),
                                             readback.buffers[write_index], 1, &readback_region);
    readback_offset += memexport_range.size_bytes;
  }
  PushBufferMemoryBarrier(readback.buffers[write_index], 0, total_size,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
  readback.submission_written[write_index] = GetCurrentSubmission();
  readback.written_size[write_index] = total_size;

  CheckSubmissionFenceAndDeviceLoss(0);
  bool previous_slot_ready =
      readback.buffers[read_index] != VK_NULL_HANDLE &&
      readback.memories[read_index] != VK_NULL_HANDLE &&
      readback.mapped_data[read_index] != nullptr && total_size <= readback.sizes[read_index] &&
      total_size <= readback.written_size[read_index] && readback.submission_written[read_index] &&
      readback.submission_written[read_index] <= submission_completed_;
  if (!previous_slot_ready) {
    IssueDraw_MemexportReadbackFullPath(total_size);
    readback.current_index = read_index;
    return true;
  }

  VkMappedMemoryRange readback_memory_range = {};
  readback_memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
  readback_memory_range.memory = readback.memories[read_index];
  readback_memory_range.offset = 0;
  readback_memory_range.size = VK_WHOLE_SIZE;
  dfn.vkInvalidateMappedMemoryRanges(device, 1, &readback_memory_range);

  const uint8_t* readback_bytes =
      reinterpret_cast<const uint8_t*>(readback.mapped_data[read_index]);
  for (const draw_util::MemExportRange& memexport_range : memexport_ranges_) {
    std::memcpy(memory_->TranslatePhysical(memexport_range.base_address_dwords << 2),
                readback_bytes, memexport_range.size_bytes);
    readback_bytes += memexport_range.size_bytes;
  }

  readback.current_index = read_index;
  return true;
}

bool VulkanCommandProcessor::IssueCopy() {
  FrameCpuTimeline::Scope audit_copy_scope(GetFrameCpuTimeline(), FrameCpuTimeline::Bucket::kCopy);
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  // The copy timer includes nested submission and waits; report it separately.
  struct CopyAudit {
    FrameCpuProfilingStats& stats;
    bool enabled;
    std::chrono::steady_clock::time_point start;
    ~CopyAudit() {
      if (enabled) {
        stats.time_copy += std::chrono::steady_clock::now() - start;
        ++stats.copy_count;
      }
    }
  } copy_audit{frame_cpu_profiling_stats_, REXCVAR_GET(gpu_frame_audit_enable),
               REXCVAR_GET(gpu_frame_audit_enable) ? std::chrono::steady_clock::now()
                                                    : std::chrono::steady_clock::time_point{}};
  if (!BeginSubmission(true)) {
    return false;
  }
  ReadbackResolveMode readback_mode = GetReadbackResolveMode(REXCVAR_GET(vulkan_readback_resolve));
  // [DEBUG-face-resolve] Capture the current GPU resolve, not the delayed CPU slot.
  const bool face_resolve_probe = (frame_current_ == 8680 || frame_current_ == 6000) &&
      std::ifstream("/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/resolve-probe/enable.txt").good();
  if (readback_mode == ReadbackResolveMode::kDisabled && !face_resolve_probe) {
    uint32_t written_address = 0, written_length = 0;
    frame_work_counters_.resolve_attempts++;
    bool resolved = render_target_cache_->Resolve(*memory_, *shared_memory_, *texture_cache_,
                                                  written_address, written_length);
    if (resolved) {
      TraceWaitDestination("resolve", frame_current_, written_address, written_length);
      if (written_length > 0) {
        frame_work_counters_.resolve_successes++;
        frame_work_counters_.resolve_bytes += written_length;
      } else {
        frame_work_counters_.resolve_empty_or_noop++;
      }
    } else {
      frame_work_counters_.resolve_empty_or_noop++;
    }
    if (IsGpuProvenanceFrame(frame_current_)) {
      REXGPU_INFO(
          "GPU provenance frame {} resolve success={} physical_destination=0x{:08X} "
          "length=0x{:X}",
          frame_current_, resolved, written_address, written_length);
    }
    if (FlickerDiagEnabled()) {
      flicker_frame_csv.NoteCopy(resolved, written_address, written_length);
    }
    return resolved;
  }

  return IssueCopy_ReadbackResolvePath();
}

bool VulkanCommandProcessor::IssueCopy_ReadbackResolvePath() {
  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  uint32_t written_address = 0, written_length = 0;
  frame_work_counters_.resolve_attempts++;
  if (!render_target_cache_->Resolve(*memory_, *shared_memory_, *texture_cache_, written_address,
                                     written_length)) {
    frame_work_counters_.resolve_empty_or_noop++;
    if (IsGpuProvenanceFrame(frame_current_)) {
      REXGPU_INFO("GPU provenance frame {} resolve success=false", frame_current_);
    }
    if (FlickerDiagEnabled()) {
      flicker_frame_csv.NoteCopy(false, written_address, written_length);
    }
    return false;
  }
  TraceWaitDestination("resolve", frame_current_, written_address, written_length);
  if (written_length > 0) {
    frame_work_counters_.resolve_successes++;
    frame_work_counters_.resolve_bytes += written_length;
  } else {
    frame_work_counters_.resolve_empty_or_noop++;
  }
  if (FlickerDiagEnabled()) {
    flicker_frame_csv.NoteCopy(true, written_address, written_length);
  }
  if (IsGpuProvenanceFrame(frame_current_)) {
    REXGPU_INFO(
        "GPU provenance frame {} resolve success=true physical_destination=0x{:08X} "
        "length=0x{:X}",
        frame_current_, written_address, written_length);
  }

  if (!written_length) {
    return true;
  }

  if (!memory_->TranslatePhysical(written_address)) {
    return true;
  }

  const bool face_resolve_probe = (frame_current_ == 8680 || frame_current_ == 6000) &&
      std::ifstream("/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/resolve-probe/enable.txt").good();
  ReadbackResolveMode readback_mode = face_resolve_probe ? ReadbackResolveMode::kFull
      : GetReadbackResolveMode(REXCVAR_GET(vulkan_readback_resolve));
  if (readback_mode == ReadbackResolveMode::kDisabled) {
    return true;
  }

  auto ensure_readback_slot = [&](ReadbackBuffer& readback, uint32_t index, uint32_t size) -> bool {
    if (readback.buffers[index] != VK_NULL_HANDLE && size <= readback.sizes[index] &&
        readback.mapped_data[index] != nullptr) {
      return true;
    }

    VkBuffer new_buffer = VK_NULL_HANDLE;
    VkDeviceMemory new_memory = VK_NULL_HANDLE;
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
            vulkan_device, size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            ui::vulkan::util::MemoryPurpose::kReadback, new_buffer, new_memory)) {
      REXGPU_ERROR("Failed to create a {} MB Vulkan resolve readback buffer", size >> 20);
      return false;
    }

    void* new_mapping = nullptr;
    if (dfn.vkMapMemory(device, new_memory, 0, VK_WHOLE_SIZE, 0, &new_mapping) != VK_SUCCESS) {
      REXGPU_ERROR("Failed to map a Vulkan resolve readback buffer");
      dfn.vkDestroyBuffer(device, new_buffer, nullptr);
      dfn.vkFreeMemory(device, new_memory, nullptr);
      return false;
    }

    if (readback.buffers[index] != VK_NULL_HANDLE || readback.memories[index] != VK_NULL_HANDLE) {
      if (!AwaitAllQueueOperationsCompletion()) {
        dfn.vkUnmapMemory(device, new_memory);
        dfn.vkDestroyBuffer(device, new_buffer, nullptr);
        dfn.vkFreeMemory(device, new_memory, nullptr);
        return false;
      }
      if (readback.mapped_data[index] != nullptr && readback.memories[index] != VK_NULL_HANDLE) {
        dfn.vkUnmapMemory(device, readback.memories[index]);
      }
      if (readback.buffers[index] != VK_NULL_HANDLE) {
        dfn.vkDestroyBuffer(device, readback.buffers[index], nullptr);
      }
      if (readback.memories[index] != VK_NULL_HANDLE) {
        dfn.vkFreeMemory(device, readback.memories[index], nullptr);
      }
    }

    readback.buffers[index] = new_buffer;
    readback.memories[index] = new_memory;
    readback.mapped_data[index] = new_mapping;
    readback.sizes[index] = size;
    return true;
  };

  bool is_scaled = texture_cache_->IsDrawResolutionScaled();
  uint64_t resolve_key = MakeReadbackResolveKey(written_address, written_length);
  ReadbackBuffer& readback = readback_buffers_[resolve_key];
  readback.last_used_frame = frame_current_;
  uint32_t write_index = readback.current_index;
  uint32_t readback_size = AlignReadbackBufferSize(written_length);
  if (!ensure_readback_slot(readback, write_index, readback_size)) {
    return true;
  }

  if (is_scaled) {
    if (!resolve_downscale_pipeline_ || !resolve_downscale_pipeline_layout_) {
      return true;
    }

    reg::RB_COPY_DEST_INFO copy_dest_info = register_file_->Get<reg::RB_COPY_DEST_INFO>();
    const FormatInfo* format_info = FormatInfo::Get(uint32_t(copy_dest_info.copy_dest_format));
    uint32_t bits_per_pixel = format_info->bits_per_pixel;
    if (bits_per_pixel != 8 && bits_per_pixel != 16 && bits_per_pixel != 32 &&
        bits_per_pixel != 64) {
      return true;
    }

    uint32_t pixel_size_log2;
    if (!rex::bit_scan_forward(bits_per_pixel >> 3, &pixel_size_log2)) {
      return true;
    }
    uint32_t tile_size_1x = 32 * 32 * (uint32_t(1) << pixel_size_log2);
    uint32_t tile_count = written_length / tile_size_1x;
    if (!tile_count) {
      return true;
    }

    uint64_t scaled_start = 0, scaled_length = 0;
    if (!texture_cache_->GetScaledResolveRange(written_address, written_length, 0, scaled_start,
                                               scaled_length)) {
      return true;
    }
    if (!scaled_length) {
      return true;
    }

    VkBuffer scaled_resolve_buffer = texture_cache_->scaled_resolve_buffer();
    if (scaled_resolve_buffer == VK_NULL_HANDLE || scaled_start > uint64_t(UINT32_MAX)) {
      return true;
    }

    uint32_t downscale_buffer_size = AlignReadbackBufferSize(written_length);
    if (downscale_buffer_size > resolve_downscale_buffer_size_) {
      VkBuffer new_buffer = VK_NULL_HANDLE;
      VkDeviceMemory new_memory = VK_NULL_HANDLE;
      if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
              vulkan_device, downscale_buffer_size,
              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
              ui::vulkan::util::MemoryPurpose::kDeviceLocal, new_buffer, new_memory)) {
        REXGPU_ERROR("Failed to create a {} MB Vulkan resolve downscale buffer",
                     downscale_buffer_size >> 20);
        return true;
      }
      if (resolve_downscale_buffer_ != VK_NULL_HANDLE ||
          resolve_downscale_buffer_memory_ != VK_NULL_HANDLE) {
        if (!AwaitAllQueueOperationsCompletion()) {
          dfn.vkDestroyBuffer(device, new_buffer, nullptr);
          dfn.vkFreeMemory(device, new_memory, nullptr);
          return true;
        }
        if (resolve_downscale_buffer_ != VK_NULL_HANDLE) {
          dfn.vkDestroyBuffer(device, resolve_downscale_buffer_, nullptr);
        }
        if (resolve_downscale_buffer_memory_ != VK_NULL_HANDLE) {
          dfn.vkFreeMemory(device, resolve_downscale_buffer_memory_, nullptr);
        }
      }
      resolve_downscale_buffer_ = new_buffer;
      resolve_downscale_buffer_memory_ = new_memory;
      resolve_downscale_buffer_size_ = downscale_buffer_size;
    }
    if (resolve_downscale_buffer_ == VK_NULL_HANDLE) {
      return true;
    }

    VkDescriptorSet descriptor_set = AllocateSingleTransientDescriptor(
        SingleTransientDescriptorLayout::kStorageBufferPairCompute);
    if (descriptor_set == VK_NULL_HANDLE) {
      return true;
    }

    VkDescriptorBufferInfo buffer_infos[2] = {};
    buffer_infos[0].buffer = scaled_resolve_buffer;
    buffer_infos[0].offset = 0;
    buffer_infos[0].range = VK_WHOLE_SIZE;
    buffer_infos[1].buffer = resolve_downscale_buffer_;
    buffer_infos[1].offset = 0;
    buffer_infos[1].range = written_length;

    VkWriteDescriptorSet descriptor_writes[2] = {};
    for (uint32_t i = 0; i < 2; ++i) {
      descriptor_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
      descriptor_writes[i].dstSet = descriptor_set;
      descriptor_writes[i].dstBinding = i;
      descriptor_writes[i].descriptorCount = 1;
      descriptor_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
      descriptor_writes[i].pBufferInfo = &buffer_infos[i];
    }
    dfn.vkUpdateDescriptorSets(device, 2, descriptor_writes, 0, nullptr);

    texture_cache_->UseScaledResolveBufferForRead();
    SubmitBarriers(true);

    VkBufferMemoryBarrier pre_barrier = {};
    pre_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    pre_barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    pre_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    pre_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    pre_barrier.buffer = resolve_downscale_buffer_;
    pre_barrier.offset = 0;
    pre_barrier.size = written_length;
    deferred_command_buffer_.CmdVkPipelineBarrier(
        VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 1, &pre_barrier, 0, nullptr);

    BindExternalComputePipeline(resolve_downscale_pipeline_);

    ResolveDownscaleConstants constants;
    constants.scale_x = texture_cache_->draw_resolution_scale_x();
    constants.scale_y = texture_cache_->draw_resolution_scale_y();
    constants.pixel_size_log2 = pixel_size_log2;
    constants.tile_count = tile_count;
    constants.source_offset_bytes = uint32_t(scaled_start);
    constants.half_pixel_offset = (REXCVAR_GET(readback_resolve_half_pixel_offset) &&
                                   (constants.scale_x > 1 || constants.scale_y > 1))
                                      ? 1u
                                      : 0u;
    deferred_command_buffer_.CmdVkPushConstants(resolve_downscale_pipeline_layout_,
                                                VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(constants),
                                                &constants);
    deferred_command_buffer_.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE,
                                                     resolve_downscale_pipeline_layout_, 0, 1,
                                                     &descriptor_set, 0, nullptr);
    uint32_t downscale_scope = BeginScopeProfile(
        ScopeCategory::kCompute, tile_count * 32, 32, "DOWNSCALE",
        VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
        VK_ATTACHMENT_LOAD_OP_DONT_CARE, VK_ATTACHMENT_STORE_OP_DONT_CARE,
        "resolve_downscale");
    deferred_command_buffer_.CmdVkDispatch(tile_count, 1, 1, WorkPurpose::kDownscale);
    EndScopeProfile(downscale_scope);

    VkBufferMemoryBarrier downscale_barrier = {};
    downscale_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    downscale_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    downscale_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    downscale_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    downscale_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    downscale_barrier.buffer = resolve_downscale_buffer_;
    downscale_barrier.offset = 0;
    downscale_barrier.size = written_length;
    deferred_command_buffer_.CmdVkPipelineBarrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                                  VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1,
                                                  &downscale_barrier, 0, nullptr);

    VkBufferCopy readback_region = {};
    readback_region.srcOffset = 0;
    readback_region.dstOffset = 0;
    readback_region.size = written_length;
    deferred_command_buffer_.CmdVkCopyBuffer(resolve_downscale_buffer_,
                                             readback.buffers[write_index], 1, &readback_region);
  } else {
    shared_memory_->Use(VulkanSharedMemory::Usage::kRead);
    SubmitBarriers(true);

    VkBufferCopy readback_region = {};
    readback_region.srcOffset = written_address;
    readback_region.dstOffset = 0;
    readback_region.size = written_length;
    deferred_command_buffer_.CmdVkCopyBuffer(shared_memory_->buffer(),
                                             readback.buffers[write_index], 1, &readback_region);
  }

  PushBufferMemoryBarrier(readback.buffers[write_index], 0, written_length,
                          VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                          VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);

  bool use_delayed_sync =
      readback_mode == ReadbackResolveMode::kFast || readback_mode == ReadbackResolveMode::kSome;
  uint32_t read_index = write_index;
  if (use_delayed_sync) {
    read_index = 1 - write_index;
  } else if (!AwaitAllQueueOperationsCompletion()) {
    return true;
  }

  bool is_cache_miss = false;
  if (use_delayed_sync && (readback.buffers[read_index] == VK_NULL_HANDLE ||
                           written_length > readback.sizes[read_index] ||
                           readback.mapped_data[read_index] == nullptr)) {
    is_cache_miss = true;
    read_index = write_index;
    if (!AwaitAllQueueOperationsCompletion()) {
      return true;
    }
  }

  bool should_copy = (readback_mode == ReadbackResolveMode::kSome) ? is_cache_miss : true;
  if (should_copy && readback.buffers[read_index] != VK_NULL_HANDLE &&
      written_length <= readback.sizes[read_index] && readback.mapped_data[read_index] != nullptr) {
    VkMappedMemoryRange readback_memory_range = {};
    readback_memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    readback_memory_range.memory = readback.memories[read_index];
    readback_memory_range.offset = 0;
    readback_memory_range.size = VK_WHOLE_SIZE;
    dfn.vkInvalidateMappedMemoryRanges(device, 1, &readback_memory_range);

    if (face_resolve_probe &&
        (frame_current_ == 8680 || written_address == UINT32_C(0x0DE20000))) {
      static uint32_t probe_sequence = 0;
      const std::string stem = fmt::format(
          "/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/resolve-probe/resolve_{:03}_0x{:08X}",
          ++probe_sequence, written_address);
      std::ofstream bytes(stem + ".raw", std::ios::binary);
      bytes.write(reinterpret_cast<const char*>(readback.mapped_data[read_index]), written_length);
      const auto pitch = register_file_->Get<reg::RB_COPY_DEST_PITCH>();
      const auto info = register_file_->Get<reg::RB_COPY_DEST_INFO>();
      REXGPU_INFO("[DEBUG-face-resolve] frame={} sequence={} address=0x{:08X} length={} pitch={} height={} format={} endian={} current_slot={} scaled={} file={}",
          frame_current_, probe_sequence, written_address, written_length,
          uint32_t(pitch.copy_dest_pitch), uint32_t(pitch.copy_dest_height), uint32_t(info.copy_dest_format),
          uint32_t(info.copy_dest_endian), read_index, is_scaled, stem);
      // Diagnostic readback must not upload stale/current bytes into guest memory.
    } else {
      uint8_t* destination = memory_->TranslatePhysical(written_address);
      if (destination) {
        std::memcpy(destination, readback.mapped_data[read_index], written_length);
      }
    }
  }

  readback.current_index = 1 - readback.current_index;
  return true;
}

void VulkanCommandProcessor::EvictOldReadbackBuffers(
    std::unordered_map<uint64_t, ReadbackBuffer>& buffer_map) {
  if (buffer_map.empty()) {
    return;
  }

  const uint64_t eviction_frame_floor = (frame_current_ > kReadbackBufferEvictionAgeFrames)
                                            ? (frame_current_ - kReadbackBufferEvictionAgeFrames)
                                            : 0;
  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  for (auto it = buffer_map.begin(); it != buffer_map.end();) {
    ReadbackBuffer& readback = it->second;
    bool evict =
        buffer_map.size() > kMaxReadbackBuffers || readback.last_used_frame < eviction_frame_floor;
    if (!evict) {
      ++it;
      continue;
    }
    for (uint32_t i = 0; i < 2; ++i) {
      if (readback.mapped_data[i] && readback.memories[i] != VK_NULL_HANDLE) {
        dfn.vkUnmapMemory(device, readback.memories[i]);
      }
      if (readback.buffers[i] != VK_NULL_HANDLE) {
        dfn.vkDestroyBuffer(device, readback.buffers[i], nullptr);
      }
      if (readback.memories[i] != VK_NULL_HANDLE) {
        dfn.vkFreeMemory(device, readback.memories[i], nullptr);
      }
      readback.buffers[i] = VK_NULL_HANDLE;
      readback.memories[i] = VK_NULL_HANDLE;
      readback.mapped_data[i] = nullptr;
      readback.sizes[i] = 0;
      readback.submission_written[i] = 0;
      readback.written_size[i] = 0;
    }
    it = buffer_map.erase(it);
  }
}

bool VulkanCommandProcessor::InitializeOcclusionQueryResources() {
  active_occlusion_query_ = {};
  occlusion_query_cursor_ = 0;
  occlusion_query_resources_available_ = false;
  occlusion_query_pool_ = VK_NULL_HANDLE;
  occlusion_query_readback_buffer_ = VK_NULL_HANDLE;
  occlusion_query_readback_memory_ = VK_NULL_HANDLE;
  occlusion_query_readback_memory_type_ = UINT32_MAX;
  occlusion_query_readback_memory_size_ = 0;
  occlusion_query_readback_mapping_ = nullptr;

  const ui::vulkan::VulkanDevice* vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  VkDevice device = vulkan_device->device();

  VkQueryPoolCreateInfo pool_info = {};
  pool_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
  pool_info.queryType = VK_QUERY_TYPE_OCCLUSION;
  pool_info.queryCount = kMaxOcclusionQueries;
  if (dfn.vkCreateQueryPool(device, &pool_info, nullptr, &occlusion_query_pool_) != VK_SUCCESS) {
    REXGPU_WARN(
        "VulkanCommandProcessor: Failed to create occlusion query pool, using fake sample counts");
    return false;
  }

  if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
          vulkan_device, sizeof(uint64_t) * kMaxOcclusionQueries, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
          ui::vulkan::util::MemoryPurpose::kReadback, occlusion_query_readback_buffer_,
          occlusion_query_readback_memory_, &occlusion_query_readback_memory_type_,
          &occlusion_query_readback_memory_size_)) {
    REXGPU_WARN(
        "VulkanCommandProcessor: Failed to create occlusion query readback buffer, using fake "
        "sample counts");
    ShutdownOcclusionQueryResources();
    return false;
  }

  if (dfn.vkMapMemory(device, occlusion_query_readback_memory_, 0, VK_WHOLE_SIZE, 0,
                      reinterpret_cast<void**>(&occlusion_query_readback_mapping_)) != VK_SUCCESS) {
    REXGPU_WARN(
        "VulkanCommandProcessor: Failed to map occlusion query readback buffer, using fake sample "
        "counts");
    ShutdownOcclusionQueryResources();
    return false;
  }

  occlusion_query_resources_available_ = true;
  return true;
}

void VulkanCommandProcessor::ShutdownOcclusionQueryResources() {
  DisableHostOcclusionQueries();

  const ui::vulkan::VulkanDevice* vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  VkDevice device = vulkan_device->device();

  if (occlusion_query_readback_mapping_ && occlusion_query_readback_memory_ != VK_NULL_HANDLE) {
    dfn.vkUnmapMemory(device, occlusion_query_readback_memory_);
  }
  occlusion_query_readback_mapping_ = nullptr;
  if (occlusion_query_readback_buffer_ != VK_NULL_HANDLE) {
    dfn.vkDestroyBuffer(device, occlusion_query_readback_buffer_, nullptr);
  }
  if (occlusion_query_readback_memory_ != VK_NULL_HANDLE) {
    dfn.vkFreeMemory(device, occlusion_query_readback_memory_, nullptr);
  }
  if (occlusion_query_pool_ != VK_NULL_HANDLE) {
    dfn.vkDestroyQueryPool(device, occlusion_query_pool_, nullptr);
  }

  occlusion_query_pool_ = VK_NULL_HANDLE;
  occlusion_query_readback_buffer_ = VK_NULL_HANDLE;
  occlusion_query_readback_memory_ = VK_NULL_HANDLE;
  occlusion_query_readback_memory_type_ = UINT32_MAX;
  occlusion_query_readback_memory_size_ = 0;
}

bool VulkanCommandProcessor::AcquireOcclusionQueryIndex(uint32_t& host_index_out) {
  if (occlusion_query_cursor_ >= kMaxOcclusionQueries) {
    occlusion_query_cursor_ = 0;
  }
  host_index_out = occlusion_query_cursor_++;
  return true;
}

void VulkanCommandProcessor::DisableHostOcclusionQueries() {
  if (active_occlusion_query_.valid && occlusion_query_pool_ != VK_NULL_HANDLE) {
    if (BeginSubmission(true)) {
      deferred_command_buffer_.CmdVkEndQuery(occlusion_query_pool_,
                                             active_occlusion_query_.host_index);
      EndSubmission(false);
    }
  }
  active_occlusion_query_ = {};
  occlusion_query_cursor_ = 0;
  occlusion_query_resources_available_ = false;
}

bool VulkanCommandProcessor::BeginGuestOcclusionQuery(uint32_t sample_count_address) {
  if (!REXCVAR_GET(occlusion_query_enable) || !occlusion_query_resources_available_ ||
      occlusion_query_pool_ == VK_NULL_HANDLE || occlusion_query_readback_mapping_ == nullptr) {
    return false;
  }
  if (active_occlusion_query_.valid) {
    REXGPU_WARN(
        "VulkanCommandProcessor: Occlusion query begin issued while another query is active");
    DisableHostOcclusionQueries();
    return false;
  }

  uint32_t host_index = 0;
  if (!AcquireOcclusionQueryIndex(host_index)) {
    return false;
  }
  if (!BeginSubmission(true)) {
    return false;
  }

  EndRenderPass();

  DeferredCommandBuffer& command_buffer = deferred_command_buffer();
  command_buffer.CmdVkResetQueryPool(occlusion_query_pool_, host_index, 1);
  command_buffer.CmdVkBeginQuery(occlusion_query_pool_, host_index, 0);
  active_occlusion_query_.sample_count_address = sample_count_address;
  active_occlusion_query_.host_index = host_index;
  active_occlusion_query_.valid = true;
  return true;
}

bool VulkanCommandProcessor::EndGuestOcclusionQuery(uint32_t sample_count_address) {
  if (!REXCVAR_GET(occlusion_query_enable) || !occlusion_query_resources_available_ ||
      !active_occlusion_query_.valid || occlusion_query_pool_ == VK_NULL_HANDLE ||
      occlusion_query_readback_mapping_ == nullptr) {
    return false;
  }

  uint32_t host_index = active_occlusion_query_.host_index;
  active_occlusion_query_ = {};

  if (!BeginSubmission(true)) {
    return false;
  }

  EndRenderPass();

  DeferredCommandBuffer& command_buffer = deferred_command_buffer();
  command_buffer.CmdVkEndQuery(occlusion_query_pool_, host_index);
  command_buffer.CmdVkCopyQueryPoolResults(occlusion_query_pool_, host_index, 1,
                                           occlusion_query_readback_buffer_,
                                           sizeof(uint64_t) * host_index, sizeof(uint64_t),
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

  if (!EndSubmission(false)) {
    return false;
  }
  const bool audit_wait = REXCVAR_GET(gpu_frame_audit_enable);
  auto audit_wait_start = audit_wait ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  bool wait_completed = AwaitAllQueueOperationsCompletion();
  if (audit_wait) {
    frame_cpu_profiling_stats_.time_occlusion_wait += std::chrono::steady_clock::now() - audit_wait_start;
    ++frame_cpu_profiling_stats_.occlusion_wait_count;
  }
  if (!wait_completed) {
    return false;
  }

  const ui::vulkan::VulkanDevice* vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  if (!(vulkan_device->memory_types().host_coherent &
        (uint32_t(1) << occlusion_query_readback_memory_type_))) {
    VkMappedMemoryRange memory_range = {};
    memory_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    memory_range.memory = occlusion_query_readback_memory_;
    memory_range.offset = 0;
    memory_range.size =
        std::min(rex::round_up(VkDeviceSize(sizeof(uint64_t) * kMaxOcclusionQueries),
                               vulkan_device->properties().nonCoherentAtomSize),
                 occlusion_query_readback_memory_size_);
    dfn.vkInvalidateMappedMemoryRanges(device, 1, &memory_range);
  }

  const uint64_t* results = reinterpret_cast<const uint64_t*>(occlusion_query_readback_mapping_);
  uint64_t samples = NormalizeOcclusionSamples(results[host_index]);
  WriteGuestOcclusionResult(sample_count_address, samples);
  return true;
}

uint64_t VulkanCommandProcessor::NormalizeOcclusionSamples(uint64_t samples) const {
  if (samples == 0 || !texture_cache_) {
    return samples;
  }
  uint64_t scale_x = texture_cache_->draw_resolution_scale_x();
  uint64_t scale_y = texture_cache_->draw_resolution_scale_y();
  uint64_t scale = scale_x * scale_y;
  if (scale <= 1) {
    return samples;
  }
  return (samples + (scale >> 1)) / scale;
}

void VulkanCommandProcessor::WriteGuestOcclusionResult(uint32_t sample_count_address,
                                                       uint64_t samples) {
  auto* sample_counts =
      memory_->TranslatePhysical<xenos::xe_gpu_depth_sample_counts*>(sample_count_address);
  if (!sample_counts) {
    return;
  }
  uint32_t clamped = samples > uint64_t(UINT32_MAX) ? UINT32_MAX : uint32_t(samples);
  sample_counts->Total_A = clamped;
  sample_counts->Total_B = 0;
  sample_counts->ZPass_A = clamped;
  sample_counts->ZPass_B = 0;
  sample_counts->ZFail_A = 0;
  sample_counts->ZFail_B = 0;
  sample_counts->StencilFail_A = 0;
  sample_counts->StencilFail_B = 0;
}

void VulkanCommandProcessor::CheckSubmissionFenceAndDeviceLoss(uint64_t await_submission) {
  // Only report once, no need to retry a wait that won't succeed anyway.
  if (device_lost_) {
    return;
  }

  if (await_submission >= GetCurrentSubmission()) {
    if (submission_open_) {
      EndSubmission(false);
    }
    // A submission won't be ended if it hasn't been started, or if ending
    // has failed - clamp the index.
    await_submission = GetCurrentSubmission() - 1;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  size_t fences_total = submissions_in_flight_fences_.size();
  size_t fences_awaited = 0;
  if (await_submission > submission_completed_) {
    // Await in a blocking way if requested.
    // TODO(Triang3l): Await only one fence. "Fence signal operations that are
    // defined by vkQueueSubmit additionally include in the first
    // synchronization scope all commands that occur earlier in submission
    // order."
    // FrameCpuProfilingStats is accumulated until its 120-frame log point, so
    // include every blocking wait in that interval rather than sampling only
    // the wait that happens to land on the log frame.
    bool profile_wait = REXCVAR_GET(gpu_profile_cpu) || REXCVAR_GET(gpu_frame_audit_enable);
    auto wait_start = profile_wait ? std::chrono::steady_clock::now()
                                   : std::chrono::steady_clock::time_point{};
    FrameCpuTimeline::Scope audit_fence(GetFrameCpuTimeline(), FrameCpuTimeline::Bucket::kFence);
    VkResult wait_result =
        dfn.vkWaitForFences(device, uint32_t(await_submission - submission_completed_),
                            submissions_in_flight_fences_.data(), VK_TRUE, UINT64_MAX);
    if (profile_wait) {
      auto wait_duration = std::chrono::steady_clock::now() - wait_start;
      frame_cpu_profiling_stats_.fence_wait_count++;
      frame_cpu_profiling_stats_.time_fence_waits += wait_duration;
      interval_cpu_profiling_stats_.total_fence_wait_count++;
      interval_cpu_profiling_stats_.time_fence_waits += wait_duration;
    }
    if (wait_result == VK_SUCCESS) {
      fences_awaited += await_submission - submission_completed_;
    } else {
      REXGPU_ERROR("Failed to await submission completion Vulkan fences");
      if (wait_result == VK_ERROR_DEVICE_LOST) {
        device_lost_ = true;
      }
    }
  }
  // Check how far into the submissions the GPU currently is, in order because
  // submission themselves can be executed out of order, but Xenia serializes
  // that for simplicity.
  while (fences_awaited < fences_total) {
    VkResult fence_status =
        dfn.vkWaitForFences(device, 1, &submissions_in_flight_fences_[fences_awaited], VK_TRUE, 0);
    if (fence_status != VK_SUCCESS) {
      if (fence_status == VK_ERROR_DEVICE_LOST) {
        device_lost_ = true;
      }
      break;
    }
    ++fences_awaited;
  }
  if (device_lost_) {
    if (graphics_system_) {
      graphics_system_->OnHostGpuLossFromAnyThread(true);
    }
    return;
  }
  if (!fences_awaited) {
    // Not updated - no need to reclaim or download things.
    return;
  }
  // Reclaim fences.
  fences_free_.reserve(fences_free_.size() + fences_awaited);
  auto submissions_in_flight_fences_awaited_end = submissions_in_flight_fences_.cbegin();
  std::advance(submissions_in_flight_fences_awaited_end, fences_awaited);
  fences_free_.insert(fences_free_.cend(), submissions_in_flight_fences_.cbegin(),
                      submissions_in_flight_fences_awaited_end);
  submissions_in_flight_fences_.erase(submissions_in_flight_fences_.cbegin(),
                                      submissions_in_flight_fences_awaited_end);
  submission_completed_ += fences_awaited;

  // Reclaim semaphores.
  while (!submissions_in_flight_semaphores_.empty()) {
    const auto& semaphore_submission = submissions_in_flight_semaphores_.front();
    if (semaphore_submission.first > submission_completed_) {
      break;
    }
    semaphores_free_.push_back(semaphore_submission.second);
    submissions_in_flight_semaphores_.pop_front();
  }

  // Reclaim command pools.
  while (!command_buffers_submitted_.empty()) {
    const auto& command_buffer_pair = command_buffers_submitted_.front();
    if (command_buffer_pair.first > submission_completed_) {
      break;
    }
    command_buffers_writable_.push_back(command_buffer_pair.second);
    command_buffers_submitted_.pop_front();
  }

  shared_memory_->CompletedSubmissionUpdated();

  primitive_processor_->CompletedSubmissionUpdated();

  render_target_cache_->CompletedSubmissionUpdated();

  texture_cache_->CompletedSubmissionUpdated(submission_completed_);

  // Destroy objects scheduled for destruction.
  while (!destroy_framebuffers_.empty()) {
    const auto& destroy_pair = destroy_framebuffers_.front();
    if (destroy_pair.first > submission_completed_) {
      break;
    }
    dfn.vkDestroyFramebuffer(device, destroy_pair.second, nullptr);
    destroy_framebuffers_.pop_front();
  }
  while (!destroy_image_views_.empty()) {
    const auto& destroy_pair = destroy_image_views_.front();
    if (destroy_pair.first > submission_completed_) {
      break;
    }
    dfn.vkDestroyImageView(device, destroy_pair.second, nullptr);
    destroy_image_views_.pop_front();
  }
  while (!destroy_buffers_.empty()) {
    const auto& destroy_pair = destroy_buffers_.front();
    if (destroy_pair.first > submission_completed_) {
      break;
    }
    dfn.vkDestroyBuffer(device, destroy_pair.second, nullptr);
    destroy_buffers_.pop_front();
  }
  while (!destroy_images_.empty()) {
    const auto& destroy_pair = destroy_images_.front();
    if (destroy_pair.first > submission_completed_) {
      break;
    }
    dfn.vkDestroyImage(device, destroy_pair.second, nullptr);
    destroy_images_.pop_front();
  }
  while (!destroy_memory_.empty()) {
    const auto& destroy_pair = destroy_memory_.front();
    if (destroy_pair.first > submission_completed_) {
      break;
    }
    dfn.vkFreeMemory(device, destroy_pair.second, nullptr);
    destroy_memory_.pop_front();
  }

  FlushCompletedAuditRows(false);
}

bool VulkanCommandProcessor::CanEndSubmissionImmediately() const {
  return !submission_open_ || (!scratch_buffer_used_ && !pipeline_cache_->IsCreatingPipelines());
}

bool VulkanCommandProcessor::BeginSubmission(bool is_guest_command) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  if (device_lost_) {
    return false;
  }

  bool is_opening_frame = is_guest_command && !frame_open_;
  if (submission_open_ && !is_opening_frame) {
    return true;
  }

  // Check the fence - needed for all kinds of submissions (to reclaim transient
  // resources early) and specifically for frames (not to queue too many), and
  // await the availability of the current frame. Also check whether the device
  // is still available, and whether the await was successful.
  uint64_t await_submission =
      is_opening_frame ? closed_frame_submissions_[frame_current_ % kMaxFramesInFlight] : 0;
  auto opening_wait_before = frame_cpu_profiling_stats_.time_fence_waits;
  auto opening_wait_count_before = frame_cpu_profiling_stats_.fence_wait_count;
  CheckSubmissionFenceAndDeviceLoss(await_submission);
  if (is_opening_frame && REXCVAR_GET(gpu_frame_audit_enable)) {
    frame_cpu_profiling_stats_.time_pre_wall_fence +=
        frame_cpu_profiling_stats_.time_fence_waits - opening_wait_before;
    frame_cpu_profiling_stats_.pre_wall_fence_count +=
        frame_cpu_profiling_stats_.fence_wait_count - opening_wait_count_before;
  }
  if (device_lost_ || submission_completed_ < await_submission) {
    return false;
  }

  if (is_opening_frame) {
    // Update the completed frame index, also obtaining the actual completed
    // frame number (since the CPU may be actually less than 3 frames behind)
    // before reclaiming resources tracked with the frame number.
    frame_completed_ = std::max(frame_current_, uint64_t(kMaxFramesInFlight)) - kMaxFramesInFlight;
    for (uint64_t frame = frame_completed_ + 1; frame < frame_current_; ++frame) {
      if (closed_frame_submissions_[frame % kMaxFramesInFlight] > submission_completed_) {
        break;
      }
      frame_completed_ = frame;
    }
  }

  if (!submission_open_) {
    submission_open_ = true;

    // Start a new deferred command buffer - will submit it to the real one in
    // the end of the submission (when async pipeline object creation requests
    // are fulfilled).
    deferred_command_buffer_.Reset();
    prologue_deferred_command_buffer_.Reset();
    submission_read_ranges_.clear();
    hoisted_uploads_pending_ = false;

    // Reset cached state of the command buffer.
    dynamic_viewport_update_needed_ = true;
    dynamic_scissor_update_needed_ = true;
    dynamic_depth_bias_update_needed_ = true;
    dynamic_blend_constants_update_needed_ = true;
    dynamic_stencil_compare_mask_front_update_needed_ = true;
    dynamic_stencil_compare_mask_back_update_needed_ = true;
    dynamic_stencil_write_mask_front_update_needed_ = true;
    dynamic_stencil_write_mask_back_update_needed_ = true;
    dynamic_stencil_reference_front_update_needed_ = true;
    dynamic_stencil_reference_back_update_needed_ = true;
    current_render_pass_ = VK_NULL_HANDLE;
    current_framebuffer_ = nullptr;
    in_render_pass_ = false;
    current_dynamic_rendering_is_transfer_ = false;
    current_dynamic_rendering_has_depth_ = false;
    current_dynamic_rendering_has_stencil_ = false;
    current_dynamic_rendering_color_count_ = 0;
    current_dynamic_rendering_transfer_dest_view_ = VK_NULL_HANDLE;
    current_dynamic_rendering_transfer_depth_view_ = VK_NULL_HANDLE;
    current_dynamic_rendering_transfer_dest_is_depth_ = false;
    current_dynamic_rendering_render_pass_key_ = {};
    current_dynamic_rendering_area_ = {};
    current_guest_graphics_pipeline_ = VK_NULL_HANDLE;
    current_external_graphics_pipeline_ = VK_NULL_HANDLE;
    current_external_compute_pipeline_ = VK_NULL_HANDLE;
    current_guest_graphics_pipeline_layout_ = nullptr;
    current_graphics_descriptor_sets_bound_up_to_date_ = 0;
    bound_index_buffer_ = VK_NULL_HANDLE;
    bound_index_buffer_offset_ = 0;
    bound_index_type_ = VK_INDEX_TYPE_MAX_ENUM;

    primitive_processor_->BeginSubmission();

    texture_cache_->BeginSubmission(GetCurrentSubmission());
  }

  if (is_opening_frame) {
    frame_open_ = true;
    frame_submission_counts_[frame_current_ % kMaxTimestampFramesInFlight] = 0;
    transfer_profile_counts_[frame_current_ % kMaxTimestampFramesInFlight] = 0;
    transfer_profile_reset_count_ = 0;
    uint32_t scope_slot = frame_current_ % kMaxTimestampFramesInFlight;
    scope_profile_counts_[scope_slot] = 0;
    scope_profile_reset_count_ = 0;
    frame_scopes_[scope_slot].clear();
    active_scope_index_ = UINT32_MAX;
    work_purpose_audit_.Reset();
    frame_start_time_ = std::chrono::steady_clock::now();
    if (REXCVAR_GET(gpu_frame_audit_enable) && REXCVAR_GET(gpu_frame_timeline_enable))
      GetFrameCpuTimeline().Start(frame_start_time_);
    frame_work_counters_.Reset();
    frame_pass_stats_.Reset();
    current_frame_barrier_breaks_shmem_ = 0;
    current_frame_barrier_breaks_texture_ = 0;
    current_frame_barrier_breaks_rt_ = 0;
    current_frame_barrier_breaks_prim_ = 0;
    current_frame_barrier_breaks_other_ = 0;
    current_frame_shmem_hoist_fallbacks_ = 0;
    current_frame_shmem_hoist_successes_ = 0;
    interval_cpu_profiling_stats_.frame_count++;
    frame_used_async_placeholder_pipeline_ = false;
    if (IsGpuProvenanceFrame(frame_current_)) {
      gpu_provenance_draw_index_ = 0;
    }

    // Reset bindings that depend on transient data.
    std::memset(current_float_constant_map_vertex_, 0, sizeof(current_float_constant_map_vertex_));
    std::memset(current_float_constant_map_pixel_, 0, sizeof(current_float_constant_map_pixel_));
    std::memset(current_graphics_descriptor_sets_, 0, sizeof(current_graphics_descriptor_sets_));
    current_constant_buffers_up_to_date_ = 0;
    constant_ring_frame_index_ = uint32_t(frame_current_ % kMaxFramesInFlight);
    constant_ring_frame_base_ = constant_ring_frame_index_ * kConstantRingFramePartitionSize;
    constant_ring_offset_ = constant_ring_frame_base_;
    current_graphics_descriptor_sets_[SpirvShaderTranslator::kDescriptorSetSharedMemoryAndEdram] =
        shared_memory_and_edram_descriptor_set_;
    current_graphics_descriptor_sets_[SpirvShaderTranslator::kDescriptorSetConstants] =
        constant_descriptor_sets_[constant_ring_frame_index_];
    current_graphics_descriptor_set_values_up_to_date_ =
        (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetSharedMemoryAndEdram) |
        (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetConstants);
    bound_textures_vertex_count_ = UINT32_MAX;
    bound_samplers_vertex_count_ = UINT32_MAX;
    bound_textures_pixel_count_ = UINT32_MAX;
    bound_samplers_pixel_count_ = UINT32_MAX;
    last_bound_constant_buffer_dynamic_offsets_.fill(UINT32_MAX);
    validated_shared_memory_ranges_count_ = 0;
    validated_shared_memory_ranges_epoch_ = 0;

    // Reclaim pool pages - no need to do this every small submission since some
    // may be reused.
    // FIXME(Triang3l): This will result in a memory leak if the guest is not
    // presenting.
    uniform_buffer_pool_->Reclaim(frame_completed_);
    while (!single_transient_descriptors_used_.empty()) {
      const UsedSingleTransientDescriptor& used_transient_descriptor =
          single_transient_descriptors_used_.front();
      if (used_transient_descriptor.frame > frame_completed_) {
        break;
      }
      single_transient_descriptors_free_[size_t(used_transient_descriptor.layout)].push_back(
          used_transient_descriptor.set);
      single_transient_descriptors_used_.pop_front();
    }
    while (!constants_transient_descriptors_used_.empty()) {
      const std::pair<uint64_t, VkDescriptorSet>& used_transient_descriptor =
          constants_transient_descriptors_used_.front();
      if (used_transient_descriptor.first > frame_completed_) {
        break;
      }
      constants_transient_descriptors_free_.push_back(used_transient_descriptor.second);
      constants_transient_descriptors_used_.pop_front();
    }
    while (!texture_transient_descriptor_sets_used_.empty()) {
      const UsedTextureTransientDescriptorSet& used_transient_descriptor_set =
          texture_transient_descriptor_sets_used_.front();
      if (used_transient_descriptor_set.frame > frame_completed_) {
        break;
      }
      auto it = texture_transient_descriptor_sets_free_.find(used_transient_descriptor_set.layout);
      if (it == texture_transient_descriptor_sets_free_.end()) {
        it = texture_transient_descriptor_sets_free_
                 .emplace(std::piecewise_construct,
                          std::forward_as_tuple(used_transient_descriptor_set.layout),
                          std::forward_as_tuple())
                 .first;
      }
      it->second.push_back(used_transient_descriptor_set.set);
      texture_transient_descriptor_sets_used_.pop_front();
    }

    EvictOldReadbackBuffers(readback_buffers_);
    EvictOldReadbackBuffers(memexport_readback_buffers_);

    primitive_processor_->BeginFrame();

    texture_cache_->BeginFrame();
  }

  return true;
}

bool VulkanCommandProcessor::EndSubmission(bool is_swap) {
  FrameCpuTimeline::Scope audit_end_submission(GetFrameCpuTimeline(), FrameCpuTimeline::Bucket::kOther);
  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  // Make sure everything needed for submitting exist.
  if (submission_open_) {
    if (fences_free_.empty()) {
      VkFenceCreateInfo fence_create_info;
      fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
      fence_create_info.pNext = nullptr;
      fence_create_info.flags = 0;
      VkFence fence;
      if (dfn.vkCreateFence(device, &fence_create_info, nullptr, &fence) != VK_SUCCESS) {
        REXGPU_ERROR("Failed to create a Vulkan fence");
        // Try to submit later. Completely dropping the submission is not
        // permitted because resources would be left in an undefined state.
        return false;
      }
      fences_free_.push_back(fence);
    }
    if (!sparse_memory_binds_.empty() && semaphores_free_.empty()) {
      VkSemaphoreCreateInfo semaphore_create_info;
      semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
      semaphore_create_info.pNext = nullptr;
      semaphore_create_info.flags = 0;
      VkSemaphore semaphore;
      if (dfn.vkCreateSemaphore(device, &semaphore_create_info, nullptr, &semaphore) !=
          VK_SUCCESS) {
        REXGPU_ERROR("Failed to create a Vulkan semaphore");
        return false;
      }
      semaphores_free_.push_back(semaphore);
    }
    if (command_buffers_writable_.empty()) {
      CommandBuffer command_buffer;
      VkCommandPoolCreateInfo command_pool_create_info;
      command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
      command_pool_create_info.pNext = nullptr;
      command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
      command_pool_create_info.queueFamilyIndex = vulkan_device->queue_family_graphics_compute();
      if (dfn.vkCreateCommandPool(device, &command_pool_create_info, nullptr,
                                  &command_buffer.pool) != VK_SUCCESS) {
        REXGPU_ERROR("Failed to create a Vulkan command pool");
        return false;
      }
      VkCommandBufferAllocateInfo command_buffer_allocate_info;
      command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
      command_buffer_allocate_info.pNext = nullptr;
      command_buffer_allocate_info.commandPool = command_buffer.pool;
      command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
      command_buffer_allocate_info.commandBufferCount = 1;
      if (dfn.vkAllocateCommandBuffers(device, &command_buffer_allocate_info,
                                       &command_buffer.buffer) != VK_SUCCESS) {
        REXGPU_ERROR("Failed to allocate a Vulkan command buffer");
        dfn.vkDestroyCommandPool(device, command_buffer.pool, nullptr);
        return false;
      }
      command_buffers_writable_.push_back(command_buffer);
    }
  }

  bool is_closing_frame = is_swap && frame_open_;
  if (is_closing_frame && audit_swap_start_ != std::chrono::steady_clock::time_point{}) {
    frame_cpu_profiling_stats_.time_swap_pre += std::chrono::steady_clock::now() - audit_swap_start_;
    audit_swap_start_ = {};
  }

  if (is_closing_frame && diagnostic_snapshot_enabled_ &&
      frame_current_ == diagnostic_snapshot_request_.frame &&
      diagnostic_snapshot_head_draw_seen_) {
    if (AwaitAllQueueOperationsCompletion()) {
      render_target_cache_->FinishDiagnosticSnapshotCapture(frame_current_);
    } else {
      REXGPU_ERROR("Render target snapshot frame {} did not complete", frame_current_);
    }
    diagnostic_snapshot_enabled_ = false;
  }

  if (is_closing_frame) {
    texture_cache_->EndFrame();

    primitive_processor_->EndFrame();
  }

  if (submission_open_) {
    assert_false(scratch_buffer_used_);

    if (active_occlusion_query_.valid && occlusion_query_pool_ != VK_NULL_HANDLE) {
      deferred_command_buffer_.CmdVkEndQuery(occlusion_query_pool_,
                                             active_occlusion_query_.host_index);
      active_occlusion_query_ = {};
    }

    if (render_target_cache_->HasPendingDeferredClear()) {
      render_target_cache_->FlushPendingDeferredClear();
    }
    if (in_render_pass_ && REXCVAR_GET(gpu_frame_audit_enable)) ++frame_cpu_profiling_stats_.break_submission;
    EndRenderPass();

    pipeline_cache_->EndSubmission();

    render_target_cache_->EndSubmission();

    primitive_processor_->EndSubmission();

    shared_memory_->EndSubmission();

    uniform_buffer_pool_->FlushWrites();

    // Publish the constants written into the ring during this frame. Only a
    // no-op when the upload memory type is host-coherent, as it is on Metal.
    if (constant_ring_mapping_ != nullptr && constant_ring_offset_ > constant_ring_frame_base_) {
      ui::vulkan::util::FlushMappedMemoryRange(
          vulkan_device, constant_ring_buffer_memory_, constant_ring_upload_memory_type_,
          constant_ring_frame_base_, constant_ring_offset_ - constant_ring_frame_base_);
    }

    // Submit sparse binds earlier, before executing the deferred command
    // buffer, to reduce latency.
    if (!sparse_memory_binds_.empty()) {
      sparse_buffer_bind_infos_temp_.clear();
      sparse_buffer_bind_infos_temp_.reserve(sparse_buffer_binds_.size());
      for (const SparseBufferBind& sparse_buffer_bind : sparse_buffer_binds_) {
        VkSparseBufferMemoryBindInfo& sparse_buffer_bind_info =
            sparse_buffer_bind_infos_temp_.emplace_back();
        sparse_buffer_bind_info.buffer = sparse_buffer_bind.buffer;
        sparse_buffer_bind_info.bindCount = sparse_buffer_bind.bind_count;
        sparse_buffer_bind_info.pBinds =
            sparse_memory_binds_.data() + sparse_buffer_bind.bind_offset;
      }
      assert_false(semaphores_free_.empty());
      VkSemaphore bind_sparse_semaphore = semaphores_free_.back();
      VkBindSparseInfo bind_sparse_info;
      bind_sparse_info.sType = VK_STRUCTURE_TYPE_BIND_SPARSE_INFO;
      bind_sparse_info.pNext = nullptr;
      bind_sparse_info.waitSemaphoreCount = 0;
      bind_sparse_info.pWaitSemaphores = nullptr;
      bind_sparse_info.bufferBindCount = uint32_t(sparse_buffer_bind_infos_temp_.size());
      bind_sparse_info.pBufferBinds =
          !sparse_buffer_bind_infos_temp_.empty() ? sparse_buffer_bind_infos_temp_.data() : nullptr;
      bind_sparse_info.imageOpaqueBindCount = 0;
      bind_sparse_info.pImageOpaqueBinds = nullptr;
      bind_sparse_info.imageBindCount = 0;
      bind_sparse_info.pImageBinds = 0;
      bind_sparse_info.signalSemaphoreCount = 1;
      bind_sparse_info.pSignalSemaphores = &bind_sparse_semaphore;
      VkResult bind_sparse_result;
      {
        ui::vulkan::VulkanDevice::Queue::Acquisition queue_acquisition = [&] {
          FrameCpuTimeline::Scope audit_queue_acquire(GetFrameCpuTimeline(),
                                                      FrameCpuTimeline::Bucket::kQueueAcquire);
          auto acquired =
              vulkan_device->AcquireQueue(vulkan_device->queue_family_sparse_binding(), 0);
          GetFrameCpuTimeline().NoteQueueAcquire();
          return acquired;
        }();
        bind_sparse_result =
            dfn.vkQueueBindSparse(queue_acquisition.queue(), 1, &bind_sparse_info, VK_NULL_HANDLE);
      }
      if (bind_sparse_result != VK_SUCCESS) {
        REXGPU_ERROR("Failed to submit Vulkan sparse binds");
        return false;
      }
      current_submission_wait_semaphores_.push_back(bind_sparse_semaphore);
      semaphores_free_.pop_back();
      current_submission_wait_stage_masks_.push_back(sparse_bind_wait_stage_mask_);
      sparse_bind_wait_stage_mask_ = 0;
      sparse_buffer_binds_.clear();
      sparse_memory_binds_.clear();
    }

    SubmitBarriers(true);

    assert_false(command_buffers_writable_.empty());
    CommandBuffer command_buffer = command_buffers_writable_.back();
    if (dfn.vkResetCommandPool(device, command_buffer.pool, 0) != VK_SUCCESS) {
      REXGPU_ERROR("Failed to reset a Vulkan command pool");
      return false;
    }
    VkCommandBufferBeginInfo command_buffer_begin_info;
    command_buffer_begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    command_buffer_begin_info.pNext = nullptr;
    command_buffer_begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    command_buffer_begin_info.pInheritanceInfo = nullptr;
    if (dfn.vkBeginCommandBuffer(command_buffer.buffer, &command_buffer_begin_info) != VK_SUCCESS) {
      REXGPU_ERROR("Failed to begin a Vulkan command buffer");
      return false;
    }
    uint32_t timestamp_slot = frame_current_ % kMaxTimestampFramesInFlight;
    uint32_t sub_idx = frame_submission_counts_[timestamp_slot];
    bool record_timestamps = (timestamp_query_pool_ != VK_NULL_HANDLE && timestamp_valid_bits_ > 0 &&
                              frame_open_ && sub_idx * 2 + 1 < kMaxQueriesPerFrameSlot);
    if (record_timestamps) {
      uint32_t start_q = timestamp_slot * kMaxQueriesPerFrameSlot + sub_idx * 2;
      dfn.vkCmdResetQueryPool(command_buffer.buffer, timestamp_query_pool_, start_q, 2);
      dfn.vkCmdWriteTimestamp(command_buffer.buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                              timestamp_query_pool_, start_q);
    }
    uint32_t transfer_count = transfer_profile_counts_[timestamp_slot];
    if (transfer_profile_pool_ && transfer_count > transfer_profile_reset_count_) {
      dfn.vkCmdResetQueryPool(command_buffer.buffer, transfer_profile_pool_,
          timestamp_slot * kTransferQueriesPerFrame + transfer_profile_reset_count_,
          transfer_count - transfer_profile_reset_count_);
      transfer_profile_reset_count_ = transfer_count;
    }
    uint32_t scope_count = scope_profile_counts_[timestamp_slot];
    if (scope_profile_pool_ && scope_count > scope_profile_reset_count_) {
      dfn.vkCmdResetQueryPool(command_buffer.buffer, scope_profile_pool_,
          timestamp_slot * kScopeQueriesPerFrame + scope_profile_reset_count_,
          scope_count - scope_profile_reset_count_);
      scope_profile_reset_count_ = scope_count;
    }
    const bool audit_submission = REXCVAR_GET(gpu_frame_audit_enable) && frame_open_;
    const uint64_t purpose_end = REXCVAR_GET(gpu_frame_audit_window_end);
    WorkPurposeAudit* purpose_audit =
        audit_submission && frame_current_ >= REXCVAR_GET(gpu_frame_audit_window_start) &&
                (!purpose_end || frame_current_ <= purpose_end)
            ? &work_purpose_audit_ : nullptr;
    if (purpose_audit) purpose_audit->BeginSubmission();
    auto audit_submit_start = audit_submission ? std::chrono::steady_clock::now()
                                                : std::chrono::steady_clock::time_point{};
    {
      FrameCpuTimeline::Scope audit_exec(GetFrameCpuTimeline(), FrameCpuTimeline::Bucket::kExecute);
      if (hoisted_uploads_pending_ && shared_memory_ != nullptr) {
        VkBufferMemoryBarrier pre_barrier;
        pre_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        pre_barrier.pNext = nullptr;
        pre_barrier.srcAccessMask = VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_SHADER_READ_BIT |
                                    VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        pre_barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        pre_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        pre_barrier.buffer = shared_memory_->buffer();
        pre_barrier.offset = 0;
        pre_barrier.size = VK_WHOLE_SIZE;
        dfn.vkCmdPipelineBarrier(
            command_buffer.buffer,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 1, &pre_barrier, 0, nullptr);

        prologue_deferred_command_buffer_.Execute(command_buffer.buffer, purpose_audit, true);

        VkBufferMemoryBarrier post_barrier;
        post_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        post_barrier.pNext = nullptr;
        post_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        post_barrier.dstAccessMask = VK_ACCESS_INDEX_READ_BIT | VK_ACCESS_SHADER_READ_BIT |
                                     VK_ACCESS_TRANSFER_READ_BIT;
        post_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        post_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        post_barrier.buffer = shared_memory_->buffer();
        post_barrier.offset = 0;
        post_barrier.size = VK_WHOLE_SIZE;
        dfn.vkCmdPipelineBarrier(
            command_buffer.buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT |
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &post_barrier, 0, nullptr);
      }
      deferred_command_buffer_.Execute(command_buffer.buffer, purpose_audit);
      if (purpose_audit) purpose_audit->EndSubmission();
    }
    if (record_timestamps) {
      uint32_t end_q = timestamp_slot * kMaxQueriesPerFrameSlot + sub_idx * 2 + 1;
      dfn.vkCmdWriteTimestamp(command_buffer.buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                              timestamp_query_pool_, end_q);
      frame_submission_counts_[timestamp_slot]++;
    }
    VkResult end_result;
    {
      FrameCpuTimeline::Scope audit_end_buffer(GetFrameCpuTimeline(), FrameCpuTimeline::Bucket::kEndCommandBuffer);
      end_result = dfn.vkEndCommandBuffer(command_buffer.buffer);
    }
    if (end_result != VK_SUCCESS) {
      REXGPU_ERROR("Failed to end a Vulkan command buffer");
      return false;
    }

    VkSubmitInfo submit_info;
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.pNext = nullptr;
    if (!current_submission_wait_semaphores_.empty()) {
      submit_info.waitSemaphoreCount = uint32_t(current_submission_wait_semaphores_.size());
      submit_info.pWaitSemaphores = current_submission_wait_semaphores_.data();
      submit_info.pWaitDstStageMask = current_submission_wait_stage_masks_.data();
    } else {
      submit_info.waitSemaphoreCount = 0;
      submit_info.pWaitSemaphores = nullptr;
      submit_info.pWaitDstStageMask = nullptr;
    }
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer.buffer;
    submit_info.signalSemaphoreCount = 0;
    submit_info.pSignalSemaphores = nullptr;
    assert_false(fences_free_.empty());
    VkFence fence = fences_free_.back();
    VkResult reset_fence_result;
    {
      FrameCpuTimeline::Scope audit_reset_fence(GetFrameCpuTimeline(),
                                                FrameCpuTimeline::Bucket::kResetFence);
      reset_fence_result = dfn.vkResetFences(device, 1, &fence);
    }
    if (reset_fence_result != VK_SUCCESS) {
      REXGPU_ERROR("Failed to reset a Vulkan submission fence");
      return false;
    }
    VkResult submit_result;
    ui::vulkan::VulkanDevice::Queue::Acquisition queue_acquisition = [&] {
      FrameCpuTimeline::Scope audit_queue_acquire(GetFrameCpuTimeline(),
                                                  FrameCpuTimeline::Bucket::kQueueAcquire);
      auto acquired =
          vulkan_device->AcquireQueue(vulkan_device->queue_family_graphics_compute(), 0);
      GetFrameCpuTimeline().NoteQueueAcquire();
      return acquired;
    }();
    {
      FrameCpuTimeline::Scope audit_submit(GetFrameCpuTimeline(), FrameCpuTimeline::Bucket::kSubmit);
      submit_result = dfn.vkQueueSubmit(queue_acquisition.queue(), 1, &submit_info, fence);
    }
    if (audit_submission) {
      frame_cpu_profiling_stats_.time_exec_submit += std::chrono::steady_clock::now() - audit_submit_start;
      ++frame_cpu_profiling_stats_.submit_count;
    }
    if (submit_result != VK_SUCCESS) {
      REXGPU_ERROR("Failed to submit a Vulkan command buffer");
      if (submit_result == VK_ERROR_DEVICE_LOST && !device_lost_) {
        device_lost_ = true;
        if (graphics_system_) {
          graphics_system_->OnHostGpuLossFromAnyThread(true);
        }
      }
      return false;
    }
    uint64_t submission_current = GetCurrentSubmission();
    current_submission_wait_stage_masks_.clear();
    for (VkSemaphore semaphore : current_submission_wait_semaphores_) {
      submissions_in_flight_semaphores_.emplace_back(submission_current, semaphore);
    }
    current_submission_wait_semaphores_.clear();
    command_buffers_submitted_.emplace_back(submission_current, command_buffer);
    command_buffers_writable_.pop_back();
    // Increments the current submission number, going to the next submission.
    submissions_in_flight_fences_.push_back(fence);
    fences_free_.pop_back();

    if (is_closing_frame) {
      gpu_completion_tracker_.TrackFrameClosingFence(frame_current_, fence);
    }

    submission_open_ = false;
  }

  if (is_closing_frame) {
    if (diagnostic_storage_enabled_ && frame_current_ == diagnostic_storage_frame_ &&
        !diagnostic_storage_attempt_consumed_) {
      LogFragmentDiagnosticOutcome(
          frame_current_, diagnostic_storage_last_vertex_hash_, diagnostic_storage_last_pixel_hash_,
          diagnostic_storage_last_draw_ordinal_, "not_selected", 0, false, "no_eligible_matching_draw");
    }
    if (FlickerDiagEnabled()) {
      flicker_frame_csv.Flush(frame_current_, frame_cpu_profiling_stats_.draw_count);
      flicker_frame_pixel_texture_slots = 0;
      flicker_frame_null_pixel_textures = 0;
    }
    auto frame_now = std::chrono::steady_clock::now();
    GetFrameCpuTimeline().Stop(frame_now);
    double frame_wall_ms =
        std::chrono::duration<double, std::milli>(frame_now - frame_start_time_).count();

    bool frame_was_profiled = (frame_cpu_profiling_stats_.draw_count > 0);
    double total_draw_ms = frame_cpu_profiling_stats_.time_draw_total.count() / 1e6;
    double prim_ms = frame_cpu_profiling_stats_.time_primitive_processing.count() / 1e6;
    double rts_ms = frame_cpu_profiling_stats_.time_render_target_update.count() / 1e6;
    double pipe_ms = frame_cpu_profiling_stats_.time_pipeline_configuration.count() / 1e6;
    double bind_ms = frame_cpu_profiling_stats_.time_binding_updates.count() / 1e6;
    double shmem_ms = frame_cpu_profiling_stats_.time_shared_memory_residency.count() / 1e6;
    double bar_rp_ms = frame_cpu_profiling_stats_.time_barriers_and_render_pass.count() / 1e6;
    double vk_draw_ms = frame_cpu_profiling_stats_.time_vk_draw.count() / 1e6;
    double fence_wait_ms = frame_cpu_profiling_stats_.time_fence_waits.count() / 1e6;
    double other_ms = frame_was_profiled
                          ? (total_draw_ms - (prim_ms + rts_ms + pipe_ms + bind_ms + shmem_ms +
                                              bar_rp_ms + vk_draw_ms))
                          : 0.0;

    interval_cpu_profiling_stats_.total_draw_count += frame_cpu_profiling_stats_.draw_count;

    bool audit_enabled = REXCVAR_GET(gpu_frame_audit_enable);
    uint64_t audit_start = REXCVAR_GET(gpu_frame_audit_window_start);
    uint64_t audit_end = REXCVAR_GET(gpu_frame_audit_window_end);
    bool in_audit_window = audit_enabled &&
                           (audit_start == 0 || frame_current_ >= audit_start) &&
                           (audit_end == 0 || frame_current_ <= audit_end);

    if (RasterVariantAuditActive(frame_current_)) {
      FlushRasterVariantAudit();
    } else {
      raster_variant_audit_.Clear();
    }
    if (in_audit_window) {
      RecordFrameAuditRow(frame_current_, frame_wall_ms);
    }
    FlushCompletedAuditRows(false);
    if (audit_end != 0 && frame_current_ == audit_end) {
      FlushCompletedAuditRows(true);
      LogScopeProfileAggregateSummary();
      LogTopBarrierTuples();
    }
    if (IsGpuProvenanceFrame(frame_current_)) {
      FlushCompletedAuditRows(true);
      LogTopBarrierTuples();
    }

    if (frame_was_profiled) {
      double interval_fence_wait_ms =
          interval_cpu_profiling_stats_.time_fence_waits.count() / 1e6;
      uint64_t interval_frames =
          std::max<uint64_t>(1, interval_cpu_profiling_stats_.frame_count);
      REXGPU_INFO(
          "[Pillar 10 CPU Profile] Frame {} ({} draws, wall={:.2f}ms): total_draw_recording={:.2f}ms "
          "[prim={:.2f}ms rts={:.2f}ms pipe={:.2f}ms bind={:.2f}ms shmem={:.2f}ms "
          "bar_rp={:.2f}ms vk_draw={:.2f}ms frame_fence_wait={:.2f}ms/{} interval_fence_wait={:.2f}ms/{} over {}f other={:.2f}ms]",
          frame_current_, frame_cpu_profiling_stats_.draw_count, frame_wall_ms,
          total_draw_ms, prim_ms, rts_ms, pipe_ms, bind_ms, shmem_ms, bar_rp_ms, vk_draw_ms,
          fence_wait_ms, frame_cpu_profiling_stats_.fence_wait_count,
          interval_fence_wait_ms, interval_cpu_profiling_stats_.total_fence_wait_count, interval_frames,
          other_ms);
      interval_cpu_profiling_stats_.Reset();
    }

    // Always reset per-frame statistics so fence waits and draw timings are never misaligned or leaked across frames!
    frame_cpu_profiling_stats_.Reset();
    frame_work_counters_.Reset();
    frame_pass_stats_.Reset();
    current_frame_barrier_breaks_shmem_ = 0;
    current_frame_barrier_breaks_texture_ = 0;
    current_frame_barrier_breaks_rt_ = 0;
    current_frame_barrier_breaks_prim_ = 0;
    current_frame_barrier_breaks_other_ = 0;
    current_frame_shmem_hoist_fallbacks_ = 0;
    current_frame_shmem_hoist_successes_ = 0;

    if (s_prev_ben_delta.frame != 0 && s_current_ben_delta.char_draw_count > 0) {
      bool changed = false;
      if (s_current_ben_delta.char_draw_count != s_prev_ben_delta.char_draw_count) changed = true;
      if (s_current_ben_delta.texture_hash != s_prev_ben_delta.texture_hash) changed = true;
      if (s_current_ben_delta.blend_hash != s_prev_ben_delta.blend_hash) changed = true;
      if (s_current_ben_delta.constant_hash != s_prev_ben_delta.constant_hash) changed = true;
      for (int i = 0; i < 8; ++i) {
        if (s_current_ben_delta.submesh_counts[i] != s_prev_ben_delta.submesh_counts[i]) changed = true;
      }
      if (changed) {
        LogFragmentDiagnostic(fmt::format(
            "BEN_FLICKER_DELTA: frame={} (prev={}) draws={}(prev={}) tex_changed={} blend_changed={} const_changed={} "
            "submeshes=[{},{},{},{},{},{},{},{}] prev=[{},{},{},{},{},{},{},{}]",
            frame_current_, s_prev_ben_delta.frame,
            s_current_ben_delta.char_draw_count, s_prev_ben_delta.char_draw_count,
            (s_current_ben_delta.texture_hash != s_prev_ben_delta.texture_hash),
            (s_current_ben_delta.blend_hash != s_prev_ben_delta.blend_hash),
            (s_current_ben_delta.constant_hash != s_prev_ben_delta.constant_hash),
            s_current_ben_delta.submesh_counts[0], s_current_ben_delta.submesh_counts[1],
            s_current_ben_delta.submesh_counts[2], s_current_ben_delta.submesh_counts[3],
            s_current_ben_delta.submesh_counts[4], s_current_ben_delta.submesh_counts[5],
            s_current_ben_delta.submesh_counts[6], s_current_ben_delta.submesh_counts[7],
            s_prev_ben_delta.submesh_counts[0], s_prev_ben_delta.submesh_counts[1],
            s_prev_ben_delta.submesh_counts[2], s_prev_ben_delta.submesh_counts[3],
            s_prev_ben_delta.submesh_counts[4], s_prev_ben_delta.submesh_counts[5],
            s_prev_ben_delta.submesh_counts[6], s_prev_ben_delta.submesh_counts[7]));
      }
    }
    if (frame_current_ % 300 == 0 && s_current_ben_delta.char_draw_count > 0) {
      LogFragmentDiagnostic(fmt::format(
          "BEN_IDLE_HEARTBEAT: frame={} draws={} submeshes=[{},{},{},{},{},{},{},{}]",
          frame_current_, s_current_ben_delta.char_draw_count,
          s_current_ben_delta.submesh_counts[0], s_current_ben_delta.submesh_counts[1],
          s_current_ben_delta.submesh_counts[2], s_current_ben_delta.submesh_counts[3],
          s_current_ben_delta.submesh_counts[4], s_current_ben_delta.submesh_counts[5],
          s_current_ben_delta.submesh_counts[6], s_current_ben_delta.submesh_counts[7]));
    }
    s_current_ben_delta.frame = uint32_t(frame_current_);
    s_prev_ben_delta = s_current_ben_delta;
    s_current_ben_delta = {};

    if (REXCVAR_GET(clear_memory_page_state) && shared_memory_) {
      shared_memory_->SetSystemPageBlocksValidWithGpuDataWritten();
    }
    if (REXCVAR_GET(vulkan_early_z_opaque) &&
        (IsGpuProvenanceFrame(frame_current_) || in_audit_window)) {
      REXGPU_INFO("[EARLY_Z_FRAME] frame={} early_z_draws={}/{}", frame_current_,
                  current_frame_early_z_draws_, frame_cpu_profiling_stats_.draw_count);
    }
    current_frame_early_z_draws_ = 0;

    frame_open_ = false;
    // Submission already closed now, so minus 1.
    closed_frame_submissions_[(frame_current_++) % kMaxFramesInFlight] = GetCurrentSubmission() - 1;

    if (cache_clear_requested_ && AwaitAllQueueOperationsCompletion()) {
      cache_clear_requested_ = false;

      DestroyScratchBuffer();

      for (SwapFramebuffer& swap_framebuffer : swap_framebuffers_) {
        ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyFramebuffer, device,
                                               swap_framebuffer.framebuffer);
      }

      assert_true(command_buffers_submitted_.empty());
      for (const CommandBuffer& command_buffer : command_buffers_writable_) {
        dfn.vkDestroyCommandPool(device, command_buffer.pool, nullptr);
      }
      command_buffers_writable_.clear();

      ClearTransientDescriptorPools();

      uniform_buffer_pool_->ClearCache();

      texture_cache_->ClearCache();

      render_target_cache_->ClearCache();

      // Not clearing the pipeline layouts and the descriptor set layouts as
      // they're referenced by pipelines, which are not destroyed.

      primitive_processor_->ClearCache();
      bound_index_buffer_ = VK_NULL_HANDLE;
      bound_index_buffer_offset_ = 0;
      bound_index_type_ = VK_INDEX_TYPE_MAX_ENUM;

      shared_memory_->ClearCache();
      validated_shared_memory_ranges_count_ = 0;
      validated_shared_memory_ranges_epoch_ = 0;
    }
  }

  return true;
}

void VulkanCommandProcessor::ClearTransientDescriptorPools() {
  texture_transient_descriptor_sets_free_.clear();
  texture_transient_descriptor_sets_used_.clear();
  transient_descriptor_allocator_textures_.Reset();

  constants_transient_descriptors_free_.clear();
  constants_transient_descriptors_used_.clear();
  // The Set 1 constant descriptor sets are persistent - allocated and written
  // once in SetupContext and bound with dynamic offsets only - so there is
  // nothing to recycle for them here.
  last_bound_constant_buffer_dynamic_offsets_.fill(UINT32_MAX);

  for (std::vector<VkDescriptorSet>& transient_descriptors_free :
       single_transient_descriptors_free_) {
    transient_descriptors_free.clear();
  }
  single_transient_descriptors_used_.clear();
  transient_descriptor_allocator_storage_buffer_.Reset();
  transient_descriptor_allocator_uniform_buffer_.Reset();
}

void VulkanCommandProcessor::SplitPendingBarrier() {
  size_t pending_buffer_memory_barrier_count = pending_barriers_buffer_memory_barriers_.size();
  size_t pending_image_memory_barrier_count = pending_barriers_image_memory_barriers_.size();
  if (!current_pending_barrier_.src_stage_mask && !current_pending_barrier_.dst_stage_mask &&
      current_pending_barrier_.buffer_memory_barriers_offset >=
          pending_buffer_memory_barrier_count &&
      current_pending_barrier_.image_memory_barriers_offset >= pending_image_memory_barrier_count) {
    return;
  }
  pending_barriers_.emplace_back(current_pending_barrier_);
  current_pending_barrier_.src_stage_mask = 0;
  current_pending_barrier_.dst_stage_mask = 0;
  current_pending_barrier_.buffer_memory_barriers_offset = pending_buffer_memory_barrier_count;
  current_pending_barrier_.image_memory_barriers_offset = pending_image_memory_barrier_count;
}

void VulkanCommandProcessor::DestroyScratchBuffer() {
  assert_false(scratch_buffer_used_);

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  scratch_buffer_last_usage_submission_ = 0;
  scratch_buffer_last_access_mask_ = 0;
  scratch_buffer_last_stage_mask_ = 0;
  scratch_buffer_size_ = 0;
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, scratch_buffer_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, scratch_buffer_memory_);
}

bool VulkanCommandProcessor::EnsureSwapFxaaSourceImage(uint32_t width, uint32_t height) {
  assert_true(submission_open_);
  if (!width || !height) {
    return false;
  }
  if (swap_fxaa_source_image_ != VK_NULL_HANDLE && swap_fxaa_source_image_width_ == width &&
      swap_fxaa_source_image_height_ == height) {
    return true;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  if (swap_fxaa_source_image_ != VK_NULL_HANDLE) {
    const uint64_t destroy_submission = swap_fxaa_source_image_submission_;
    const uint64_t deferred_destroy_submission = GetCurrentSubmission();
    if (submission_completed_ >= destroy_submission) {
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device,
                                             swap_fxaa_source_image_view_);
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, swap_fxaa_source_image_);
      ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                             swap_fxaa_source_image_memory_);
    } else {
      if (swap_fxaa_source_image_view_ != VK_NULL_HANDLE) {
        destroy_image_views_.emplace_back(deferred_destroy_submission,
                                          swap_fxaa_source_image_view_);
        swap_fxaa_source_image_view_ = VK_NULL_HANDLE;
      }
      if (swap_fxaa_source_image_ != VK_NULL_HANDLE) {
        destroy_images_.emplace_back(deferred_destroy_submission, swap_fxaa_source_image_);
        swap_fxaa_source_image_ = VK_NULL_HANDLE;
      }
      if (swap_fxaa_source_image_memory_ != VK_NULL_HANDLE) {
        destroy_memory_.emplace_back(deferred_destroy_submission, swap_fxaa_source_image_memory_);
        swap_fxaa_source_image_memory_ = VK_NULL_HANDLE;
      }
    }
  }
  swap_fxaa_source_image_width_ = 0;
  swap_fxaa_source_image_height_ = 0;
  swap_fxaa_source_image_submission_ = 0;
  swap_fxaa_source_stage_mask_ = 0;
  swap_fxaa_source_access_mask_ = 0;
  swap_fxaa_source_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;

  VkImageCreateInfo image_create_info;
  image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_create_info.pNext = nullptr;
  image_create_info.flags = 0;
  image_create_info.imageType = VK_IMAGE_TYPE_2D;
  image_create_info.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  image_create_info.extent.width = width;
  image_create_info.extent.height = height;
  image_create_info.extent.depth = 1;
  image_create_info.mipLevels = 1;
  image_create_info.arrayLayers = 1;
  image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_create_info.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_create_info.queueFamilyIndexCount = 0;
  image_create_info.pQueueFamilyIndices = nullptr;
  image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  if (!ui::vulkan::util::CreateDedicatedAllocationImage(
          vulkan_device, image_create_info, ui::vulkan::util::MemoryPurpose::kDeviceLocal,
          swap_fxaa_source_image_, swap_fxaa_source_image_memory_)) {
    REXGPU_ERROR("Failed to create the FXAA source image");
    return false;
  }

  VkImageViewCreateInfo image_view_create_info;
  image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  image_view_create_info.pNext = nullptr;
  image_view_create_info.flags = 0;
  image_view_create_info.image = swap_fxaa_source_image_;
  image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
  image_view_create_info.format = image_create_info.format;
  image_view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
  image_view_create_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange();
  if (dfn.vkCreateImageView(device, &image_view_create_info, nullptr,
                            &swap_fxaa_source_image_view_) != VK_SUCCESS) {
    REXGPU_ERROR("Failed to create the FXAA source image view");
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, swap_fxaa_source_image_);
    ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device,
                                           swap_fxaa_source_image_memory_);
    return false;
  }

  swap_fxaa_source_image_width_ = width;
  swap_fxaa_source_image_height_ = height;
  swap_fxaa_source_image_submission_ = 0;
  swap_fxaa_source_stage_mask_ = 0;
  swap_fxaa_source_access_mask_ = 0;
  swap_fxaa_source_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
  return true;
}

void VulkanCommandProcessor::DestroySwapFxaaSourceImage() {
  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImageView, device,
                                         swap_fxaa_source_image_view_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyImage, device, swap_fxaa_source_image_);
  ui::vulkan::util::DestroyAndNullHandle(dfn.vkFreeMemory, device, swap_fxaa_source_image_memory_);
  swap_fxaa_source_image_width_ = 0;
  swap_fxaa_source_image_height_ = 0;
  swap_fxaa_source_image_submission_ = 0;
  swap_fxaa_source_stage_mask_ = 0;
  swap_fxaa_source_access_mask_ = 0;
  swap_fxaa_source_layout_ = VK_IMAGE_LAYOUT_UNDEFINED;
}

void VulkanCommandProcessor::UpdateDynamicState(const draw_util::ViewportInfo& viewport_info,
                                                bool primitive_polygonal,
                                                reg::RB_DEPTHCONTROL normalized_depth_control) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  const RegisterFile& regs = *register_file_;
  uint32_t draw_resolution_scale_x = texture_cache_ ? texture_cache_->draw_resolution_scale_x() : 1;
  uint32_t draw_resolution_scale_y = texture_cache_ ? texture_cache_->draw_resolution_scale_y() : 1;

  // Window parameters.
  // http://ftp.tku.edu.tw/NetBSD/NetBSD-current/xsrc/external/mit/xf86-video-ati/dist/src/r600_reg_auto_r6xx.h
  // See r200UpdateWindow:
  // https://github.com/freedreno/mesa/blob/master/src/mesa/drivers/dri/r200/r200_state.c
  auto pa_sc_window_offset = regs.Get<reg::PA_SC_WINDOW_OFFSET>();

  // Viewport.
  VkViewport viewport;
  if (viewport_info.xy_extent[0] && viewport_info.xy_extent[1]) {
    viewport.x = float(viewport_info.xy_offset[0]);
    viewport.y = float(viewport_info.xy_offset[1]);
    viewport.width = float(viewport_info.xy_extent[0]);
    viewport.height = float(viewport_info.xy_extent[1]);
  } else {
    // Vulkan viewport width must be greater than 0.0f, but the Xenia  viewport
    // may be empty for various reasons - set the viewport to outside the
    // framebuffer.
    viewport.x = -1.0f;
    viewport.y = -1.0f;
    viewport.width = 1.0f;
    viewport.height = 1.0f;
  }
  viewport.minDepth = viewport_info.z_min;
  viewport.maxDepth = viewport_info.z_max;
  SetViewport(viewport);

  // Scissor.
  draw_util::Scissor scissor;
  draw_util::GetScissor(regs, scissor);
  scissor.offset[0] *= draw_resolution_scale_x;
  scissor.offset[1] *= draw_resolution_scale_y;
  scissor.extent[0] *= draw_resolution_scale_x;
  scissor.extent[1] *= draw_resolution_scale_y;
  VkRect2D scissor_rect;
  scissor_rect.offset.x = int32_t(scissor.offset[0]);
  scissor_rect.offset.y = int32_t(scissor.offset[1]);
  scissor_rect.extent.width = scissor.extent[0];
  scissor_rect.extent.height = scissor.extent[1];
  SetScissor(scissor_rect);

  if (render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets) {
    // Depth bias.
    float depth_bias_constant_factor, depth_bias_slope_factor;
    draw_util::GetPreferredFacePolygonOffset(regs, primitive_polygonal, depth_bias_slope_factor,
                                             depth_bias_constant_factor);
    depth_bias_constant_factor *=
        regs.Get<reg::RB_DEPTH_INFO>().depth_format == xenos::DepthRenderTargetFormat::kD24S8
            ? draw_util::kD3D10PolygonOffsetFactorUnorm24
            : draw_util::kD3D10PolygonOffsetFactorFloat24;
    // With non-square resolution scaling, make sure the worst-case impact is
    // reverted (slope only along the scaled axis), thus max. More bias is
    // better than less bias, because less bias means Z fighting with the
    // background is more likely.
    depth_bias_slope_factor *= xenos::kPolygonOffsetScaleSubpixelUnit *
                               float(std::max(draw_resolution_scale_x, draw_resolution_scale_y));
    // std::memcmp instead of != so in case of NaN, every draw won't be
    // invalidating it.
    dynamic_depth_bias_update_needed_ |=
        std::memcmp(&dynamic_depth_bias_constant_factor_, &depth_bias_constant_factor,
                    sizeof(float)) != 0;
    dynamic_depth_bias_update_needed_ |= std::memcmp(&dynamic_depth_bias_slope_factor_,
                                                     &depth_bias_slope_factor, sizeof(float)) != 0;
    if (dynamic_depth_bias_update_needed_) {
      dynamic_depth_bias_constant_factor_ = depth_bias_constant_factor;
      dynamic_depth_bias_slope_factor_ = depth_bias_slope_factor;
      deferred_command_buffer_.CmdVkSetDepthBias(dynamic_depth_bias_constant_factor_, 0.0f,
                                                 dynamic_depth_bias_slope_factor_);
      dynamic_depth_bias_update_needed_ = false;
    }

    // Blend constants.
    float blend_constants[] = {
        regs.Get<float>(XE_GPU_REG_RB_BLEND_RED),
        regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN),
        regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE),
        regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA),
    };
    dynamic_blend_constants_update_needed_ |=
        std::memcmp(dynamic_blend_constants_, blend_constants, sizeof(float) * 4) != 0;
    if (dynamic_blend_constants_update_needed_) {
      std::memcpy(dynamic_blend_constants_, blend_constants, sizeof(float) * 4);
      deferred_command_buffer_.CmdVkSetBlendConstants(dynamic_blend_constants_);
      dynamic_blend_constants_update_needed_ = false;
    }

    // Stencil masks and references.
    // Due to pretty complex conditions involving registers not directly related
    // to stencil (primitive type, culling), changing the values only when
    // stencil is actually needed. However, due to the way dynamic state needs
    // to be set in Vulkan, which doesn't take into account whether the state
    // actually has effect on drawing, and because the masks and the references
    // are always dynamic in Xenia guest pipelines, they must be set in the
    // command buffer before any draw.
    if (normalized_depth_control.stencil_enable) {
      Register stencil_ref_mask_front_reg, stencil_ref_mask_back_reg;
      if (primitive_polygonal && normalized_depth_control.backface_enable) {
        if (GetVulkanDevice()->properties().separateStencilMaskRef) {
          stencil_ref_mask_front_reg = XE_GPU_REG_RB_STENCILREFMASK;
          stencil_ref_mask_back_reg = XE_GPU_REG_RB_STENCILREFMASK_BF;
        } else {
          // Choose the back face values only if drawing only back faces.
          stencil_ref_mask_front_reg = regs.Get<reg::PA_SU_SC_MODE_CNTL>().cull_front
                                           ? XE_GPU_REG_RB_STENCILREFMASK_BF
                                           : XE_GPU_REG_RB_STENCILREFMASK;
          stencil_ref_mask_back_reg = stencil_ref_mask_front_reg;
        }
      } else {
        stencil_ref_mask_front_reg = XE_GPU_REG_RB_STENCILREFMASK;
        stencil_ref_mask_back_reg = XE_GPU_REG_RB_STENCILREFMASK;
      }
      auto stencil_ref_mask_front = regs.Get<reg::RB_STENCILREFMASK>(stencil_ref_mask_front_reg);
      auto stencil_ref_mask_back = regs.Get<reg::RB_STENCILREFMASK>(stencil_ref_mask_back_reg);
      // Compare mask.
      dynamic_stencil_compare_mask_front_update_needed_ |=
          dynamic_stencil_compare_mask_front_ != stencil_ref_mask_front.stencilmask;
      dynamic_stencil_compare_mask_front_ = stencil_ref_mask_front.stencilmask;
      dynamic_stencil_compare_mask_back_update_needed_ |=
          dynamic_stencil_compare_mask_back_ != stencil_ref_mask_back.stencilmask;
      dynamic_stencil_compare_mask_back_ = stencil_ref_mask_back.stencilmask;
      // Write mask.
      dynamic_stencil_write_mask_front_update_needed_ |=
          dynamic_stencil_write_mask_front_ != stencil_ref_mask_front.stencilwritemask;
      dynamic_stencil_write_mask_front_ = stencil_ref_mask_front.stencilwritemask;
      dynamic_stencil_write_mask_back_update_needed_ |=
          dynamic_stencil_write_mask_back_ != stencil_ref_mask_back.stencilwritemask;
      dynamic_stencil_write_mask_back_ = stencil_ref_mask_back.stencilwritemask;
      // Reference.
      dynamic_stencil_reference_front_update_needed_ |=
          dynamic_stencil_reference_front_ != stencil_ref_mask_front.stencilref;
      dynamic_stencil_reference_front_ = stencil_ref_mask_front.stencilref;
      dynamic_stencil_reference_back_update_needed_ |=
          dynamic_stencil_reference_back_ != stencil_ref_mask_back.stencilref;
      dynamic_stencil_reference_back_ = stencil_ref_mask_back.stencilref;
    }
    // Using VK_STENCIL_FACE_FRONT_AND_BACK for higher safety when running on
    // the Vulkan portability subset without separateStencilMaskRef.
    if (dynamic_stencil_compare_mask_front_update_needed_ ||
        dynamic_stencil_compare_mask_back_update_needed_) {
      if (dynamic_stencil_compare_mask_front_ == dynamic_stencil_compare_mask_back_) {
        deferred_command_buffer_.CmdVkSetStencilCompareMask(VK_STENCIL_FACE_FRONT_AND_BACK,
                                                            dynamic_stencil_compare_mask_front_);
      } else {
        if (dynamic_stencil_compare_mask_front_update_needed_) {
          deferred_command_buffer_.CmdVkSetStencilCompareMask(VK_STENCIL_FACE_FRONT_BIT,
                                                              dynamic_stencil_compare_mask_front_);
        }
        if (dynamic_stencil_compare_mask_back_update_needed_) {
          deferred_command_buffer_.CmdVkSetStencilCompareMask(VK_STENCIL_FACE_BACK_BIT,
                                                              dynamic_stencil_compare_mask_back_);
        }
      }
      dynamic_stencil_compare_mask_front_update_needed_ = false;
      dynamic_stencil_compare_mask_back_update_needed_ = false;
    }
    if (dynamic_stencil_write_mask_front_update_needed_ ||
        dynamic_stencil_write_mask_back_update_needed_) {
      if (dynamic_stencil_write_mask_front_ == dynamic_stencil_write_mask_back_) {
        deferred_command_buffer_.CmdVkSetStencilWriteMask(VK_STENCIL_FACE_FRONT_AND_BACK,
                                                          dynamic_stencil_write_mask_front_);
      } else {
        if (dynamic_stencil_write_mask_front_update_needed_) {
          deferred_command_buffer_.CmdVkSetStencilWriteMask(VK_STENCIL_FACE_FRONT_BIT,
                                                            dynamic_stencil_write_mask_front_);
        }
        if (dynamic_stencil_write_mask_back_update_needed_) {
          deferred_command_buffer_.CmdVkSetStencilWriteMask(VK_STENCIL_FACE_BACK_BIT,
                                                            dynamic_stencil_write_mask_back_);
        }
      }
      dynamic_stencil_write_mask_front_update_needed_ = false;
      dynamic_stencil_write_mask_back_update_needed_ = false;
    }
    if (dynamic_stencil_reference_front_update_needed_ ||
        dynamic_stencil_reference_back_update_needed_) {
      if (dynamic_stencil_reference_front_ == dynamic_stencil_reference_back_) {
        deferred_command_buffer_.CmdVkSetStencilReference(VK_STENCIL_FACE_FRONT_AND_BACK,
                                                          dynamic_stencil_reference_front_);
      } else {
        if (dynamic_stencil_reference_front_update_needed_) {
          deferred_command_buffer_.CmdVkSetStencilReference(VK_STENCIL_FACE_FRONT_BIT,
                                                            dynamic_stencil_reference_front_);
        }
        if (dynamic_stencil_reference_back_update_needed_) {
          deferred_command_buffer_.CmdVkSetStencilReference(VK_STENCIL_FACE_BACK_BIT,
                                                            dynamic_stencil_reference_back_);
        }
      }
      dynamic_stencil_reference_front_update_needed_ = false;
      dynamic_stencil_reference_back_update_needed_ = false;
    }
  }

  // TODO(Triang3l): VK_EXT_extended_dynamic_state and
  // VK_EXT_extended_dynamic_state2.
}

void VulkanCommandProcessor::UpdateSystemConstantValues(
    bool primitive_polygonal,
    const PrimitiveProcessor::ProcessingResult& primitive_processing_result,
    bool shader_32bit_index_dma, uint32_t compute_memexport_vertex_count,
    const draw_util::ViewportInfo& viewport_info, uint32_t used_texture_mask,
    reg::RB_DEPTHCONTROL normalized_depth_control, uint32_t normalized_color_mask) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  const RegisterFile& regs = *register_file_;
  auto pa_cl_clip_cntl = regs.Get<reg::PA_CL_CLIP_CNTL>();
  auto pa_cl_vte_cntl = regs.Get<reg::PA_CL_VTE_CNTL>();
  auto pa_su_sc_mode_cntl = regs.Get<reg::PA_SU_SC_MODE_CNTL>();
  auto rb_alpha_ref = regs.Get<float>(XE_GPU_REG_RB_ALPHA_REF);
  auto rb_colorcontrol = regs.Get<reg::RB_COLORCONTROL>();
  auto rb_depth_info = regs.Get<reg::RB_DEPTH_INFO>();
  auto rb_stencilrefmask = regs.Get<reg::RB_STENCILREFMASK>();
  auto rb_stencilrefmask_bf = regs.Get<reg::RB_STENCILREFMASK>(XE_GPU_REG_RB_STENCILREFMASK_BF);
  auto rb_surface_info = regs.Get<reg::RB_SURFACE_INFO>();
  auto vgt_dma_size = regs.Get<reg::VGT_DMA_SIZE>();
  auto vgt_draw_initiator = regs.Get<reg::VGT_DRAW_INITIATOR>();
  auto vgt_indx_offset = regs.Get<int32_t>(XE_GPU_REG_VGT_INDX_OFFSET);
  auto vgt_max_vtx_indx = regs.Get<uint32_t>(XE_GPU_REG_VGT_MAX_VTX_INDX);
  auto vgt_min_vtx_indx = regs.Get<uint32_t>(XE_GPU_REG_VGT_MIN_VTX_INDX);

  bool edram_fragment_shader_interlock =
      render_target_cache_->GetPath() == RenderTargetCache::Path::kPixelShaderInterlock;
  uint32_t draw_resolution_scale_x = texture_cache_->draw_resolution_scale_x();
  uint32_t draw_resolution_scale_y = texture_cache_->draw_resolution_scale_y();

  // Get the color info register values for each render target. Also, for FSI,
  // exclude components that don't exist in the format from the write mask.
  // Don't exclude fully overlapping render targets, however - two render
  // targets with the same base address are used in the lighting pass of
  // 4D5307E6, for example, with the needed one picked with dynamic control
  // flow.
  reg::RB_COLOR_INFO color_infos[xenos::kMaxColorRenderTargets];
  float rt_clamp[4][4];
  // Two UINT32_MAX if no components actually existing in the RT are written.
  uint32_t rt_keep_masks[4][2];
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    auto color_info = regs.Get<reg::RB_COLOR_INFO>(reg::RB_COLOR_INFO::rt_register_indices[i]);
    color_infos[i] = color_info;
    if (edram_fragment_shader_interlock) {
      RenderTargetCache::GetPSIColorFormatInfo(
          color_info.color_format, (normalized_color_mask >> (i * 4)) & 0b1111, rt_clamp[i][0],
          rt_clamp[i][1], rt_clamp[i][2], rt_clamp[i][3], rt_keep_masks[i][0], rt_keep_masks[i][1]);
    }
  }

  // Disable depth and stencil if it aliases a color render target (for
  // instance, during the XBLA logo in 58410954, though depth writing is already
  // disabled there).
  bool depth_stencil_enabled =
      normalized_depth_control.stencil_enable || normalized_depth_control.z_enable;
  if (edram_fragment_shader_interlock && depth_stencil_enabled) {
    for (uint32_t i = 0; i < 4; ++i) {
      if (rb_depth_info.depth_base == color_infos[i].color_base &&
          (rt_keep_masks[i][0] != UINT32_MAX || rt_keep_masks[i][1] != UINT32_MAX)) {
        depth_stencil_enabled = false;
        break;
      }
    }
  }

  bool dirty = false;

  // [DEBUG-face-stage] One-process, bounded component/UV sequence. No mode is
  // active unless this explicit control file exists; normal output at 8680
  // and 8690 brackets the nine diagnostic outputs at 8681..8690, and
  // 8691..8694 repeat the diffuse/UV pair with and without forced LOD 0.
  static const bool face_stage_enabled = [] {
    std::ifstream control("/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/enable.txt");
    return control.good();
  }();
  uint32_t face_stage_mode = 0;
  uint32_t face_stage_lod0 = 0;
  if (face_stage_enabled) {
    // Ordered so the decisive probes land on the earliest frames, which are the
    // ones the capture window reliably covers: 18 = fixed-UV reference sample,
    // 17 = full-precision coordinates, 16 = raw RGBA sample, 9 = white coverage.
    static constexpr uint32_t kFaceStageModeSequence[] = {
        9, 16, 10, 15, 8, 1, 2, 5, 7, 3, 4, 6, 11, 12, 13, 14, 17, 18,
    };
    constexpr uint32_t kFaceStageModeSequenceLength =
        sizeof(kFaceStageModeSequence) / sizeof(kFaceStageModeSequence[0]);
    constexpr uint32_t kFaceStageModeFirstFrame = 8681;
    if (frame_current_ >= kFaceStageModeFirstFrame &&
        frame_current_ < kFaceStageModeFirstFrame + kFaceStageModeSequenceLength) {
      face_stage_mode =
          kFaceStageModeSequence[frame_current_ - kFaceStageModeFirstFrame];
    }
  }
  static bool host_single_sample_logged = false;
  if (!host_single_sample_logged && diagnostics::ForceHostSingleSample()) {
    host_single_sample_logged = true;
    REXGPU_INFO("[QuadPartnerExperiment] host_single_sample=true");
  }
  if (diagnostics::QuadPartnerMarkerEnabled()) {
    face_stage_mode = frame_current_ == diagnostics::kQuadPartnerMarkerFrame
                          ? diagnostics::QuadPartnerOutputMode() : 0;
    if (frame_current_ == diagnostics::kQuadPartnerMarkerFrame &&
        active_pixel_shader() && diagnostics::IsQuadPartnerMarkerShader(
            active_pixel_shader()->ucode_data_hash())) {
      REXGPU_INFO("[QuadPartnerMarker] frame={} PS={:016X} mode={} lod0={} host_single_sample={}",
          frame_current_, active_pixel_shader()->ucode_data_hash(), face_stage_mode,
          face_stage_lod0, diagnostics::ForceHostSingleSample());
    }
  }
  dirty |= system_constants_.face_stage_mode != face_stage_mode;
  system_constants_.face_stage_mode = face_stage_mode;
  dirty |= system_constants_.face_stage_lod0 != face_stage_lod0;
  system_constants_.face_stage_lod0 = face_stage_lod0;
  static uint64_t face_stage_last_logged_frame = 0;
  if (face_stage_enabled && frame_current_ >= 8680 && frame_current_ <= 8696 &&
      face_stage_last_logged_frame != frame_current_) {
    face_stage_last_logged_frame = frame_current_;
    REXGPU_INFO("[DEBUG-face-stage] frame={} mode={} lod0={}", frame_current_, face_stage_mode,
                face_stage_lod0);
  }
  // [DEBUG-face-stage] Same-run provenance for the sampled texture: dump the
  // guest bytes of the head draw's tf0 binding once, at a labelled frame, so
  // the sampled content can be tied to this exact run and frame instead of an
  // older capture.
  if (face_stage_enabled && frame_current_ == 8696 && memory_ != nullptr &&
      register_file_ != nullptr && active_pixel_shader() != nullptr &&
      active_pixel_shader()->ucode_data_hash() == UINT64_C(0x79A3448304FA0D22)) {
    static bool face_stage_texture_dumped = false;
    if (!face_stage_texture_dumped) {
      face_stage_texture_dumped = true;
      xenos::xe_gpu_texture_fetch_t fetch = register_file_->GetTextureFetch(0);
      uint32_t width_minus_1 = 0, height_minus_1 = 0, depth_or_array_size_minus_1 = 0;
      uint32_t base_page = 0, mip_page = 0, mip_min_level = 0, mip_max_level = 0;
      texture_util::GetSubresourcesFromFetchConstant(
          fetch, &width_minus_1, &height_minus_1, &depth_or_array_size_minus_1, &base_page,
          &mip_page, &mip_min_level, &mip_max_level);
      REXGPU_INFO(
          "[DEBUG-face-stage] tf0 fetch {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} "
          "dimension={} tiled={} packed_mips={} format={} {}x{} base_page={:05X} "
          "mip_page={:05X} mip_max={}",
          fetch.dword_0, fetch.dword_1, fetch.dword_2, fetch.dword_3, fetch.dword_4,
          fetch.dword_5, uint32_t(fetch.dimension), uint32_t(fetch.tiled),
          uint32_t(fetch.packed_mips), uint32_t(fetch.format), width_minus_1 + 1,
          height_minus_1 + 1, base_page, mip_page, mip_max_level);
      // [DEBUG-face-stage] tf2 (light/specular input) and tf14 (projectively
      // addressed lookup built from rcp r3.w) are the remaining unknowns: log
      // their full fetch constants and dump their guest bytes so their identity,
      // format and content are established numerically rather than guessed.
      for (uint32_t extra_index : {2u, 14u}) {
        xenos::xe_gpu_texture_fetch_t extra_fetch = register_file_->GetTextureFetch(extra_index);
        uint32_t e_width_minus_1 = 0, e_height_minus_1 = 0, e_depth_minus_1 = 0;
        uint32_t e_base_page = 0, e_mip_page = 0, e_mip_min = 0, e_mip_max = 0;
        texture_util::GetSubresourcesFromFetchConstant(
            extra_fetch, &e_width_minus_1, &e_height_minus_1, &e_depth_minus_1, &e_base_page,
            &e_mip_page, &e_mip_min, &e_mip_max);
        REXGPU_INFO(
            "[DEBUG-face-stage] tf{} fetch {:08X} {:08X} {:08X} {:08X} {:08X} {:08X} "
            "dimension={} tiled={} packed_mips={} format={} {}x{} base_page={:05X} "
            "mip_page={:05X} mip_min={} mip_max={} clamp={},{} signs={:X}{:X}{:X}{:X}",
            extra_index, extra_fetch.dword_0, extra_fetch.dword_1, extra_fetch.dword_2,
            extra_fetch.dword_3, extra_fetch.dword_4, extra_fetch.dword_5,
            uint32_t(extra_fetch.dimension), uint32_t(extra_fetch.tiled),
            uint32_t(extra_fetch.packed_mips), uint32_t(extra_fetch.format), e_width_minus_1 + 1,
            e_height_minus_1 + 1, e_base_page, e_mip_page, e_mip_min, e_mip_max,
            uint32_t(extra_fetch.clamp_x), uint32_t(extra_fetch.clamp_y),
            uint32_t(extra_fetch.sign_x), uint32_t(extra_fetch.sign_y),
            uint32_t(extra_fetch.sign_z), uint32_t(extra_fetch.sign_w));
        if (e_base_page != 0 && memory_ != nullptr) {
          uint32_t e_base = e_base_page << 12;
          const uint8_t* e_host_data = memory_->TranslatePhysical<const uint8_t*>(e_base);
          // Size from the actual format so the whole level 0 is captured: an RGBA8
          // render target is megabytes, and a truncated dump cannot distinguish
          // "never written" from "written further into the surface".
          uint64_t e_bytes_per_texel_x2 =
              uint64_t(e_width_minus_1 + 1) * uint64_t(e_height_minus_1 + 1) * 2;
          switch (xenos::TextureFormat(uint32_t(extra_fetch.format))) {
            case xenos::TextureFormat::k_8_8_8_8:
            case xenos::TextureFormat::k_8_8_8_8_A:
            case xenos::TextureFormat::k_2_10_10_10:
              e_bytes_per_texel_x2 *= 4;
              break;
            case xenos::TextureFormat::k_8:
            case xenos::TextureFormat::k_8_A:
            case xenos::TextureFormat::k_8_B:
              e_bytes_per_texel_x2 *= 1;
              break;
            default:
              // Block-compressed (DXT1 = 8 bytes per 16 texels, DXT3/5 = 16).
              e_bytes_per_texel_x2 /= 2;
              break;
          }
          uint64_t e_dump_size = e_bytes_per_texel_x2 / 2;
          std::vector<uint8_t> e_snapshot(static_cast<size_t>(e_dump_size), uint8_t(0));
          std::memcpy(e_snapshot.data(), e_host_data, size_t(e_dump_size));
          uint64_t e_nonzero = 0;
          for (uint8_t byte : e_snapshot) {
            if (byte != 0) {
              ++e_nonzero;
            }
          }
          std::string extra_path =
              std::string("/Users/Kushal/Projects/cosmicR/out/face-lod0-ab-20261021/tf") +
              std::to_string(extra_index) + "_guest_bytes_frame8696.bin";
          std::ofstream extra_dump(extra_path, std::ios::binary);
          if (extra_dump.good()) {
            extra_dump.write(reinterpret_cast<const char*>(e_snapshot.data()),
                             std::streamsize(e_dump_size));
          }
          REXGPU_INFO(
              "[DEBUG-face-stage] tf{} dumped {} bytes from base {:08X} nonzero_bytes={} "
              "content={}",
              extra_index, e_dump_size, e_base, e_nonzero,
              e_nonzero == 0 ? "ALL ZERO (nothing resolved into this base)" : "partially written");
        }
      }
      constexpr uint32_t kFaceStageTextureDumpSize = 0x80000;
      if (base_page != 0) {
        uint32_t base = base_page << 12;
        const uint8_t* host_data = memory_->TranslatePhysical<const uint8_t*>(base);
        std::ofstream dump("/Users/Kushal/Projects/cosmicR/out/face-lod0-ab-20261021/"
                           "tf0_guest_bytes_frame8694.bin",
                           std::ios::binary);
        dump.write(reinterpret_cast<const char*>(host_data), kFaceStageTextureDumpSize);
        std::ofstream meta("/Users/Kushal/Projects/cosmicR/out/face-lod0-ab-20261021/"
                           "tf0_guest_bytes_frame8694.txt");
        meta << "frame 8694\n" << "base " << std::hex << base << std::dec << "\n"
             << "size " << kFaceStageTextureDumpSize << "\n" << "width " << (width_minus_1 + 1)
             << "\nheight " << (height_minus_1 + 1) << "\ndepth_or_array "
             << (depth_or_array_size_minus_1 + 1) << "\ntiled " << uint32_t(fetch.tiled)
             << "\npacked_mips " << uint32_t(fetch.packed_mips) << "\nformat "
             << uint32_t(fetch.format) << "\ndimension " << uint32_t(fetch.dimension)
             << "\nmip_min_level " << mip_min_level << "\nmip_max_level " << mip_max_level
             << "\ndwords";
        for (uint32_t const dword : {fetch.dword_0, fetch.dword_1, fetch.dword_2, fetch.dword_3,
                                     fetch.dword_4, fetch.dword_5}) {
          meta << ' ' << std::hex << dword;
        }
        meta << std::dec << "\n";
        REXGPU_INFO("[DEBUG-face-stage] dumped tf0 guest bytes base={:08X} size={}", base,
                    kFaceStageTextureDumpSize);
      }
    }
  }

  // Flags.
  uint32_t flags = 0;
  // Vertex index shader loading.
  if (shader_32bit_index_dma) {
    flags |= SpirvShaderTranslator::kSysFlag_VertexIndexLoad;
  }
  if (primitive_processing_result.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kHostBuiltinForDMA ||
      primitive_processing_result.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kGuestDMA ||
      primitive_processing_result.index_buffer_type ==
          PrimitiveProcessor::ProcessedIndexBufferType::kHostConverted) {
    flags |= SpirvShaderTranslator::kSysFlag_ComputeOrPrimitiveVertexIndexLoad;
    if (vgt_draw_initiator.index_size == xenos::IndexFormat::kInt32) {
      flags |= SpirvShaderTranslator ::kSysFlag_ComputeOrPrimitiveVertexIndexLoad32Bit;
    }
    // Point and rectangle expansion uses primitive restart only in the host
    // built-in strip index buffer; guest indices are not reset-based there.
    bool compute_or_primitive_vertex_index_reset =
        primitive_processing_result.host_primitive_reset_enabled &&
        primitive_processing_result.host_vertex_shader_type !=
            Shader::HostVertexShaderType::kPointListAsTriangleStrip &&
        primitive_processing_result.host_vertex_shader_type !=
            Shader::HostVertexShaderType::kRectangleListAsTriangleStrip;
    if (compute_or_primitive_vertex_index_reset) {
      flags |= SpirvShaderTranslator::kSysFlag_ComputeOrPrimitiveVertexIndexReset;
    }
  }
  if (primitive_processing_result.host_vertex_shader_type ==
      Shader::HostVertexShaderType::kTriangleDomainPatchIndexed) {
    flags |= SpirvShaderTranslator::kSysFlag_ComputeMemExportPatchIndexInRegister1;
  }
  if (primitive_processing_result.host_vertex_shader_type ==
      Shader::HostVertexShaderType::kTriangleDomainCPIndexed) {
    flags |= SpirvShaderTranslator::kSysFlag_ComputeMemExportTriangleCPIndexed;
  } else if (primitive_processing_result.host_vertex_shader_type ==
             Shader::HostVertexShaderType::kQuadDomainCPIndexed) {
    flags |= SpirvShaderTranslator::kSysFlag_ComputeMemExportQuadCPIndexed;
  }
  if (primitive_processing_result.tessellation_mode == xenos::TessellationMode::kAdaptive &&
      (primitive_processing_result.host_vertex_shader_type ==
           Shader::HostVertexShaderType::kTriangleDomainPatchIndexed ||
       primitive_processing_result.host_vertex_shader_type ==
           Shader::HostVertexShaderType::kQuadDomainPatchIndexed)) {
    flags |= SpirvShaderTranslator::kSysFlag_ComputeMemExportPatchIndexFromInvocation;
  }
  // W0 division control.
  // http://www.x.org/docs/AMD/old/evergreen_3D_registers_v2.pdf
  // 8: VTX_XY_FMT = true: the incoming XY have already been multiplied by 1/W0.
  //               = false: multiply the X, Y coordinates by 1/W0.
  // 9: VTX_Z_FMT = true: the incoming Z has already been multiplied by 1/W0.
  //              = false: multiply the Z coordinate by 1/W0.
  // 10: VTX_W0_FMT = true: the incoming W0 is not 1/W0. Perform the reciprocal
  //                        to get 1/W0.
  if (pa_cl_vte_cntl.vtx_xy_fmt) {
    flags |= SpirvShaderTranslator::kSysFlag_XYDividedByW;
  }
  if (pa_cl_vte_cntl.vtx_z_fmt) {
    flags |= SpirvShaderTranslator::kSysFlag_ZDividedByW;
  }
  if (pa_cl_vte_cntl.vtx_w0_fmt) {
    flags |= SpirvShaderTranslator::kSysFlag_WNotReciprocal;
  }
  // Whether the primitive is polygonal, and gl_FrontFacing matters.
  if (primitive_polygonal) {
    flags |= SpirvShaderTranslator::kSysFlag_PrimitivePolygonal;
  }
  // Primitive type.
  if (draw_util::IsPrimitiveLine(regs)) {
    flags |= SpirvShaderTranslator::kSysFlag_PrimitiveLine;
  }
  // MSAA sample count.
  xenos::MsaaSamples effective_msaa_samples = rb_surface_info.msaa_samples;
  if (REXCVAR_GET(vulkan_alias_4x_depth_to_1x) && !normalized_color_mask &&
      effective_msaa_samples == xenos::MsaaSamples::k4X) {
    effective_msaa_samples = xenos::MsaaSamples::k1X;
  }
  flags |= uint32_t(effective_msaa_samples)
           << SpirvShaderTranslator::kSysFlag_MsaaSamples_Shift;
  // Depth format.
  if (rb_depth_info.depth_format == xenos::DepthRenderTargetFormat::kD24FS8) {
    flags |= SpirvShaderTranslator::kSysFlag_DepthFloat24;
  }
  // Alpha test.
  xenos::CompareFunction alpha_test_function = rb_colorcontrol.alpha_test_enable
                                                   ? rb_colorcontrol.alpha_func
                                                   : xenos::CompareFunction::kAlways;
  flags |= uint32_t(alpha_test_function) << SpirvShaderTranslator::kSysFlag_AlphaPassIfLess_Shift;
  // Gamma writing.
  if (!render_target_cache_->gamma_render_target_as_unorm16()) {
    // Keep parity with D3D12: gamma targets in this path are converted via
    // explicit Xenos PWL gamma logic in shaders rather than via host sRGB
    // attachment conversion semantics.
    for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
      if (color_infos[i].color_format == xenos::ColorRenderTargetFormat::k_8_8_8_8_GAMMA) {
        flags |= SpirvShaderTranslator::kSysFlag_ConvertColor0ToGamma << i;
      }
    }
  }
  if (edram_fragment_shader_interlock && depth_stencil_enabled) {
    flags |= SpirvShaderTranslator::kSysFlag_FSIDepthStencil;
    if (normalized_depth_control.z_enable) {
      flags |= uint32_t(normalized_depth_control.zfunc)
               << SpirvShaderTranslator::kSysFlag_FSIDepthPassIfLess_Shift;
      if (normalized_depth_control.z_write_enable) {
        flags |= SpirvShaderTranslator::kSysFlag_FSIDepthWrite;
      }
    } else {
      // In case stencil is used without depth testing - always pass, and
      // don't modify the stored depth.
      flags |= SpirvShaderTranslator::kSysFlag_FSIDepthPassIfLess |
               SpirvShaderTranslator::kSysFlag_FSIDepthPassIfEqual |
               SpirvShaderTranslator::kSysFlag_FSIDepthPassIfGreater;
    }
    if (normalized_depth_control.stencil_enable) {
      flags |= SpirvShaderTranslator::kSysFlag_FSIStencilTest;
    }
    // Hint - if not applicable to the shader, will not have effect.
    if (alpha_test_function == xenos::CompareFunction::kAlways &&
        !rb_colorcontrol.alpha_to_mask_enable) {
      flags |= SpirvShaderTranslator::kSysFlag_FSIDepthStencilEarlyWrite;
    }
  }
  dirty |= system_constants_.flags != flags;
  system_constants_.flags = flags;

  // Index buffer address for loading in the shaders.
  if (flags & (SpirvShaderTranslator::kSysFlag_VertexIndexLoad |
               SpirvShaderTranslator::kSysFlag_ComputeOrPrimitiveVertexIndexLoad)) {
    dirty |=
        system_constants_.vertex_index_load_address != primitive_processing_result.guest_index_base;
    system_constants_.vertex_index_load_address = primitive_processing_result.guest_index_base;
  }

  // Primitive reset index for shader-side index loading.
  uint32_t vertex_index_reset =
      (flags & SpirvShaderTranslator::kSysFlag_ComputeOrPrimitiveVertexIndexReset)
          ? regs.Get<reg::VGT_MULTI_PRIM_IB_RESET_INDX>().reset_indx
          : 0;
  dirty |= system_constants_.vertex_index_reset != vertex_index_reset;
  system_constants_.vertex_index_reset = vertex_index_reset;
  dirty |= system_constants_.compute_memexport_vertex_count != compute_memexport_vertex_count;
  system_constants_.compute_memexport_vertex_count = compute_memexport_vertex_count;

  // Index or tessellation edge factor buffer endianness.
  xenos::Endian guest_index_endian = vgt_dma_size.swap_mode;
  if (vgt_draw_initiator.index_size == xenos::IndexFormat::kInt16 &&
      guest_index_endian != xenos::Endian::kNone && guest_index_endian != xenos::Endian::k8in16) {
    guest_index_endian =
        guest_index_endian == xenos::Endian::k8in32 ? xenos::Endian::k8in16 : xenos::Endian::kNone;
  }
  xenos::Endian index_endian =
      (flags & SpirvShaderTranslator::kSysFlag_ComputeOrPrimitiveVertexIndexLoad)
          ? guest_index_endian
          : primitive_processing_result.host_shader_index_endian;
  dirty |= system_constants_.vertex_index_endian != index_endian;
  system_constants_.vertex_index_endian = index_endian;

  dirty |= system_constants_.line_loop_closing_index !=
           primitive_processing_result.line_loop_closing_index;
  system_constants_.line_loop_closing_index = primitive_processing_result.line_loop_closing_index;

  // Vertex index offset.
  dirty |= system_constants_.vertex_base_index != vgt_indx_offset;
  system_constants_.vertex_base_index = vgt_indx_offset;

  // Vertex index range.
  dirty |= system_constants_.vertex_index_min != vgt_min_vtx_indx;
  dirty |= system_constants_.vertex_index_max != vgt_max_vtx_indx;
  system_constants_.vertex_index_min = vgt_min_vtx_indx;
  system_constants_.vertex_index_max = vgt_max_vtx_indx;

  // User clip planes (UCP_ENA_#), when not CLIP_DISABLE.
  // The shader knows only the total count - tightly packing the user clip
  // planes that are actually used.
  if (!pa_cl_clip_cntl.clip_disable) {
    float* user_clip_plane_write_ptr = system_constants_.user_clip_planes[0];
    uint32_t user_clip_planes_remaining = pa_cl_clip_cntl.ucp_ena;
    uint32_t user_clip_plane_index;
    while (rex::bit_scan_forward(user_clip_planes_remaining, &user_clip_plane_index)) {
      user_clip_planes_remaining &= ~(UINT32_C(1) << user_clip_plane_index);
      const void* user_clip_plane_regs =
          &regs[XE_GPU_REG_PA_CL_UCP_0_X + user_clip_plane_index * 4];
      if (std::memcmp(user_clip_plane_write_ptr, user_clip_plane_regs, 4 * sizeof(float))) {
        dirty = true;
        std::memcpy(user_clip_plane_write_ptr, user_clip_plane_regs, 4 * sizeof(float));
      }
      user_clip_plane_write_ptr += 4;
    }
  }

  // Tessellation factor range, plus 1.0 according to
  // https://www.slideshare.net/blackdevilvikas/next-generation-graphics-programming-on-xbox-360.
  float tessellation_factor_min = regs.Get<float>(XE_GPU_REG_VGT_HOS_MIN_TESS_LEVEL) + 1.0f;
  float tessellation_factor_max = regs.Get<float>(XE_GPU_REG_VGT_HOS_MAX_TESS_LEVEL) + 1.0f;
  dirty |= system_constants_.tessellation_factor_range_min != tessellation_factor_min;
  dirty |= system_constants_.tessellation_factor_range_max != tessellation_factor_max;
  system_constants_.tessellation_factor_range_min = tessellation_factor_min;
  system_constants_.tessellation_factor_range_max = tessellation_factor_max;

  // Conversion to host normalized device coordinates.
  for (uint32_t i = 0; i < 3; ++i) {
    dirty |= system_constants_.ndc_scale[i] != viewport_info.ndc_scale[i];
    dirty |= system_constants_.ndc_offset[i] != viewport_info.ndc_offset[i];
    system_constants_.ndc_scale[i] = viewport_info.ndc_scale[i];
    system_constants_.ndc_offset[i] = viewport_info.ndc_offset[i];
  }

  // Point size.
  if (vgt_draw_initiator.prim_type == xenos::PrimitiveType::kPointList) {
    auto pa_su_point_minmax = regs.Get<reg::PA_SU_POINT_MINMAX>();
    auto pa_su_point_size = regs.Get<reg::PA_SU_POINT_SIZE>();
    float point_vertex_diameter_min = float(pa_su_point_minmax.min_size) * (2.0f / 16.0f);
    float point_vertex_diameter_max = float(pa_su_point_minmax.max_size) * (2.0f / 16.0f);
    float point_constant_diameter_x = float(pa_su_point_size.width) * (2.0f / 16.0f);
    float point_constant_diameter_y = float(pa_su_point_size.height) * (2.0f / 16.0f);
    dirty |= system_constants_.point_vertex_diameter_min != point_vertex_diameter_min;
    dirty |= system_constants_.point_vertex_diameter_max != point_vertex_diameter_max;
    dirty |= system_constants_.point_constant_diameter[0] != point_constant_diameter_x;
    dirty |= system_constants_.point_constant_diameter[1] != point_constant_diameter_y;
    system_constants_.point_vertex_diameter_min = point_vertex_diameter_min;
    system_constants_.point_vertex_diameter_max = point_vertex_diameter_max;
    system_constants_.point_constant_diameter[0] = point_constant_diameter_x;
    system_constants_.point_constant_diameter[1] = point_constant_diameter_y;
    // 2 because 1 in the NDC is half of the viewport's axis, 0.5 for diameter
    // to radius conversion to avoid multiplying the per-vertex diameter by an
    // additional constant in the shader.
    float point_screen_diameter_to_ndc_radius_x =
        (/* 0.5f * 2.0f * */ float(draw_resolution_scale_x)) /
        std::max(viewport_info.xy_extent[0], uint32_t(1));
    float point_screen_diameter_to_ndc_radius_y =
        (/* 0.5f * 2.0f * */ float(draw_resolution_scale_y)) /
        std::max(viewport_info.xy_extent[1], uint32_t(1));
    dirty |= system_constants_.point_screen_diameter_to_ndc_radius[0] !=
             point_screen_diameter_to_ndc_radius_x;
    dirty |= system_constants_.point_screen_diameter_to_ndc_radius[1] !=
             point_screen_diameter_to_ndc_radius_y;
    system_constants_.point_screen_diameter_to_ndc_radius[0] =
        point_screen_diameter_to_ndc_radius_x;
    system_constants_.point_screen_diameter_to_ndc_radius[1] =
        point_screen_diameter_to_ndc_radius_y;
  }

  // Texture signedness / gamma.
  uint32_t textures_resolution_scaled = 0;
  {
    uint32_t textures_remaining = used_texture_mask;
    uint32_t texture_index;
    while (rex::bit_scan_forward(textures_remaining, &texture_index)) {
      textures_remaining &= ~(UINT32_C(1) << texture_index);
      uint32_t& texture_signs_uint = system_constants_.texture_swizzled_signs[texture_index >> 2];
      uint32_t texture_signs_shift = 8 * (texture_index & 3);
      uint8_t texture_signs = texture_cache_->GetActiveTextureSwizzledSigns(texture_index);
      uint32_t texture_signs_shifted = uint32_t(texture_signs) << texture_signs_shift;
      uint32_t texture_signs_mask = ((UINT32_C(1) << 8) - 1) << texture_signs_shift;
      dirty |= (texture_signs_uint & texture_signs_mask) != texture_signs_shifted;
      texture_signs_uint = (texture_signs_uint & ~texture_signs_mask) | texture_signs_shifted;
      textures_resolution_scaled |=
          uint32_t(texture_cache_->IsActiveTextureResolutionScaled(texture_index)) << texture_index;
    }
  }
  dirty |= system_constants_.textures_resolution_scaled != textures_resolution_scaled;
  system_constants_.textures_resolution_scaled = textures_resolution_scaled;

  // Texture host swizzle in the shader.
  if (!GetVulkanDevice()->properties().imageViewFormatSwizzle) {
    uint32_t textures_remaining = used_texture_mask;
    uint32_t texture_index;
    while (rex::bit_scan_forward(textures_remaining, &texture_index)) {
      textures_remaining &= ~(UINT32_C(1) << texture_index);
      uint32_t& texture_swizzles_uint = system_constants_.texture_swizzles[texture_index >> 1];
      uint32_t texture_swizzle_shift = 12 * (texture_index & 1);
      uint32_t texture_swizzle = texture_cache_->GetActiveTextureHostSwizzle(texture_index);
      uint32_t texture_swizzle_shifted = uint32_t(texture_swizzle) << texture_swizzle_shift;
      uint32_t texture_swizzle_mask = ((UINT32_C(1) << 12) - 1) << texture_swizzle_shift;
      dirty |= (texture_swizzles_uint & texture_swizzle_mask) != texture_swizzle_shifted;
      texture_swizzles_uint =
          (texture_swizzles_uint & ~texture_swizzle_mask) | texture_swizzle_shifted;
    }
  }

  // Alpha test.
  dirty |= system_constants_.alpha_test_reference != rb_alpha_ref;
  system_constants_.alpha_test_reference = rb_alpha_ref;
  uint32_t alpha_to_mask =
      rb_colorcontrol.alpha_to_mask_enable ? (rb_colorcontrol.value >> 24) | (UINT32_C(1) << 8) : 0;
  dirty |= system_constants_.alpha_to_mask != alpha_to_mask;
  system_constants_.alpha_to_mask = alpha_to_mask;

  uint32_t edram_tile_dwords_scaled = xenos::kEdramTileWidthSamples *
                                      xenos::kEdramTileHeightSamples *
                                      (draw_resolution_scale_x * draw_resolution_scale_y);

  // EDRAM pitch for FSI render target writing.
  if (edram_fragment_shader_interlock) {
    // Align, then multiply by 32bpp tile size in dwords.
    uint32_t edram_32bpp_tile_pitch_dwords_scaled =
        ((rb_surface_info.surface_pitch *
          (rb_surface_info.msaa_samples >= xenos::MsaaSamples::k4X ? 2 : 1)) +
         (xenos::kEdramTileWidthSamples - 1)) /
        xenos::kEdramTileWidthSamples * edram_tile_dwords_scaled;
    dirty |= system_constants_.edram_32bpp_tile_pitch_dwords_scaled !=
             edram_32bpp_tile_pitch_dwords_scaled;
    system_constants_.edram_32bpp_tile_pitch_dwords_scaled = edram_32bpp_tile_pitch_dwords_scaled;
  }

  // Color exponent bias and FSI render target writing.
  for (uint32_t i = 0; i < xenos::kMaxColorRenderTargets; ++i) {
    reg::RB_COLOR_INFO color_info = color_infos[i];
    // Exponent bias is in bits 20:25 of RB_COLOR_INFO.
    int32_t color_exp_bias = color_info.color_exp_bias;
    if (render_target_cache_->GetPath() == RenderTargetCache::Path::kHostRenderTargets &&
        (color_info.color_format == xenos::ColorRenderTargetFormat::k_16_16 &&
             !render_target_cache_->IsFixedRG16TruncatedToMinus1To1() ||
         color_info.color_format == xenos::ColorRenderTargetFormat::k_16_16_16_16 &&
             !render_target_cache_->IsFixedRGBA16TruncatedToMinus1To1())) {
      // Remap from -32...32 to -1...1 by dividing the output values by 32,
      // losing blending correctness, but getting the full range.
      color_exp_bias -= 5;
    }
    float color_exp_bias_scale;
    *reinterpret_cast<int32_t*>(&color_exp_bias_scale) =
        UINT32_C(0x3F800000) + (color_exp_bias << 23);
    dirty |= system_constants_.color_exp_bias[i] != color_exp_bias_scale;
    system_constants_.color_exp_bias[i] = color_exp_bias_scale;
    if (edram_fragment_shader_interlock) {
      dirty |= system_constants_.edram_rt_keep_mask[i][0] != rt_keep_masks[i][0];
      system_constants_.edram_rt_keep_mask[i][0] = rt_keep_masks[i][0];
      dirty |= system_constants_.edram_rt_keep_mask[i][1] != rt_keep_masks[i][1];
      system_constants_.edram_rt_keep_mask[i][1] = rt_keep_masks[i][1];
      if (rt_keep_masks[i][0] != UINT32_MAX || rt_keep_masks[i][1] != UINT32_MAX) {
        uint32_t rt_base_dwords_scaled = color_info.color_base * edram_tile_dwords_scaled;
        dirty |= system_constants_.edram_rt_base_dwords_scaled[i] != rt_base_dwords_scaled;
        system_constants_.edram_rt_base_dwords_scaled[i] = rt_base_dwords_scaled;
        uint32_t format_flags = RenderTargetCache::AddPSIColorFormatFlags(color_info.color_format);
        dirty |= system_constants_.edram_rt_format_flags[i] != format_flags;
        system_constants_.edram_rt_format_flags[i] = format_flags;
        uint32_t blend_factors_ops =
            regs[reg::RB_BLENDCONTROL::rt_register_indices[i]] & 0x1FFF1FFF;
        dirty |= system_constants_.edram_rt_blend_factors_ops[i] != blend_factors_ops;
        system_constants_.edram_rt_blend_factors_ops[i] = blend_factors_ops;
        // Can't do float comparisons here because NaNs would result in always
        // setting the dirty flag.
        dirty |=
            std::memcmp(system_constants_.edram_rt_clamp[i], rt_clamp[i], 4 * sizeof(float)) != 0;
        std::memcpy(system_constants_.edram_rt_clamp[i], rt_clamp[i], 4 * sizeof(float));
      }
    }
  }

  if (edram_fragment_shader_interlock) {
    uint32_t depth_base_dwords_scaled = rb_depth_info.depth_base * edram_tile_dwords_scaled;
    dirty |= system_constants_.edram_depth_base_dwords_scaled != depth_base_dwords_scaled;
    system_constants_.edram_depth_base_dwords_scaled = depth_base_dwords_scaled;

    // For non-polygons, front polygon offset is used, and it's enabled if
    // POLY_OFFSET_PARA_ENABLED is set, for polygons, separate front and back
    // are used.
    float poly_offset_front_scale = 0.0f, poly_offset_front_offset = 0.0f;
    float poly_offset_back_scale = 0.0f, poly_offset_back_offset = 0.0f;
    if (primitive_polygonal) {
      if (pa_su_sc_mode_cntl.poly_offset_front_enable) {
        poly_offset_front_scale = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE);
        poly_offset_front_offset = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET);
      }
      if (pa_su_sc_mode_cntl.poly_offset_back_enable) {
        poly_offset_back_scale = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_SCALE);
        poly_offset_back_offset = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_BACK_OFFSET);
      }
    } else {
      if (pa_su_sc_mode_cntl.poly_offset_para_enable) {
        poly_offset_front_scale = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_SCALE);
        poly_offset_front_offset = regs.Get<float>(XE_GPU_REG_PA_SU_POLY_OFFSET_FRONT_OFFSET);
        poly_offset_back_scale = poly_offset_front_scale;
        poly_offset_back_offset = poly_offset_front_offset;
      }
    }
    // With non-square resolution scaling, make sure the worst-case impact is
    // reverted (slope only along the scaled axis), thus max. More bias is
    // better than less bias, because less bias means Z fighting with the
    // background is more likely.
    float poly_offset_scale_factor = xenos::kPolygonOffsetScaleSubpixelUnit *
                                     std::max(draw_resolution_scale_x, draw_resolution_scale_y);
    poly_offset_front_scale *= poly_offset_scale_factor;
    poly_offset_back_scale *= poly_offset_scale_factor;
    dirty |= system_constants_.edram_poly_offset_front_scale != poly_offset_front_scale;
    system_constants_.edram_poly_offset_front_scale = poly_offset_front_scale;
    dirty |= system_constants_.edram_poly_offset_front_offset != poly_offset_front_offset;
    system_constants_.edram_poly_offset_front_offset = poly_offset_front_offset;
    dirty |= system_constants_.edram_poly_offset_back_scale != poly_offset_back_scale;
    system_constants_.edram_poly_offset_back_scale = poly_offset_back_scale;
    dirty |= system_constants_.edram_poly_offset_back_offset != poly_offset_back_offset;
    system_constants_.edram_poly_offset_back_offset = poly_offset_back_offset;

    if (depth_stencil_enabled && normalized_depth_control.stencil_enable) {
      uint32_t stencil_front_reference_masks = rb_stencilrefmask.value & 0xFFFFFF;
      dirty |=
          system_constants_.edram_stencil_front_reference_masks != stencil_front_reference_masks;
      system_constants_.edram_stencil_front_reference_masks = stencil_front_reference_masks;
      uint32_t stencil_func_ops = (normalized_depth_control.value >> 8) & ((1 << 12) - 1);
      dirty |= system_constants_.edram_stencil_front_func_ops != stencil_func_ops;
      system_constants_.edram_stencil_front_func_ops = stencil_func_ops;

      if (primitive_polygonal && normalized_depth_control.backface_enable) {
        uint32_t stencil_back_reference_masks = rb_stencilrefmask_bf.value & 0xFFFFFF;
        dirty |=
            system_constants_.edram_stencil_back_reference_masks != stencil_back_reference_masks;
        system_constants_.edram_stencil_back_reference_masks = stencil_back_reference_masks;
        uint32_t stencil_func_ops_bf = (normalized_depth_control.value >> 20) & ((1 << 12) - 1);
        dirty |= system_constants_.edram_stencil_back_func_ops != stencil_func_ops_bf;
        system_constants_.edram_stencil_back_func_ops = stencil_func_ops_bf;
      } else {
        dirty |= std::memcmp(system_constants_.edram_stencil_back,
                             system_constants_.edram_stencil_front, 2 * sizeof(uint32_t)) != 0;
        std::memcpy(system_constants_.edram_stencil_back, system_constants_.edram_stencil_front,
                    2 * sizeof(uint32_t));
      }
    }

    dirty |= system_constants_.edram_blend_constant[0] != regs.Get<float>(XE_GPU_REG_RB_BLEND_RED);
    system_constants_.edram_blend_constant[0] = regs.Get<float>(XE_GPU_REG_RB_BLEND_RED);
    dirty |=
        system_constants_.edram_blend_constant[1] != regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN);
    system_constants_.edram_blend_constant[1] = regs.Get<float>(XE_GPU_REG_RB_BLEND_GREEN);
    dirty |= system_constants_.edram_blend_constant[2] != regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE);
    system_constants_.edram_blend_constant[2] = regs.Get<float>(XE_GPU_REG_RB_BLEND_BLUE);
    dirty |=
        system_constants_.edram_blend_constant[3] != regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA);
    system_constants_.edram_blend_constant[3] = regs.Get<float>(XE_GPU_REG_RB_BLEND_ALPHA);
  }

  if (dirty) {
    current_constant_buffers_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem);
  }
}

bool VulkanCommandProcessor::RequestSharedMemoryRange(uint32_t start, uint32_t length) {
  return shared_memory_->RequestRange(start, length);
}

bool VulkanCommandProcessor::UpdateBindings(const VulkanShader* vertex_shader,
                                            const VulkanShader* pixel_shader) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  const RegisterFile& regs = *register_file_;

  const ui::vulkan::VulkanDevice* const vulkan_device = GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  // Invalidate constant buffers and descriptors for changed data.

  // Float constants.
  // These are the constant base addresses/ranges for shaders.
  // We have these hardcoded right now cause nothing seems to differ on the Xbox
  // 360 (however, OpenGL ES on Adreno 200 on Android has different ranges).
  assert_true(regs[XE_GPU_REG_SQ_VS_CONST] == 0x000FF000 ||
              regs[XE_GPU_REG_SQ_VS_CONST] == 0x00000000);
  assert_true(regs[XE_GPU_REG_SQ_PS_CONST] == 0x000FF100 ||
              regs[XE_GPU_REG_SQ_PS_CONST] == 0x00000000);
  // Check if the float constant layout is still the same and get the counts.
  const Shader::ConstantRegisterMap& float_constant_map_vertex =
      vertex_shader->constant_register_map();
  uint32_t float_constant_count_vertex = float_constant_map_vertex.float_count;
  for (uint32_t i = 0; i < 4; ++i) {
    if (current_float_constant_map_vertex_[i] != float_constant_map_vertex.float_bitmap[i]) {
      current_float_constant_map_vertex_[i] = float_constant_map_vertex.float_bitmap[i];
      // If no float constants at all, any buffer can be reused for them, so not
      // invalidating.
      if (float_constant_count_vertex) {
        current_constant_buffers_up_to_date_ &=
            ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex);
      }
    }
  }
  uint32_t float_constant_count_pixel = 0;
  if (pixel_shader != nullptr) {
    const Shader::ConstantRegisterMap& float_constant_map_pixel =
        pixel_shader->constant_register_map();
    float_constant_count_pixel = float_constant_map_pixel.float_count;
    for (uint32_t i = 0; i < 4; ++i) {
      if (current_float_constant_map_pixel_[i] != float_constant_map_pixel.float_bitmap[i]) {
        current_float_constant_map_pixel_[i] = float_constant_map_pixel.float_bitmap[i];
        if (float_constant_count_pixel) {
          current_constant_buffers_up_to_date_ &=
              ~(UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatPixel);
        }
      }
    }
  } else {
    std::memset(current_float_constant_map_pixel_, 0, sizeof(current_float_constant_map_pixel_));
  }

  // Write the new constant buffers.
  constexpr uint32_t kAllConstantBuffersMask =
      (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferCount) - 1;
  assert_zero(current_constant_buffers_up_to_date_ & ~kAllConstantBuffersMask);
  if ((current_constant_buffers_up_to_date_ & kAllConstantBuffersMask) != kAllConstantBuffersMask) {
    size_t uniform_buffer_alignment =
        size_t(vulkan_device->properties().minUniformBufferOffsetAlignment);

    // Sub-allocate from this frame's ring partition. The ring never hands out a
    // different VkBuffer, so the dynamic offsets below always address the
    // buffer they were written into.
    auto allocate_constant_ring = [&](size_t size) -> std::pair<uint8_t*, uint32_t> {
      size_t aligned_size = rex::align(size, uniform_buffer_alignment);
      if (constant_ring_offset_ + aligned_size >
          constant_ring_frame_base_ + kConstantRingFramePartitionSize) {
        // Wrapping mid-frame would overwrite constants still referenced by draws
        // already recorded for this frame, so never do it silently.
        REXGPU_ERROR("Constant ring partition exhausted within a frame");
        constant_ring_offset_ = constant_ring_frame_base_;
      }
      uint32_t offset = uint32_t(constant_ring_offset_);
      constant_ring_offset_ += aligned_size;
      return {constant_ring_mapping_ + offset, offset};
    };

    // System constants.
    if (!(current_constant_buffers_up_to_date_ &
          (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferSystem))) {
      auto [ptr, offset] = allocate_constant_ring(sizeof(SpirvShaderTranslator::SystemConstants));
      std::memcpy(ptr, &system_constants_, sizeof(SpirvShaderTranslator::SystemConstants));
      current_constant_buffer_dynamic_offsets_[SpirvShaderTranslator::kConstantBufferSystem] =
          offset;
      current_constant_buffers_up_to_date_ |= UINT32_C(1)
                                              << SpirvShaderTranslator::kConstantBufferSystem;
    }
    // Vertex shader float constants.
    if (!(current_constant_buffers_up_to_date_ &
          (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatVertex))) {
      // Even if the shader doesn't need any float constants, a valid binding
      // must still be provided (the pipeline layout always has float constants,
      // for both the vertex shader and the pixel shader), so if the first draw
      // in the frame doesn't have float constants at all, still reserve space.
      // The full 256 vec4 range is reserved because the descriptor binding
      // covers all of it regardless of how many constants this draw writes.
      constexpr size_t kMaxFloatConstantsSize = sizeof(float) * 4 * 256;
      auto [mapping, offset] = allocate_constant_ring(kMaxFloatConstantsSize);
      for (uint32_t i = 0; i < 4; ++i) {
        uint64_t float_constant_map_entry = current_float_constant_map_vertex_[i];
        uint32_t float_constant_index;
        while (rex::bit_scan_forward(float_constant_map_entry, &float_constant_index)) {
          float_constant_map_entry &= ~(1ull << float_constant_index);
          std::memcpy(
              mapping,
              &regs[XE_GPU_REG_SHADER_CONSTANT_000_X + (i << 8) + (float_constant_index << 2)],
              sizeof(float) * 4);
          mapping += sizeof(float) * 4;
        }
      }
      current_constant_buffer_dynamic_offsets_[SpirvShaderTranslator::kConstantBufferFloatVertex] =
          offset;
      current_constant_buffers_up_to_date_ |= UINT32_C(1)
                                              << SpirvShaderTranslator::kConstantBufferFloatVertex;
    }
    // Pixel shader float constants.
    if (!(current_constant_buffers_up_to_date_ &
          (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFloatPixel))) {
      constexpr size_t kMaxFloatConstantsSize = sizeof(float) * 4 * 256;
      auto [mapping, offset] = allocate_constant_ring(kMaxFloatConstantsSize);
      for (uint32_t i = 0; i < 4; ++i) {
        uint64_t float_constant_map_entry = current_float_constant_map_pixel_[i];
        uint32_t float_constant_index;
        while (rex::bit_scan_forward(float_constant_map_entry, &float_constant_index)) {
          float_constant_map_entry &= ~(1ull << float_constant_index);
          std::memcpy(
              mapping,
              &regs[PixelFloatConstantRegister((i << 6) + float_constant_index, 0)],
              sizeof(float) * 4);
          mapping += sizeof(float) * 4;
        }
      }
      current_constant_buffer_dynamic_offsets_[SpirvShaderTranslator::kConstantBufferFloatPixel] =
          offset;
      current_constant_buffers_up_to_date_ |= UINT32_C(1)
                                              << SpirvShaderTranslator::kConstantBufferFloatPixel;
    }
    // Bool and loop constants.
    if (!(current_constant_buffers_up_to_date_ &
          (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferBoolLoop))) {
      constexpr size_t kBoolLoopConstantsSize = sizeof(uint32_t) * (8 + 32);
      auto [ptr, offset] = allocate_constant_ring(kBoolLoopConstantsSize);
      std::memcpy(ptr, &regs[XE_GPU_REG_SHADER_CONSTANT_BOOL_000_031], kBoolLoopConstantsSize);
      current_constant_buffer_dynamic_offsets_[SpirvShaderTranslator::kConstantBufferBoolLoop] =
          offset;
      current_constant_buffers_up_to_date_ |= UINT32_C(1)
                                              << SpirvShaderTranslator::kConstantBufferBoolLoop;
    }
    // Fetch constants.
    if (!(current_constant_buffers_up_to_date_ &
          (UINT32_C(1) << SpirvShaderTranslator::kConstantBufferFetch))) {
      constexpr size_t kFetchConstantsSize = sizeof(uint32_t) * 6 * 32;
      auto [ptr, offset] = allocate_constant_ring(kFetchConstantsSize);
      std::memcpy(ptr, &regs[XE_GPU_REG_SHADER_CONSTANT_FETCH_00_0], kFetchConstantsSize);
      current_constant_buffer_dynamic_offsets_[SpirvShaderTranslator::kConstantBufferFetch] = offset;
      current_constant_buffers_up_to_date_ |= UINT32_C(1)
                                              << SpirvShaderTranslator::kConstantBufferFetch;
    }
  }

  // Host-only capture of the exact uploaded UBO slices for the synthetic
  // translated-shader fixture. No guest or fragment storage writes are added.
  static const bool capture_translated_bindings = [] {
    std::ifstream gate(diagnostics::kQuadPartnerMarkerPath);
    for (std::string line; std::getline(gate, line);) {
      if (line == "capture_translated_bindings=1") return true;
    }
    return false;
  }();
  static bool translated_bindings_captured = false;
  if (capture_translated_bindings && !translated_bindings_captured && frame_current_ == 6000 &&
      pixel_shader && diagnostics::IsQuadPartnerMarkerShader(pixel_shader->ucode_data_hash())) {
    draw_util::Scissor scissor{};
    draw_util::GetScissor(regs, scissor);
    if (scissor.extent[1] == 336) {
      constexpr const char* directory =
          "/Users/Kushal/Projects/cosmicR/out/quad-partner-isolation-20261026/translated/input/";
      bool saved = true;
      const auto save = [&](const char* name, uint32_t binding, size_t size) {
        const std::string path = std::string(directory) + name;
        FILE* file = std::fopen(path.c_str(), "wb");
        if (!file) return false;
        const bool written = std::fwrite(constant_ring_mapping_ +
            current_constant_buffer_dynamic_offsets_[binding], 1, size, file) == size;
        return std::fclose(file) == 0 && written;
      };
      saved &= save("system.bin", SpirvShaderTranslator::kConstantBufferSystem,
                    sizeof(SpirvShaderTranslator::SystemConstants));
      saved &= save("float.bin", SpirvShaderTranslator::kConstantBufferFloatPixel,
                    size_t(float_constant_count_pixel) * 16);
      saved &= save("bool-loop.bin", SpirvShaderTranslator::kConstantBufferBoolLoop, 160);
      saved &= save("fetch.bin", SpirvShaderTranslator::kConstantBufferFetch, 768);
      const std::string metadata = fmt::format(
          "{{\"frame\":6000,\"draw\":{},\"PS\":\"{:016X}\",\"scissor_height\":{},"
          "\"float_bytes\":{},\"mode\":{},\"saved\":{}}}\n",
          diagnostic_snapshot_draw_ordinal_, pixel_shader->ucode_data_hash(), scissor.extent[1],
          float_constant_count_pixel * 16, system_constants_.face_stage_mode, saved);
      std::ofstream(std::string(directory) + "capture.json") << metadata;
      translated_bindings_captured = saved;
      REXGPU_INFO("[TranslatedProbeBindings] {}", metadata);
    }
  }

  // Textures and samplers.
  const std::vector<VulkanShader::SamplerBinding>& samplers_vertex =
      vertex_shader->GetSamplerBindingsAfterTranslation();
  const std::vector<VulkanShader::TextureBinding>& textures_vertex =
      vertex_shader->GetTextureBindingsAfterTranslation();
  uint32_t sampler_count_vertex = uint32_t(samplers_vertex.size());
  uint32_t texture_count_vertex = uint32_t(textures_vertex.size());
  const std::vector<VulkanShader::SamplerBinding>* samplers_pixel;
  const std::vector<VulkanShader::TextureBinding>* textures_pixel;
  uint32_t sampler_count_pixel, texture_count_pixel;
  if (pixel_shader) {
    samplers_pixel = &pixel_shader->GetSamplerBindingsAfterTranslation();
    textures_pixel = &pixel_shader->GetTextureBindingsAfterTranslation();
    sampler_count_pixel = uint32_t(samplers_pixel->size());
    texture_count_pixel = uint32_t(textures_pixel->size());
  } else {
    samplers_pixel = nullptr;
    textures_pixel = nullptr;
    sampler_count_pixel = 0;
    texture_count_pixel = 0;
  }
  if (FlickerDiagEnabled() && texture_count_pixel) {
    // [flicker-diag] Exactly what the pixel shader would sample this draw. A
    // null view resolves to the cleared null image (all channels 0), which is
    // the shape of the defect: untextured geometry stays, textures vanish.
    for (uint32_t i = 0; i < texture_count_pixel; ++i) {
      const VulkanShader::TextureBinding& flicker_binding = (*textures_pixel)[i];
      ++flicker_frame_pixel_texture_slots;
      if (texture_cache_->GetActiveBindingOrNullImageView(
              flicker_binding.fetch_constant, flicker_binding.dimension,
              bool(flicker_binding.is_signed)) == VK_NULL_HANDLE) {
        ++flicker_frame_null_pixel_textures;
      }
    }
  }
  // Reuse texture and sampler bindings if not changed.
  bool textures_vertex_dirty =
      !(current_graphics_descriptor_set_values_up_to_date_ &
        (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex));
  if (!texture_count_vertex && !sampler_count_vertex) {
    bound_textures_vertex_count_ = UINT32_MAX;
    bound_samplers_vertex_count_ = UINT32_MAX;
    current_graphics_descriptor_set_values_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex);
  } else if (!textures_vertex_dirty) {
    if (bound_textures_vertex_count_ != texture_count_vertex ||
        bound_samplers_vertex_count_ != sampler_count_vertex ||
        texture_count_vertex > kMaxCachedShaderTextures ||
        sampler_count_vertex > kMaxCachedShaderSamplers) {
      textures_vertex_dirty = true;
    } else {
      for (uint32_t i = 0; i < texture_count_vertex; ++i) {
        const VulkanShader::TextureBinding& binding = textures_vertex[i];
        VkImageView image_view = texture_cache_->GetActiveBindingOrNullImageView(
            binding.fetch_constant, binding.dimension, bool(binding.is_signed));
        if (bound_texture_image_views_vertex_[i] != image_view) {
          textures_vertex_dirty = true;
          break;
        }
      }
      if (!textures_vertex_dirty) {
        for (uint32_t i = 0; i < sampler_count_vertex; ++i) {
          if (bound_samplers_vertex_[i] != current_samplers_vertex_[i].second) {
            textures_vertex_dirty = true;
            break;
          }
        }
      }
    }
    if (textures_vertex_dirty) {
      current_graphics_descriptor_set_values_up_to_date_ &=
          ~(UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex);
    }
  }

  bool textures_pixel_dirty =
      !(current_graphics_descriptor_set_values_up_to_date_ &
        (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel));
  if (!texture_count_pixel && !sampler_count_pixel) {
    bound_textures_pixel_count_ = UINT32_MAX;
    bound_samplers_pixel_count_ = UINT32_MAX;
    current_graphics_descriptor_set_values_up_to_date_ &=
        ~(UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel);
  } else if (!textures_pixel_dirty) {
    if (bound_textures_pixel_count_ != texture_count_pixel ||
        bound_samplers_pixel_count_ != sampler_count_pixel ||
        texture_count_pixel > kMaxCachedShaderTextures ||
        sampler_count_pixel > kMaxCachedShaderSamplers) {
      textures_pixel_dirty = true;
    } else {
      for (uint32_t i = 0; i < texture_count_pixel; ++i) {
        const VulkanShader::TextureBinding& binding = (*textures_pixel)[i];
        VkImageView image_view = texture_cache_->GetActiveBindingOrNullImageView(
            binding.fetch_constant, binding.dimension, bool(binding.is_signed));
        if (bound_texture_image_views_pixel_[i] != image_view) {
          textures_pixel_dirty = true;
          break;
        }
      }
      if (!textures_pixel_dirty) {
        for (uint32_t i = 0; i < sampler_count_pixel; ++i) {
          if (bound_samplers_pixel_[i] != current_samplers_pixel_[i].second) {
            textures_pixel_dirty = true;
            break;
          }
        }
      }
    }
    if (textures_pixel_dirty) {
      current_graphics_descriptor_set_values_up_to_date_ &=
          ~(UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel);
    }
  }

  // Make sure new descriptor sets are bound to the command buffer.

  current_graphics_descriptor_sets_bound_up_to_date_ &=
      current_graphics_descriptor_set_values_up_to_date_;

  // Fill the texture and sampler write image infos.

  bool write_vertex_textures =
      (texture_count_vertex || sampler_count_vertex) &&
      !(current_graphics_descriptor_set_values_up_to_date_ &
        (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex));
  bool write_pixel_textures =
      (texture_count_pixel || sampler_count_pixel) &&
      !(current_graphics_descriptor_set_values_up_to_date_ &
        (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel));
  descriptor_write_image_info_.clear();
  descriptor_write_image_info_.reserve(
      (write_vertex_textures ? texture_count_vertex + sampler_count_vertex : 0) +
      (write_pixel_textures ? texture_count_pixel + sampler_count_pixel : 0));
  size_t vertex_texture_image_info_offset = descriptor_write_image_info_.size();
  if (write_vertex_textures && texture_count_vertex) {
    for (const VulkanShader::TextureBinding& texture_binding : textures_vertex) {
      VkDescriptorImageInfo& descriptor_image_info = descriptor_write_image_info_.emplace_back();
      descriptor_image_info.imageView = texture_cache_->GetActiveBindingOrNullImageView(
          texture_binding.fetch_constant, texture_binding.dimension,
          bool(texture_binding.is_signed));
      descriptor_image_info.imageLayout = descriptor_image_info.imageView != VK_NULL_HANDLE
                                              ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                              : VK_IMAGE_LAYOUT_UNDEFINED;
    }
  }
  size_t vertex_sampler_image_info_offset = descriptor_write_image_info_.size();
  if (write_vertex_textures && sampler_count_vertex) {
    for (const std::pair<VulkanTextureCache::SamplerParameters, VkSampler>& sampler_pair :
         current_samplers_vertex_) {
      VkDescriptorImageInfo& descriptor_image_info = descriptor_write_image_info_.emplace_back();
      descriptor_image_info.sampler = sampler_pair.second;
    }
  }
  size_t pixel_texture_image_info_offset = descriptor_write_image_info_.size();
  if (write_pixel_textures && texture_count_pixel) {
    for (const VulkanShader::TextureBinding& texture_binding : *textures_pixel) {
      VkDescriptorImageInfo& descriptor_image_info = descriptor_write_image_info_.emplace_back();
      descriptor_image_info.imageView = texture_cache_->GetActiveBindingOrNullImageView(
          texture_binding.fetch_constant, texture_binding.dimension,
          bool(texture_binding.is_signed));
      descriptor_image_info.imageLayout = descriptor_image_info.imageView != VK_NULL_HANDLE
                                              ? VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                                              : VK_IMAGE_LAYOUT_UNDEFINED;
    }
  }
  size_t pixel_sampler_image_info_offset = descriptor_write_image_info_.size();
  if (write_pixel_textures && sampler_count_pixel) {
    for (const std::pair<VulkanTextureCache::SamplerParameters, VkSampler>& sampler_pair :
         current_samplers_pixel_) {
      VkDescriptorImageInfo& descriptor_image_info = descriptor_write_image_info_.emplace_back();
      descriptor_image_info.sampler = sampler_pair.second;
    }
  }

  // Write the new descriptor sets.

  // Consecutive bindings updated via a single VkWriteDescriptorSet must have
  // identical stage flags. Plus vertex and pixel texture images and samplers.
  std::array<VkWriteDescriptorSet, 2 * 2> write_descriptor_sets;
  uint32_t write_descriptor_set_count = 0;
  uint32_t write_descriptor_set_bits = 0;
  assert_not_zero(current_graphics_descriptor_set_values_up_to_date_ &
                  (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetSharedMemoryAndEdram));
  // Vertex shader textures and samplers.
  if (write_vertex_textures) {
    VkWriteDescriptorSet* write_textures =
        write_descriptor_sets.data() + write_descriptor_set_count;
    uint32_t texture_descriptor_set_write_count = WriteTransientTextureBindings(
        true, texture_count_vertex, sampler_count_vertex,
        current_guest_graphics_pipeline_layout_->descriptor_set_layout_textures_vertex_ref(),
        descriptor_write_image_info_.data() + vertex_texture_image_info_offset,
        descriptor_write_image_info_.data() + vertex_sampler_image_info_offset, write_textures);
    if (!texture_descriptor_set_write_count) {
      return false;
    }
    write_descriptor_set_count += texture_descriptor_set_write_count;
    write_descriptor_set_bits |= UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex;
    current_graphics_descriptor_sets_[SpirvShaderTranslator::kDescriptorSetTexturesVertex] =
        write_textures[0].dstSet;
    bound_textures_vertex_count_ = texture_count_vertex;
    bound_samplers_vertex_count_ = sampler_count_vertex;
    if (texture_count_vertex <= kMaxCachedShaderTextures) {
      for (uint32_t i = 0; i < texture_count_vertex; ++i) {
        bound_texture_image_views_vertex_[i] =
            descriptor_write_image_info_[vertex_texture_image_info_offset + i].imageView;
      }
    }
    if (sampler_count_vertex <= kMaxCachedShaderSamplers) {
      for (uint32_t i = 0; i < sampler_count_vertex; ++i) {
        bound_samplers_vertex_[i] =
            descriptor_write_image_info_[vertex_sampler_image_info_offset + i].sampler;
      }
    }
  }
  // Pixel shader textures and samplers.
  if (write_pixel_textures) {
    VkWriteDescriptorSet* write_textures =
        write_descriptor_sets.data() + write_descriptor_set_count;
    uint32_t texture_descriptor_set_write_count = WriteTransientTextureBindings(
        false, texture_count_pixel, sampler_count_pixel,
        current_guest_graphics_pipeline_layout_->descriptor_set_layout_textures_pixel_ref(),
        descriptor_write_image_info_.data() + pixel_texture_image_info_offset,
        descriptor_write_image_info_.data() + pixel_sampler_image_info_offset, write_textures);
    if (!texture_descriptor_set_write_count) {
      return false;
    }
    write_descriptor_set_count += texture_descriptor_set_write_count;
    write_descriptor_set_bits |= UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel;
    current_graphics_descriptor_sets_[SpirvShaderTranslator::kDescriptorSetTexturesPixel] =
        write_textures[0].dstSet;
    bound_textures_pixel_count_ = texture_count_pixel;
    bound_samplers_pixel_count_ = sampler_count_pixel;
    if (texture_count_pixel <= kMaxCachedShaderTextures) {
      for (uint32_t i = 0; i < texture_count_pixel; ++i) {
        bound_texture_image_views_pixel_[i] =
            descriptor_write_image_info_[pixel_texture_image_info_offset + i].imageView;
      }
    }
    if (sampler_count_pixel <= kMaxCachedShaderSamplers) {
      for (uint32_t i = 0; i < sampler_count_pixel; ++i) {
        bound_samplers_pixel_[i] =
            descriptor_write_image_info_[pixel_sampler_image_info_offset + i].sampler;
      }
    }
  }
  // Write.
  if (write_descriptor_set_count) {
    dfn.vkUpdateDescriptorSets(device, write_descriptor_set_count, write_descriptor_sets.data(), 0,
                               nullptr);
  }
  // Only make valid if all descriptor sets have been allocated and written
  // successfully.
  current_graphics_descriptor_set_values_up_to_date_ |= write_descriptor_set_bits;

  // Set 1 (constants) is bound with dynamic offsets only, using the persistent
  // descriptor set for this frame in flight.
  if (!(current_graphics_descriptor_sets_bound_up_to_date_ &
        (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetConstants)) ||
      last_bound_constant_buffer_dynamic_offsets_ != current_constant_buffer_dynamic_offsets_) {
    deferred_command_buffer_.CmdVkBindDescriptorSets(
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        current_guest_graphics_pipeline_layout_->GetPipelineLayout(),
        SpirvShaderTranslator::kDescriptorSetConstants, 1,
        &constant_descriptor_sets_[constant_ring_frame_index_],
        uint32_t(current_constant_buffer_dynamic_offsets_.size()),
        current_constant_buffer_dynamic_offsets_.data());
    last_bound_constant_buffer_dynamic_offsets_ = current_constant_buffer_dynamic_offsets_;
    current_graphics_descriptor_sets_bound_up_to_date_ |=
        UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetConstants;
  }

  // Bind the new descriptor sets (excluding Set 1 which was bound with dynamic offsets above).
  uint32_t descriptor_sets_needed = (UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetCount) - 1;
  descriptor_sets_needed &= ~(UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetConstants);
  if (!texture_count_vertex && !sampler_count_vertex) {
    descriptor_sets_needed &= ~(UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesVertex);
  }
  if (!texture_count_pixel && !sampler_count_pixel) {
    descriptor_sets_needed &= ~(UINT32_C(1) << SpirvShaderTranslator::kDescriptorSetTexturesPixel);
  }
  uint32_t descriptor_sets_remaining =
      descriptor_sets_needed & ~current_graphics_descriptor_sets_bound_up_to_date_;
  uint32_t descriptor_set_index;
  while (rex::bit_scan_forward(descriptor_sets_remaining, &descriptor_set_index)) {
    uint32_t descriptor_set_mask_tzcnt =
        rex::tzcnt(~(descriptor_sets_remaining | ((UINT32_C(1) << descriptor_set_index) - 1)));
    deferred_command_buffer_.CmdVkBindDescriptorSets(
        VK_PIPELINE_BIND_POINT_GRAPHICS,
        current_guest_graphics_pipeline_layout_->GetPipelineLayout(), descriptor_set_index,
        descriptor_set_mask_tzcnt - descriptor_set_index,
        current_graphics_descriptor_sets_ + descriptor_set_index, 0, nullptr);
    if (descriptor_set_mask_tzcnt >= 32) {
      break;
    }
    descriptor_sets_remaining &= ~((UINT32_C(1) << descriptor_set_mask_tzcnt) - 1);
  }
  current_graphics_descriptor_sets_bound_up_to_date_ |= descriptor_sets_needed;

  return true;
}

uint32_t VulkanCommandProcessor::WriteTransientTextureBindings(
    bool is_vertex, uint32_t texture_count, uint32_t sampler_count,
    VkDescriptorSetLayout descriptor_set_layout, const VkDescriptorImageInfo* texture_image_info,
    const VkDescriptorImageInfo* sampler_image_info,
    VkWriteDescriptorSet* descriptor_set_writes_out) {
  assert_true(frame_open_);
  if (!texture_count && !sampler_count) {
    return 0;
  }
  TextureDescriptorSetLayoutKey texture_descriptor_set_layout_key;
  texture_descriptor_set_layout_key.texture_count = texture_count;
  texture_descriptor_set_layout_key.sampler_count = sampler_count;
  texture_descriptor_set_layout_key.is_vertex = uint32_t(is_vertex);
  VkDescriptorSet texture_descriptor_set;
  auto textures_free_it =
      texture_transient_descriptor_sets_free_.find(texture_descriptor_set_layout_key);
  if (textures_free_it != texture_transient_descriptor_sets_free_.end() &&
      !textures_free_it->second.empty()) {
    texture_descriptor_set = textures_free_it->second.back();
    textures_free_it->second.pop_back();
  } else {
    std::array<VkDescriptorPoolSize, 2> texture_descriptor_counts;
    uint32_t texture_descriptor_counts_count = 0;
    if (texture_count) {
      VkDescriptorPoolSize& texture_descriptor_count =
          texture_descriptor_counts[texture_descriptor_counts_count++];
      texture_descriptor_count.type = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
      texture_descriptor_count.descriptorCount = texture_count;
    }
    if (sampler_count) {
      VkDescriptorPoolSize& texture_descriptor_count =
          texture_descriptor_counts[texture_descriptor_counts_count++];
      texture_descriptor_count.type = VK_DESCRIPTOR_TYPE_SAMPLER;
      texture_descriptor_count.descriptorCount = sampler_count;
    }
    assert_not_zero(texture_descriptor_counts_count);
    texture_descriptor_set = transient_descriptor_allocator_textures_.Allocate(
        descriptor_set_layout, texture_descriptor_counts.data(), texture_descriptor_counts_count);
    if (texture_descriptor_set == VK_NULL_HANDLE) {
      return 0;
    }
  }
  UsedTextureTransientDescriptorSet& used_texture_descriptor_set =
      texture_transient_descriptor_sets_used_.emplace_back();
  used_texture_descriptor_set.frame = frame_current_;
  used_texture_descriptor_set.layout = texture_descriptor_set_layout_key;
  used_texture_descriptor_set.set = texture_descriptor_set;
  uint32_t descriptor_set_write_count = 0;
  if (texture_count) {
    VkWriteDescriptorSet& descriptor_set_write =
        descriptor_set_writes_out[descriptor_set_write_count++];
    descriptor_set_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_set_write.pNext = nullptr;
    descriptor_set_write.dstSet = texture_descriptor_set;
    descriptor_set_write.dstBinding = 0;
    descriptor_set_write.dstArrayElement = 0;
    descriptor_set_write.descriptorCount = texture_count;
    descriptor_set_write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE;
    descriptor_set_write.pImageInfo = texture_image_info;
    descriptor_set_write.pBufferInfo = nullptr;
    descriptor_set_write.pTexelBufferView = nullptr;
  }
  if (sampler_count) {
    VkWriteDescriptorSet& descriptor_set_write =
        descriptor_set_writes_out[descriptor_set_write_count++];
    descriptor_set_write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_set_write.pNext = nullptr;
    descriptor_set_write.dstSet = texture_descriptor_set;
    descriptor_set_write.dstBinding = texture_count;
    descriptor_set_write.dstArrayElement = 0;
    descriptor_set_write.descriptorCount = sampler_count;
    descriptor_set_write.descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER;
    descriptor_set_write.pImageInfo = sampler_image_info;
    descriptor_set_write.pBufferInfo = nullptr;
    descriptor_set_write.pTexelBufferView = nullptr;
  }
  assert_not_zero(descriptor_set_write_count);
  return descriptor_set_write_count;
}

}  // namespace rex::graphics::vulkan
