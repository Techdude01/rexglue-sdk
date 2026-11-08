// Local diagnostic accounting only; never selects or modifies a pipeline.
#ifndef REX_GRAPHICS_VULKAN_RASTER_VARIANT_AUDIT_H_
#define REX_GRAPHICS_VULKAN_RASTER_VARIANT_AUDIT_H_
#include <array>
#include <cstddef>
#include <cstdint>

namespace rex::graphics::vulkan {
enum class RasterAuditState : uint8_t { kUnknown, kFalse, kTrue };
struct RasterVariantDecision {
  bool valid = false;
  uint64_t shader_hash = 0;
  uint64_t modification = 0;
  uint8_t depth_mode = 255;
  uint8_t alpha_func = 255;
  uint8_t guest_msaa = 255;
  uint8_t depth_test = 255;
  uint8_t depth_write = 255;
  uint8_t depth_compare = 255;
  RasterAuditState host_render_targets = RasterAuditState::kUnknown;
  RasterAuditState float24_conversion = RasterAuditState::kUnknown;
  RasterAuditState implicit_early_z_write = RasterAuditState::kUnknown;
  RasterAuditState writes_color_target0 = RasterAuditState::kUnknown;
  RasterAuditState coverage_alpha_initial = RasterAuditState::kUnknown;
  RasterAuditState coverage_alpha_final = RasterAuditState::kUnknown;
  RasterAuditState alpha_to_mask = RasterAuditState::kUnknown;
  RasterAuditState alpha_compare_pass = RasterAuditState::kUnknown;
  RasterAuditState alpha_ref_le_128 = RasterAuditState::kUnknown;
  RasterAuditState c102_w_ge_999 = RasterAuditState::kUnknown;
  RasterAuditState texture_cache_present = RasterAuditState::kUnknown;
  RasterAuditState fetch0_opaque = RasterAuditState::kUnknown;
  uint32_t alpha_inputs_valid = 0;
  uint32_t alpha_ref_bits = 0;
  uint32_t c102_w_bits = 0;
  bool operator==(const RasterVariantDecision&) const = default;
  RasterVariantDecision GateKey() const {
    auto key = *this;
    key.alpha_inputs_valid = key.alpha_ref_bits = key.c102_w_bits = 0;
    return key;
  }
};
constexpr bool IsRasterVariantTarget(uint64_t hash) {
  return hash == UINT64_C(0x6E6F06115807B602) || hash == UINT64_C(0x2D4C010D3D75DEA5);
}
constexpr bool IsRasterVariantWindow(bool enabled, bool audit_enabled, uint64_t frame,
                                     uint64_t start, uint64_t end) {
  return enabled && audit_enabled && start > 0 && end >= start && end - start < 1024 &&
         frame >= start && frame <= end;
}
// Count only at the consumer's successful guest draw-emission seam. No per-draw
// allocation or logging. Overflow/invalid remain in totals, never disappear.
struct RasterVariantFrameAudit {
  struct Bucket {
    RasterVariantDecision decision;
    uint64_t draws = 0;
  };
  static constexpr size_t kCapacity = 64;
  std::array<Bucket, kCapacity> buckets{};
  size_t used = 0;
  uint64_t target_draws = 0;
  uint64_t counted = 0;
  uint64_t overflow = 0;
  uint64_t invalid = 0;
  struct AlphaPair {
    uint64_t shader_hash = 0;
    uint64_t modification = 0;
    uint32_t validity = 0;
    uint32_t alpha_ref_bits = 0;
    uint32_t c102_w_bits = 0;
    uint64_t draws = 0;
  };
  static constexpr size_t kAlphaCapacity = 64;
  std::array<AlphaPair, kAlphaCapacity> alpha_pairs{};
  size_t alpha_pairs_used = 0;
  uint64_t alpha_counted = 0;
  uint64_t alpha_overflow = 0;
  uint64_t alpha_invalid = 0;
  void Record(uint64_t hash, uint64_t final_modification, const RasterVariantDecision& decision) {
    if (!IsRasterVariantTarget(hash)) return;
    ++target_draws;
    if (!decision.valid || decision.shader_hash != hash || decision.modification != final_modification) {
      ++invalid;
      ++alpha_invalid;
      return;
    }
    // Include unknown inputs explicitly; they are not invalid draw attribution.
    // Validity bit 0: reference read, bit 1: c102.w read. Unknown bits normalize to zero.
    const uint32_t ref_bits = (decision.alpha_inputs_valid & 1) ? decision.alpha_ref_bits : 0;
    const uint32_t c102_bits = (decision.alpha_inputs_valid & 2) ? decision.c102_w_bits : 0;
    {
      bool found = false;
      for (size_t i = 0; i < alpha_pairs_used; ++i) {
        auto& pair = alpha_pairs[i];
        if (pair.shader_hash == hash && pair.modification == final_modification &&
            pair.validity == decision.alpha_inputs_valid &&
            pair.alpha_ref_bits == ref_bits && pair.c102_w_bits == c102_bits) {
          ++pair.draws;
          found = true;
          break;
        }
      }
      if (!found && alpha_pairs_used < kAlphaCapacity) {
        alpha_pairs[alpha_pairs_used++] = {hash, final_modification,
                                           decision.alpha_inputs_valid, ref_bits, c102_bits, 1};
        found = true;
      }
      if (found) ++alpha_counted;
      else ++alpha_overflow;
    }
    const auto gate_key = decision.GateKey();
    for (size_t i = 0; i < used; ++i) {
      if (buckets[i].decision == gate_key) {
        ++buckets[i].draws;
        ++counted;
        return;
      }
    }
    if (used == kCapacity) {
      ++overflow;
      return;
    }
    buckets[used++] = {gate_key, 1};
    ++counted;
  }
  void Clear() { *this = {}; }
};
}  // namespace rex::graphics::vulkan
#endif
