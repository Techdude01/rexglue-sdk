#include <catch2/catch_test_macros.hpp>
#include <array>
#include <bit>
#include <rex/graphics/registers.h>
#include <rex/graphics/vulkan/raster_variant_audit.h>
using namespace rex::graphics;
using namespace rex::graphics::vulkan;
namespace {
constexpr uint64_t kHash = UINT64_C(0x6E6F06115807B602);
RasterVariantDecision Decision(uint64_t modification) {
  RasterVariantDecision result;
  result.valid = true;
  result.shader_hash = kHash;
  result.modification = modification;
  return result;
}
}
TEST_CASE("Variant window requires enabled finite bounded frame audit", "[raster_variant]") {
  CHECK(IsRasterVariantWindow(true, true, 4000, 4000, 4499));
  CHECK(IsRasterVariantWindow(true, true, 4499, 4000, 4499));
  CHECK_FALSE(IsRasterVariantWindow(false, true, 4000, 4000, 4499));
  CHECK_FALSE(IsRasterVariantWindow(true, false, 4000, 4000, 4499));
  CHECK_FALSE(IsRasterVariantWindow(true, true, 3999, 4000, 4499));
  CHECK_FALSE(IsRasterVariantWindow(true, true, 4500, 4000, 4499));
  CHECK_FALSE(IsRasterVariantWindow(true, true, 4000, 0, 4499));
  CHECK_FALSE(IsRasterVariantWindow(true, true, 4000, 4000, 0));
  CHECK_FALSE(IsRasterVariantWindow(true, true, 4000, 4000, 5024));
}
TEST_CASE("Final variant and hash identity are required before bucket counting", "[raster_variant]") {
  RasterVariantFrameAudit audit;
  auto decision = Decision(7);
  audit.Record(kHash, 8, decision);  // stale attempt cannot be attributed to final mod.
  decision = {};                  // same reset performed for each selection retry.
  audit.Record(kHash, 8, decision);
  decision = Decision(8);
  audit.Record(kHash, 8, decision);
  audit.Record(UINT64_C(0x2D4C010D3D75DEA5), 8, decision); // wrong shader identity.
  audit.Record(123, 8, decision);  // unrelated shader is not counted.
  REQUIRE(audit.used == 1);
  CHECK(audit.buckets[0].decision.modification == 8);
  CHECK(audit.buckets[0].draws == 1);
  CHECK(audit.target_draws == 4);
  CHECK(audit.invalid == 3);
  CHECK(audit.counted == 1);
  CHECK(audit.alpha_invalid == 3);
  CHECK(audit.alpha_counted == 1);
}
TEST_CASE("Unknown opacity and false opacity are distinct consumer buckets", "[raster_variant]") {
  RasterVariantFrameAudit audit;
  auto decision = Decision(7);
  audit.Record(kHash, 7, decision);
  decision.fetch0_opaque = RasterAuditState::kFalse;
  audit.Record(kHash, 7, decision);
  audit.Record(kHash, 7, decision);
  REQUIRE(audit.used == 2);
  CHECK(audit.buckets[0].draws == 1);
  CHECK(audit.buckets[1].draws == 2);
  CHECK(audit.target_draws == audit.counted + audit.invalid + audit.overflow);
  audit.Clear();
  CHECK(audit.used == 0);
  CHECK(audit.target_draws == 0);
  audit.Record(kHash, 7, decision);
  CHECK(audit.buckets[0].draws == 1);  // no leakage from previous frame.
}
TEST_CASE("Pixel float constant address selects pixel bank", "[raster_variant]") {
  constexpr uint32_t kVertexC102W = XE_GPU_REG_SHADER_CONSTANT_000_X + 4 * 102 + 3;
  constexpr uint32_t kPixelC102W = PixelFloatConstantRegister(102, 3);
  CHECK(kPixelC102W == XE_GPU_REG_SHADER_CONSTANT_256_X + 4 * 102 + 3);
  CHECK(kPixelC102W != kVertexC102W);

  std::array<uint32_t, 0x5003> regs{};
  regs[kVertexC102W] = UINT32_C(0x00000000);
  regs[kPixelC102W] = UINT32_C(0x3F800000);
  CHECK(std::bit_cast<float>(regs[PixelFloatConstantRegister(102, 3)]) == 1.0f);
  CHECK(std::bit_cast<float>(regs[kVertexC102W]) == 0.0f);
  // Match every index used by the actual four-map-word pixel upload loop.
  for (uint32_t group = 0; group < 4; ++group) {
    for (uint32_t index = 0; index < 64; ++index) {
      CHECK(PixelFloatConstantRegister((group << 6) + index, 0) ==
            XE_GPU_REG_SHADER_CONSTANT_256_X + (group << 8) + (index << 2));
    }
  }
  CHECK(PixelFloatConstantRegister(0, 0) == XE_GPU_REG_SHADER_CONSTANT_256_X);
  CHECK(PixelFloatConstantRegister(255, 3) == XE_GPU_REG_SHADER_CONSTANT_256_X + 1023);
}

TEST_CASE("Actual alpha inputs are exact, bounded, and independent of gate buckets", "[raster_variant]") {
  RasterVariantFrameAudit audit;
  auto decision = Decision(7);
  audit.Record(kHash, 7, decision);  // unknown input values are not zero-valued observations.
  decision.alpha_inputs_valid = 3;
  decision.alpha_ref_bits = UINT32_C(0x3E000001);
  decision.c102_w_bits = UINT32_C(0x3F800000);
  audit.Record(kHash, 7, decision);
  auto changed_alpha = decision;
  changed_alpha.alpha_ref_bits++;
  audit.Record(kHash, 7, changed_alpha);
  REQUIRE(audit.used == 1);  // Numeric inputs must not split the existing gate bucket.
  CHECK(audit.buckets[0].draws == 3);
  CHECK(audit.alpha_counted == 3);
  CHECK(audit.alpha_invalid == 0);
  REQUIRE(audit.alpha_pairs_used == 3);
  CHECK(audit.alpha_pairs[0].validity == 0);
  CHECK(audit.alpha_pairs[1].alpha_ref_bits == UINT32_C(0x3E000001));
  auto other_variant = decision;
  other_variant.modification = 8;
  audit.Record(kHash, 8, other_variant);
  REQUIRE(audit.alpha_pairs_used == 4);
  CHECK(audit.alpha_pairs[3].modification == 8);
  auto other_shader = decision;
  other_shader.shader_hash = UINT64_C(0x2D4C010D3D75DEA5);
  audit.Record(other_shader.shader_hash, 7, other_shader);
  REQUIRE(audit.alpha_pairs_used == 5);
  CHECK(audit.alpha_pairs[4].shader_hash == other_shader.shader_hash);

  audit.Clear();
  CHECK(audit.alpha_pairs_used == 0);
  CHECK(audit.alpha_counted == 0);
  for (size_t i = 0; i < RasterVariantFrameAudit::kAlphaCapacity + 1; ++i) {
    auto pair = Decision(7);
    pair.alpha_inputs_valid = 3;
    pair.alpha_ref_bits = static_cast<uint32_t>(i);
    pair.c102_w_bits = 0;
    audit.Record(kHash, 7, pair);
  }
  CHECK(audit.alpha_pairs_used == RasterVariantFrameAudit::kAlphaCapacity);
  CHECK(audit.alpha_overflow == 1);
  CHECK(audit.counted == RasterVariantFrameAudit::kAlphaCapacity + 1);
  CHECK(audit.used == 1);
  CHECK(audit.target_draws == audit.alpha_counted + audit.alpha_invalid + audit.alpha_overflow);
  CHECK(audit.target_draws == RasterVariantFrameAudit::kAlphaCapacity + 1);
}

TEST_CASE("Bucket exhaustion retains every emitted target draw in totals", "[raster_variant]") {
  RasterVariantFrameAudit audit;
  for (size_t i = 0; i < RasterVariantFrameAudit::kCapacity + 3; ++i) {
    const auto decision = Decision(i);
    audit.Record(kHash, i, decision);
  }
  const auto existing = Decision(0);
  audit.Record(kHash, 0, existing); // existing bucket remains usable after exhaustion.
  CHECK(audit.used == RasterVariantFrameAudit::kCapacity);
  CHECK(audit.overflow == 3);
  CHECK(audit.counted == RasterVariantFrameAudit::kCapacity + 1);
  CHECK(audit.target_draws == audit.counted + audit.invalid + audit.overflow);
}
