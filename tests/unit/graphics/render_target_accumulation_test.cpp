// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#include <array>
#include <catch2/catch_test_macros.hpp>
#include <rex/graphics/pipeline/render_target/cache.h>

namespace {
class CacheAccess : public rex::graphics::RenderTargetCache {
 public:
  using RenderTargetCache::CanKeepAccumulatedRenderTargets;
  using RenderTargetCache::RenderTarget;
  using RenderTargetCache::RenderTargetKey;
};
class Target final : public CacheAccess::RenderTarget {
 public:
  explicit Target(bool depth, uint32_t pitch = 16,
                  rex::graphics::xenos::MsaaSamples samples = rex::graphics::xenos::MsaaSamples::k1X)
      : RenderTarget(MakeKey(depth, pitch, samples)) {}
 private:
  static CacheAccess::RenderTargetKey MakeKey(bool depth, uint32_t pitch,
                                            rex::graphics::xenos::MsaaSamples samples) {
    CacheAccess::RenderTargetKey key;
    key.is_depth = depth;
    key.pitch_tiles_at_32bpp = pitch;
    key.msaa_samples = samples;
    return key;
  }
};
using Targets = std::array<CacheAccess::RenderTarget*, 5>;
}  // namespace

TEST_CASE("Depth-only draws can preserve the previous compatible color binding", "[attachment_retention]") {
  Target depth(true), color(false);
  Targets full{&depth, &color}, depth_only{&depth}, accumulated = full;
  auto keep = [&](const Targets& current, bool enabled) {
    return CacheAccess::CanKeepAccumulatedRenderTargets(
        current.data(), accumulated.data(), 16,
        rex::graphics::xenos::MsaaSamples::k1X, enabled);
  };
  CHECK(keep(full, false));
  CHECK_FALSE(keep(depth_only, false));
  // This is the production cache policy, not an independent simulator.
  CHECK(keep(depth_only, true));
  CHECK(keep(full, true));
  CHECK(accumulated == full);
}

TEST_CASE("Attachment retention preserves replacement, pitch, sample and alias rejection", "[attachment_retention]") {
  Target depth(true), color(false), replacement(false), new_slot(false);
  Targets full{&depth, &color}, depth_only{&depth};
  const auto samples = rex::graphics::xenos::MsaaSamples::k1X;
  CHECK_FALSE(CacheAccess::CanKeepAccumulatedRenderTargets(depth_only.data(), full.data(), 17, samples, true));
  CHECK_FALSE(CacheAccess::CanKeepAccumulatedRenderTargets(depth_only.data(), full.data(), 16,
      rex::graphics::xenos::MsaaSamples::k2X, true));
  Targets changed{&depth, &replacement};
  CHECK_FALSE(CacheAccess::CanKeepAccumulatedRenderTargets(changed.data(), full.data(), 16, samples, true));
  Targets grew{&depth, &color, &new_slot};
  CHECK_FALSE(CacheAccess::CanKeepAccumulatedRenderTargets(grew.data(), full.data(), 16, samples, true));
  Targets aliased{&depth, &color, &color};
  CHECK_FALSE(CacheAccess::CanKeepAccumulatedRenderTargets(depth_only.data(), aliased.data(), 16, samples, true));
}
