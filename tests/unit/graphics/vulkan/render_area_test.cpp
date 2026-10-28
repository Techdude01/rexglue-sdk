// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_test_macros.hpp>
#include <rex/graphics/vulkan/render_area.h>
#include <utility>

using rex::graphics::vulkan::DrawRenderArea;
using rex::graphics::vulkan::RenderAreaContains;

TEST_CASE("Load-only rendering bounds scale with the actual host scissor", "[vulkan][render-area]") {
  const auto area = DrawRenderArea({{0, 0}, {1280, 384}}, {1280, 1024}, true, false);
  REQUIRE(area.extent.width == 1280);
  REQUIRE(area.extent.height == 384);
  const auto scaled = DrawRenderArea({{0, 0}, {2560, 768}}, {2560, 2048}, true, false);
  REQUIRE(scaled.extent.width == 2560);
  REQUIRE(scaled.extent.height == 768);
}

TEST_CASE("Clears and disabled bounds preserve the full attachment area", "[vulkan][render-area]") {
  for (const auto [bounded, clear] : {std::pair{false, false}, std::pair{true, true}}) {
    const auto area = DrawRenderArea({{10, 20}, {100, 50}}, {1280, 1024}, bounded, clear);
    REQUIRE(area.offset.x == 0);
    REQUIRE(area.offset.y == 0);
    REQUIRE(area.extent.width == 1280);
    REQUIRE(area.extent.height == 1024);
  }
}

TEST_CASE("Render area clipping is bounded without unsigned overflow", "[vulkan][render-area]") {
  const auto clipped = DrawRenderArea({{-8, -4}, {20, 12}}, {1280, 1024}, true, false);
  REQUIRE(clipped.offset.x == 0);
  REQUIRE(clipped.offset.y == 0);
  REQUIRE(clipped.extent.width == 12);
  REQUIRE(clipped.extent.height == 8);
  const auto empty = DrawRenderArea({{INT32_MAX, 0}, {UINT32_MAX, 4}}, {1280, 1024}, true, false);
  REQUIRE(empty.extent.width == 1280);
  REQUIRE(empty.extent.height == 1024);
}

TEST_CASE("Coalescing must restart if a later draw expands the render area", "[vulkan][render-area]") {
  const VkRect2D active{{0, 0}, {1280, 336}};
  REQUIRE(RenderAreaContains(active, {{10, 20}, {100, 50}}));
  REQUIRE(RenderAreaContains(active, active));
  REQUIRE_FALSE(RenderAreaContains(active, {{0, 0}, {1280, 384}}));
  REQUIRE_FALSE(RenderAreaContains({{10, 20}, {100, 50}}, {{0, 20}, {100, 50}}));
}
