// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_test_macros.hpp>
#include <rex/graphics/vulkan/transfer_scope_reuse.h>

TEST_CASE("Transfer continuation requires exactly the same concrete attachment", "[transfer_scope]") {
  using rex::graphics::vulkan::CanReuseTransferAttachment;
  const auto view = reinterpret_cast<VkImageView>(uintptr_t(1));
  const auto other_format_view = reinterpret_cast<VkImageView>(uintptr_t(2));
  const VkExtent2D extent{1280, 720};
  CHECK(CanReuseTransferAttachment(2, false, view, view, extent, extent));
  CHECK(CanReuseTransferAttachment(1, true, view, view, extent, extent));
  // Matching dimensions never justify using a different image/format view.
  CHECK_FALSE(CanReuseTransferAttachment(2, false, view, other_format_view, extent, extent));
  CHECK_FALSE(CanReuseTransferAttachment(2, false, VK_NULL_HANDLE, VK_NULL_HANDLE, extent, extent));
  CHECK_FALSE(CanReuseTransferAttachment(2, true, view, view, extent, extent));
  CHECK_FALSE(CanReuseTransferAttachment(1, false, view, view, extent, extent));
  CHECK_FALSE(CanReuseTransferAttachment(3, false, view, view, extent, extent));
  CHECK_FALSE(CanReuseTransferAttachment(6, false, view, view, extent, extent));
  CHECK_FALSE(CanReuseTransferAttachment(0, false, view, view, extent, extent));
  CHECK_FALSE(CanReuseTransferAttachment(2, false, view, view, extent, {1280, 721}));
  CHECK_FALSE(CanReuseTransferAttachment(2, false, view, view, extent, {1279, 720}));
  CHECK_FALSE(CanReuseTransferAttachment(2, false, view, view, {0, 720}, {0, 720}));
}
