// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>
#include <vulkan/vulkan.h>

namespace rex::graphics::vulkan {
inline bool CanReuseTransferAttachment(uint32_t guest_attachment_mask, bool transfer_is_depth,
                                       VkImageView transfer_view, VkImageView guest_view,
                                       VkExtent2D transfer_extent, VkExtent2D guest_extent,
                                       bool transfer_has_depth = false,
                                       VkImageView transfer_depth_view = VK_NULL_HANDLE,
                                       VkImageView guest_depth_view = VK_NULL_HANDLE) {
  if (transfer_has_depth && !transfer_is_depth) {
    if (guest_attachment_mask == 3u) {
      return transfer_view != VK_NULL_HANDLE && transfer_view == guest_view &&
             transfer_depth_view != VK_NULL_HANDLE && transfer_depth_view == guest_depth_view &&
             transfer_extent.width && transfer_extent.height &&
             transfer_extent.width == guest_extent.width &&
             transfer_extent.height == guest_extent.height;
    }
  }
  const uint32_t required_mask = transfer_is_depth ? 1u : 2u;
  return guest_attachment_mask == required_mask && transfer_view != VK_NULL_HANDLE &&
         transfer_view == guest_view && transfer_extent.width && transfer_extent.height &&
         transfer_extent.width == guest_extent.width &&
         transfer_extent.height == guest_extent.height;
}
}  // namespace rex::graphics::vulkan
