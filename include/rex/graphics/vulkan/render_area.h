// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <algorithm>
#include <cstdint>
#include <vulkan/vulkan.h>

namespace rex::graphics::vulkan {

// Clear load operations must keep their original extent. For load-only draws,
// restrict attachment operations to the scissor, clipped to the framebuffer.
inline VkRect2D DrawRenderArea(const VkRect2D& scissor, VkExtent2D framebuffer,
                               bool bounded, bool pending_clear) {
  const VkRect2D full{{0, 0}, framebuffer};
  if (!bounded || pending_clear) {
    return full;
  }
  const int64_t left = std::max<int64_t>(0, scissor.offset.x);
  const int64_t top = std::max<int64_t>(0, scissor.offset.y);
  const int64_t right = std::min<int64_t>(
      framebuffer.width, int64_t(scissor.offset.x) + scissor.extent.width);
  const int64_t bottom = std::min<int64_t>(
      framebuffer.height, int64_t(scissor.offset.y) + scissor.extent.height);
  // An empty scissor does not draw, but avoid a zero-sized rendering scope.
  if (right <= left || bottom <= top) {
    return full;
  }
  return {{int32_t(left), int32_t(top)},
          {uint32_t(right - left), uint32_t(bottom - top)}};
}

inline bool RenderAreaContains(const VkRect2D& active, const VkRect2D& needed) {
  return needed.offset.x >= active.offset.x && needed.offset.y >= active.offset.y &&
         int64_t(needed.offset.x) + needed.extent.width <=
             int64_t(active.offset.x) + active.extent.width &&
         int64_t(needed.offset.y) + needed.extent.height <=
             int64_t(active.offset.y) + active.extent.height;
}

}  // namespace rex::graphics::vulkan
