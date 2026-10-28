// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <cstdint>

namespace rex::graphics::vulkan {
constexpr bool IsPeriodicCaptureFrame(uint64_t frame, uint32_t interval, uint64_t last_frame) {
  return interval && frame && frame <= last_frame && frame % interval == 0;
}
}  // namespace rex::graphics::vulkan
