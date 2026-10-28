// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_test_macros.hpp>
#include <rex/graphics/vulkan/frame_capture_schedule.h>

TEST_CASE("Periodic diagnostics capture exactly nine frames through verification frame", "[capture_schedule]") {
  using rex::graphics::vulkan::IsPeriodicCaptureFrame;
  unsigned count = 0;
  for (uint64_t frame = 0; frame <= 5000; ++frame) {
    if (IsPeriodicCaptureFrame(frame, 500, 4500)) {
      ++count;
      CHECK(frame == count * 500);
    }
  }
  CHECK(count == 9);
  CHECK_FALSE(IsPeriodicCaptureFrame(4500, 0, 4500));
  CHECK_FALSE(IsPeriodicCaptureFrame(0, 500, 4500));
  CHECK_FALSE(IsPeriodicCaptureFrame(500, 500, 0));
}
