// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_test_macros.hpp>
#include <cstdint>

namespace {

// Compute linear sample address before column swapping
inline uint32_t ComputeEdramAddress(uint32_t tile_index, uint32_t tile_width,
                                    uint32_t tile_height, uint32_t sample_x,
                                    uint32_t sample_y) {
  return tile_index * tile_width * tile_height + sample_y * tile_width + sample_x;
}

// Unsigned column swap arithmetic as emitted in SPIR-V
inline uint32_t SwapEdramDepthColumnUnsigned(uint32_t address, uint32_t sample_x,
                                             uint32_t tile_width_half) {
  return (sample_x < tile_width_half) ? (address + tile_width_half)
                                      : (address - tile_width_half);
}

// Legacy signed displacement arithmetic (which triggers sign-extension in NIR)
inline uint32_t SwapEdramDepthColumnSigned(uint32_t address, uint32_t sample_x,
                                           uint32_t tile_width_half) {
  int32_t disp = (sample_x < tile_width_half) ? int32_t(tile_width_half)
                                              : -int32_t(tile_width_half);
  return uint32_t(int32_t(address) + disp);
}

}  // namespace

TEST_CASE("EDRAM depth-dump column swap is algebraically equivalent to signed formula",
          "[vulkan][edram][columns]") {
  for (uint32_t scale : {1u, 2u, 3u}) {
    const uint32_t tile_width = 80u * scale;
    const uint32_t tile_height = 16u * scale;
    const uint32_t half = tile_width / 2u;

    for (uint32_t tile : {0u, 1u, 19u, 127u, 2047u}) {
      for (uint32_t y : {0u, 1u, tile_height / 2u, tile_height - 1u}) {
        for (uint32_t x = 0; x < tile_width; ++x) {
          const uint32_t original_addr = ComputeEdramAddress(tile, tile_width, tile_height, x, y);

          const uint32_t unsigned_swapped =
              SwapEdramDepthColumnUnsigned(original_addr, x, half);
          const uint32_t signed_swapped =
              SwapEdramDepthColumnSigned(original_addr, x, half);

          // 1. Unsigned and signed formulas produce identical 32-bit results
          REQUIRE(unsigned_swapped == signed_swapped);

          // 2. Swapping stays within the same row of the tile
          REQUIRE(unsigned_swapped / tile_width == original_addr / tile_width);

          // 3. Swapping moves strictly to the opposite half of the tile
          const uint32_t swapped_x = unsigned_swapped % tile_width;
          REQUIRE(swapped_x == (x + half) % tile_width);

          // 4. Involutive: applying the swap a second time returns the original address
          const uint32_t restored_addr =
              SwapEdramDepthColumnUnsigned(unsigned_swapped, swapped_x, half);
          REQUIRE(restored_addr == original_addr);
        }
      }
    }
  }
}

TEST_CASE("Unsigned EDRAM depth-dump column swap avoids signed negative displacements",
          "[vulkan][edram][columns]") {
  const uint32_t half = 40u;
  const uint32_t address = 1000u;

  // When sample_x >= half, signed formula produces -40 (0xFFFFFFD8)
  const int32_t negative_disp = -int32_t(half);
  REQUIRE(negative_disp == -40);

  // If sign-extended into 64-bit address space, int32_t(-40) becomes 0xFFFFFFFFFFFFFFD8
  const uint64_t sign_extended = uint64_t(int64_t(negative_disp));
  REQUIRE(sign_extended == 0xFFFFFFFFFFFFFFD8ULL);

  // But the unsigned formulation uses purely subtraction on non-negative operands
  REQUIRE(address >= half);
  const uint32_t unsigned_result = address - half;
  REQUIRE(unsigned_result == 960u);
}
