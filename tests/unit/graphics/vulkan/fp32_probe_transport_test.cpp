#include <array>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include <rex/graphics/vulkan/fp32_probe_transport.h>

TEST_CASE("fp32 probe transport preserves IEEE-754 payload bits", "[vulkan][diagnostics]") {
  constexpr std::array<uint32_t, 7> patterns = {
      0x00000000u, 0x80000000u, 0x3F800000u, 0xBF800000u,
      0x00000001u, 0x7F800000u, 0x7FC12345u,
  };
  for (uint32_t pattern : patterns) {
    const float decoded = rex::graphics::vulkan::diagnostics::DecodeFp32Bits(pattern);
    CHECK(rex::graphics::vulkan::diagnostics::EncodeFp32Bits(decoded) == pattern);
  }
}
