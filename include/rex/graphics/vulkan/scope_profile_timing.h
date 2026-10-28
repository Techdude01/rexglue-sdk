#pragma once

#include <cstdint>
#include <optional>

namespace rex::graphics::vulkan {

// A missing measurement is not a zero-duration scope. Vulkan timestamps may
// wrap at timestampValidBits; zero itself is a valid counter value.
inline std::optional<double> ScopeProfileMilliseconds(
    uint32_t valid_bits, double period_ns, bool ended,
    uint64_t start, uint64_t start_available,
    uint64_t end, uint64_t end_available) {
  if (!valid_bits || valid_bits > 64 || !ended ||
      !start_available || !end_available) {
    return std::nullopt;
  }
  const uint64_t mask = valid_bits == 64 ? ~uint64_t(0) :
      (uint64_t(1) << valid_bits) - 1;
  return double((end - start) & mask) * period_ns / 1e6;
}

}  // namespace rex::graphics::vulkan
