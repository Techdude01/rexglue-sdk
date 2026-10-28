// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause

#ifndef REX_GRAPHICS_VULKAN_FP32_PROBE_TRANSPORT_H_
#define REX_GRAPHICS_VULKAN_FP32_PROBE_TRANSPORT_H_

#include <bit>
#include <cstdint>

namespace rex::graphics::vulkan::diagnostics {

// Store fp32 diagnostic values as their IEEE-754 bits in a uint32 buffer. This
// avoids numeric conversion when moving values through integer GPU readback.
inline uint32_t EncodeFp32Bits(float value) { return std::bit_cast<uint32_t>(value); }
inline float DecodeFp32Bits(uint32_t bits) { return std::bit_cast<float>(bits); }

}  // namespace rex::graphics::vulkan::diagnostics

#endif  // REX_GRAPHICS_VULKAN_FP32_PROBE_TRANSPORT_H_
