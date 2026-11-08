/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <utility>

#include <rex/assert.h>
#include <rex/bit.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/cvar.h>
#include <rex/dbg.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/util/draw.h>
#include <rex/graphics/pipeline/texture/info.h>
#include <rex/graphics/pipeline/texture/util.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/graphics/vulkan/texture_cache.h>
#include <rex/logging.h>
#include <rex/math.h>
#include <rex/ui/vulkan/mem_alloc.h>
#include <rex/ui/vulkan/ui_samplers.h>
#include <rex/ui/vulkan/util.h>

REXCVAR_DEFINE_BOOL(non_seamless_cube_map, false, "GPU", "Use non-seamless cube map sampling");
REXCVAR_DEFINE_BOOL(gpu_fused_resolve_sampled_image, false, "GPU/Vulkan",
                    "Experimental fused resolve output to a sampled image (restart required)");
REXCVAR_DEFINE_BOOL(gpu_fused_resolve_image_parity, false, "GPU/Vulkan",
                    "Compare fused resolve sampled images against the legacy load");

namespace rex::graphics::vulkan {

// Generated with `xb buildshaders`.
namespace shaders {
#include "../shaders/vulkan_spirv/texture_load_128bpb_cs.h"
#include "../shaders/vulkan_spirv/texture_load_128bpb_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_16bpb_cs.h"
#include "../shaders/vulkan_spirv/texture_load_16bpb_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_32bpb_cs.h"
#include "../shaders/vulkan_spirv/texture_load_32bpb_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_64bpb_cs.h"
#include "../shaders/vulkan_spirv/texture_load_64bpb_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_8bpb_cs.h"
#include "../shaders/vulkan_spirv/texture_load_8bpb_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_bgrg8_rgb8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_ctx1_cs.h"
#include "../shaders/vulkan_spirv/texture_load_depth_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_depth_float_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_depth_unorm_cs.h"
#include "../shaders/vulkan_spirv/texture_load_depth_unorm_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxn_rg8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt1_rgba8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt3_rgba8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt3a_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt3aas1111_argb4_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt5_rgba8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_dxt5a_r8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_gbgr8_rgb8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r10g11b11_rgba16_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r10g11b11_rgba16_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r10g11b11_rgba16_snorm_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r10g11b11_rgba16_snorm_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r11g11b10_rgba16_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r11g11b10_rgba16_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r11g11b10_rgba16_snorm_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r11g11b10_rgba16_snorm_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r16_snorm_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r16_snorm_float_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r16_unorm_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r16_unorm_float_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r4g4b4a4_a4r4g4b4_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r4g4b4a4_a4r4g4b4_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r4g4b4a4_rgba8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r4g4b4a4_rgba8_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g5b5a1_b5g5r5a1_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g5b5a1_b5g5r5a1_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g5b6_b5g6r5_swizzle_rbga_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g5b6_b5g6r5_swizzle_rbga_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g6b5_b5g6r5_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g6b5_b5g6r5_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g6b5_rgba8_cs.h"
#include "../shaders/vulkan_spirv/texture_load_r5g6b5_rgba8_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rg16_snorm_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rg16_snorm_float_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rg16_unorm_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rg16_unorm_float_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rgba16_snorm_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rgba16_snorm_float_scaled_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rgba16_unorm_float_cs.h"
#include "../shaders/vulkan_spirv/texture_load_rgba16_unorm_float_scaled_cs.h"
}  // namespace shaders

namespace {

// Default-off direct storage-image load shader (the R8G8B8A8_UNORM slice).
//
// This is a 32bpp scaled-resolve load with the destination changed from the
// scratch storage buffer to the sampleable storage image. The guest tiled
// address computation, the scaled-resolve source addressing, the destination
// texel ordering (two 16-byte words = eight R8G8B8A8 texels) and the endianness
// handling are identical to texture_load_32bpb_scaled_cs, so the only
// behavioural difference is the final destination write: instead of storing two
// 16-byte words into the scratch buffer and then copying them into the image
// with vkCmdCopyBufferToImage, the eight texels are stored as normalized RGBA8
// image stores.
//
// A byte value stored as float(byte) / 255.0 is converted back to exactly the
// same byte by the UNORM8 conversion (always normalizes to round-to-nearest, and
// byte / 255 in binary32 is within far less than half an ulp of an integer when
// rescaled by 255) - see DirectStorageImageEncodeByte in direct_storage_image.h
// and its exhaustive unit test.
//
// 3D tiling is not implemented here and is excluded by the eligibility check
// before dispatch; the early return below is a guard, not a supported path.
#include "direct_storage_image_sources.h"

#if 0  // Rejected hand-port retained temporarily until parity validates shared XeSL.
static constexpr char kRejectedDirectStorageImageLoadShaderSource[] = R"(#version 450
layout(local_size_x = 4, local_size_y = 32, local_size_z = 1) in;

layout(push_constant) uniform XeTextureLoadDirectConstants {
  uint xe_is_tiled_3d_endian_scale;
  uint xe_guest_offset;
  uint xe_guest_pitch_aligned;
  uint xe_guest_z_stride_block_rows_aligned;
  uvec3 xe_size_blocks;
  uint xe_host_offset;
  uint xe_host_pitch;
  uint xe_height_texels;
};

layout(set = 1, binding = 0, std430) readonly buffer XeTextureLoadDirectSource {
  uvec4 xe_source[];
};

layout(set = 0, binding = 0, rgba8) writeonly uniform image2D xe_dest;

vec4 xe_direct_storage_image_unpack(uint texel) {
  return vec4(float(texel & 255u), float((texel >> 8u) & 255u),
              float((texel >> 16u) & 255u), float((texel >> 24u) & 255u)) *
         (1.0 / 255.0);
}

uvec4 xe_direct_storage_image_swap_endian(uvec4 value, uint endian) {
  if (endian == 1u || endian == 2u) {
    value = ((value & 0x00FF00FFu) << 8u) | ((value & 0xFF00FF00u) >> 8u);
  }
  if (endian == 2u || endian == 3u) {
    value = (value << 16u) | (value >> 16u);
  }
  return value;
}

void xe_direct_storage_image_store_word(uvec4 word, uint x, uint y, uint width) {
  if (x < width) {
    imageStore(xe_dest, ivec2(int(x), int(y)), xe_direct_storage_image_unpack(word.x));
  }
  if (x + 1u < width) {
    imageStore(xe_dest, ivec2(int(x + 1u), int(y)), xe_direct_storage_image_unpack(word.y));
  }
  if (x + 2u < width) {
    imageStore(xe_dest, ivec2(int(x + 2u), int(y)), xe_direct_storage_image_unpack(word.z));
  }
  if (x + 3u < width) {
    imageStore(xe_dest, ivec2(int(x + 3u), int(y)), xe_direct_storage_image_unpack(word.w));
  }
}

void main() {
  uvec3 invocation = gl_GlobalInvocationID;
  uvec2 pixel = uvec2(invocation.x << 3u, invocation.y);
  if (any(greaterThanEqual(pixel, xe_size_blocks.xy))) {
    return;
  }
  // Force_load_3d_tiling is not part of the supported slice.
  if ((xe_is_tiled_3d_endian_scale & 2u) != 0u) {
    return;
  }
  uint scale_x = (xe_is_tiled_3d_endian_scale >> 4u) & 7u;
  uint scale_y = (xe_is_tiled_3d_endian_scale >> 7u) & 7u;
  uint endian = (xe_is_tiled_3d_endian_scale >> 2u) & 3u;

  // Guest tiled address of the block start, exactly as in
  // texture_load_32bpb_scaled_cs (2D tiling).
  int block_x = int(pixel.x >> 2u);
  int block_y = int(pixel.y);
  int block_pitch = int(xe_guest_pitch_aligned >> 5u);
  int tile_index = (block_x >> 5) + ((block_y >> 5) * block_pitch);
  int tile_base = tile_index << 9;
  int inner = ((block_x & 7) + ((block_y & 14) << 2)) << 2;
  int swapped = ((tile_base + ((inner & -16) << 1)) + (inner & 15)) + ((block_y & 1) << 4);
  uint guest_address = uint(((swapped & -512) << 3) + ((block_y & 16) << 7) +
                            ((swapped & 448) << 2) + (((((block_y & 8) >> 2) + (block_x >> 3)) & 3) << 6) +
                            (swapped & 63));

  // Scaled-resolve source addressing.
  uvec2 block = uvec2(pixel.x >> 2u, pixel.y) / uvec2(scale_x, scale_y);
  uvec2 in_block = uvec2(pixel.x >> 2u, pixel.y) - block * uvec2(scale_x, scale_y);
  uint combined_scale = scale_x * scale_y;
  uint source_index =
      (xe_guest_offset + guest_address * combined_scale +
       ((((in_block.x * scale_y) + in_block.y) << 2u) + (pixel.x & 3u)) << 2u) >> 4u;
  uvec4 word_0 = xe_direct_storage_image_swap_endian(xe_source[source_index], endian);

  // Second 16-byte destination word, as in the buffer path.
  uint second_delta;
  if (scale_x > 1u) {
    second_delta = ((in_block.x + 1u == scale_x) ? (32u * scale_x - (in_block.x << 4u)) : 16u) * scale_y;
  } else {
    second_delta = 32u * scale_y;
  }
  uvec4 word_1 = xe_direct_storage_image_swap_endian(
      xe_source[source_index + (second_delta >> 4u)], endian);

  uint width = xe_size_blocks.x;
  uint y = pixel.y;
  xe_direct_storage_image_store_word(word_0, pixel.x, y, width);
  xe_direct_storage_image_store_word(word_1, pixel.x + 4u, y, width);
}
)";

#endif

// Keep invalid fetch constant texture reads aligned with D3D12 null SRV
// descriptors, which force all sampled channels to 0.
constexpr VkFormat kInvalidTextureFetchFallbackFormat = VK_FORMAT_R8G8B8A8_UNORM;
const VkClearColorValue kInvalidTextureFetchFallbackColor = {{0.0f, 0.0f, 0.0f, 0.0f}};

}  // namespace

static_assert(VK_FORMAT_UNDEFINED == VkFormat(0),
              "Assuming that skipping a VkFormat in an initializer results in "
              "VK_FORMAT_UNDEFINED");
const VulkanTextureCache::HostFormatPair VulkanTextureCache::kBestHostFormats[64] = {
    // k_1_REVERSE
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_1
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_8
    {{kLoadShaderIndex8bpb, VK_FORMAT_R8_UNORM},
     {kLoadShaderIndex8bpb, VK_FORMAT_R8_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_1_5_5_5
    // Red and blue swapped in the load shader for simplicity.
    {{kLoadShaderIndexR5G5B5A1ToB5G5R5A1, VK_FORMAT_A1R5G5B5_UNORM_PACK16},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_5_6_5
    // Red and blue swapped in the load shader for simplicity.
    {{kLoadShaderIndexR5G6B5ToB5G6R5, VK_FORMAT_R5G6B5_UNORM_PACK16},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_6_5_5
    // On the host, green bits in blue, blue bits in green.
    {{kLoadShaderIndexR5G5B6ToB5G6R5WithRBGASwizzle, VK_FORMAT_R5G6B5_UNORM_PACK16},
     {kLoadShaderIndexUnknown},
     XE_GPU_MAKE_TEXTURE_SWIZZLE(R, B, G, G)},
    // k_8_8_8_8
    {{kLoadShaderIndex32bpb, VK_FORMAT_R8G8B8A8_UNORM},
     {kLoadShaderIndex32bpb, VK_FORMAT_R8G8B8A8_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
     true},
    // k_2_10_10_10
    // Signed 2_10_10_10 is unsupported on D3D12, keep Vulkan behavior aligned.
    {{kLoadShaderIndex32bpb, VK_FORMAT_A2B10G10R10_UNORM_PACK32},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_8_A
    {{kLoadShaderIndex8bpb, VK_FORMAT_R8_UNORM},
     {kLoadShaderIndex8bpb, VK_FORMAT_R8_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_8_B
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_8_8
    {{kLoadShaderIndex16bpb, VK_FORMAT_R8G8_UNORM},
     {kLoadShaderIndex16bpb, VK_FORMAT_R8G8_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG,
     true},
    // k_Cr_Y1_Cb_Y0_REP
    // VK_FORMAT_G8B8G8R8_422_UNORM (added in
    // VK_KHR_sampler_ycbcr_conversion and promoted to Vulkan 1.1) is
    // optional.
    // Signed variant is unsupported on D3D12.
    {{kLoadShaderIndex32bpb, VK_FORMAT_G8B8G8R8_422_UNORM, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_Y1_Cr_Y0_Cb_REP
    // VK_FORMAT_B8G8R8G8_422_UNORM (added in
    // VK_KHR_sampler_ycbcr_conversion and promoted to Vulkan 1.1) is
    // optional.
    // Signed variant is unsupported on D3D12.
    {{kLoadShaderIndex32bpb, VK_FORMAT_B8G8R8G8_422_UNORM, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_16_16_EDRAM
    // Not usable as a texture, also has -32...32 range.
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_8_8_8_8_A
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_4_4_4_4
    // Components swapped in the load shader for simplicity.
    {{kLoadShaderIndexRGBA4ToARGB4, VK_FORMAT_B4G4R4A4_UNORM_PACK16},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_10_11_11
    // Runtime fallback to float16 may be enabled if normalized RGBA16 host
    // formats are not linearly filterable.
    {{kLoadShaderIndexR11G11B10ToRGBA16, VK_FORMAT_R16G16B16A16_UNORM},
     {kLoadShaderIndexR11G11B10ToRGBA16SNorm, VK_FORMAT_R16G16B16A16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_11_11_10
    // Runtime fallback to float16 may be enabled if normalized RGBA16 host
    // formats are not linearly filterable.
    {{kLoadShaderIndexR10G11B11ToRGBA16, VK_FORMAT_R16G16B16A16_UNORM},
     {kLoadShaderIndexR10G11B11ToRGBA16SNorm, VK_FORMAT_R16G16B16A16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_DXT1
    // VK_FORMAT_BC1_RGBA_UNORM_BLOCK is optional.
    {{kLoadShaderIndex64bpb, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT2_3
    // VK_FORMAT_BC2_UNORM_BLOCK is optional.
    {{kLoadShaderIndex128bpb, VK_FORMAT_BC2_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT4_5
    // VK_FORMAT_BC3_UNORM_BLOCK is optional.
    {{kLoadShaderIndex128bpb, VK_FORMAT_BC3_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_16_16_16_16_EDRAM
    // Not usable as a texture, also has -32...32 range.
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_24_8
    // Keep signed and unsigned sampling on the same R32 float descriptor path
    // for parity with D3D12.
    {{kLoadShaderIndexDepthUnorm, VK_FORMAT_R32_SFLOAT},
     {kLoadShaderIndexDepthUnorm, VK_FORMAT_R32_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_24_8_FLOAT
    // Keep signed and unsigned sampling on the same R32 float descriptor path
    // for parity with D3D12.
    {{kLoadShaderIndexDepthFloat, VK_FORMAT_R32_SFLOAT},
     {kLoadShaderIndexDepthFloat, VK_FORMAT_R32_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_16
    // VK_FORMAT_R16_UNORM and VK_FORMAT_R16_SNORM are optional.
    {{kLoadShaderIndex16bpb, VK_FORMAT_R16_UNORM},
     {kLoadShaderIndex16bpb, VK_FORMAT_R16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_16_16
    // VK_FORMAT_R16G16_UNORM and VK_FORMAT_R16G16_SNORM are optional.
    {{kLoadShaderIndex32bpb, VK_FORMAT_R16G16_UNORM},
     {kLoadShaderIndex32bpb, VK_FORMAT_R16G16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG,
     true},
    // k_16_16_16_16
    // VK_FORMAT_R16G16B16A16_UNORM and VK_FORMAT_R16G16B16A16_SNORM are
    // optional.
    {{kLoadShaderIndex64bpb, VK_FORMAT_R16G16B16A16_UNORM},
     {kLoadShaderIndex64bpb, VK_FORMAT_R16G16B16A16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
     true},
    // k_16_EXPAND
    {{kLoadShaderIndex16bpb, VK_FORMAT_R16_SFLOAT},
     {kLoadShaderIndex16bpb, VK_FORMAT_R16_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_16_16_EXPAND
    {{kLoadShaderIndex32bpb, VK_FORMAT_R16G16_SFLOAT},
     {kLoadShaderIndex32bpb, VK_FORMAT_R16G16_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG,
     true},
    // k_16_16_16_16_EXPAND
    {{kLoadShaderIndex64bpb, VK_FORMAT_R16G16B16A16_SFLOAT},
     {kLoadShaderIndex64bpb, VK_FORMAT_R16G16B16A16_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
     true},
    // k_16_FLOAT
    {{kLoadShaderIndex16bpb, VK_FORMAT_R16_SFLOAT},
     {kLoadShaderIndex16bpb, VK_FORMAT_R16_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_16_16_FLOAT
    {{kLoadShaderIndex32bpb, VK_FORMAT_R16G16_SFLOAT},
     {kLoadShaderIndex32bpb, VK_FORMAT_R16G16_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG,
     true},
    // k_16_16_16_16_FLOAT
    {{kLoadShaderIndex64bpb, VK_FORMAT_R16G16B16A16_SFLOAT},
     {kLoadShaderIndex64bpb, VK_FORMAT_R16G16B16A16_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
     true},
    // k_32
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_32
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_32_32_32_32
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_32_FLOAT
    {{kLoadShaderIndex32bpb, VK_FORMAT_R32_SFLOAT},
     {kLoadShaderIndex32bpb, VK_FORMAT_R32_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
     true},
    // k_32_32_FLOAT
    {{kLoadShaderIndex64bpb, VK_FORMAT_R32G32_SFLOAT},
     {kLoadShaderIndex64bpb, VK_FORMAT_R32G32_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG,
     true},
    // k_32_32_32_32_FLOAT
    {{kLoadShaderIndex128bpb, VK_FORMAT_R32G32B32A32_SFLOAT},
     {kLoadShaderIndex128bpb, VK_FORMAT_R32G32B32A32_SFLOAT},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
     true},
    // k_32_AS_8
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_AS_8_8
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_16_MPEG
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16_16_MPEG
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_8_INTERLACED
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_AS_8_INTERLACED
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_32_AS_8_8_INTERLACED
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_16_INTERLACED
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16_MPEG_INTERLACED
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_16_16_MPEG_INTERLACED
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_DXN
    // VK_FORMAT_BC5_UNORM_BLOCK is optional.
    {{kLoadShaderIndex128bpb, VK_FORMAT_BC5_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_8_8_8_8_AS_16_16_16_16
    {{kLoadShaderIndex32bpb, VK_FORMAT_R8G8B8A8_UNORM},
     {kLoadShaderIndex32bpb, VK_FORMAT_R8G8B8A8_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
     true},
    // k_DXT1_AS_16_16_16_16
    // VK_FORMAT_BC1_RGBA_UNORM_BLOCK is optional.
    {{kLoadShaderIndex64bpb, VK_FORMAT_BC1_RGBA_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT2_3_AS_16_16_16_16
    // VK_FORMAT_BC2_UNORM_BLOCK is optional.
    {{kLoadShaderIndex128bpb, VK_FORMAT_BC2_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_DXT4_5_AS_16_16_16_16
    // VK_FORMAT_BC3_UNORM_BLOCK is optional.
    {{kLoadShaderIndex128bpb, VK_FORMAT_BC3_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_2_10_10_10_AS_16_16_16_16
    // Signed 2_10_10_10 is unsupported on D3D12, keep Vulkan behavior aligned.
    {{kLoadShaderIndex32bpb, VK_FORMAT_A2B10G10R10_UNORM_PACK32},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_10_11_11_AS_16_16_16_16
    // Runtime fallback to float16 may be enabled if normalized RGBA16 host
    // formats are not linearly filterable.
    {{kLoadShaderIndexR11G11B10ToRGBA16, VK_FORMAT_R16G16B16A16_UNORM},
     {kLoadShaderIndexR11G11B10ToRGBA16SNorm, VK_FORMAT_R16G16B16A16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_11_11_10_AS_16_16_16_16
    // Runtime fallback to float16 may be enabled if normalized RGBA16 host
    // formats are not linearly filterable.
    {{kLoadShaderIndexR10G11B11ToRGBA16, VK_FORMAT_R16G16B16A16_UNORM},
     {kLoadShaderIndexR10G11B11ToRGBA16SNorm, VK_FORMAT_R16G16B16A16_SNORM},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_32_32_32_FLOAT
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB},
    // k_DXT3A
    {{kLoadShaderIndexDXT3A, VK_FORMAT_R8_UNORM},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_DXT5A
    // VK_FORMAT_BC4_UNORM_BLOCK is optional.
    {{kLoadShaderIndex64bpb, VK_FORMAT_BC4_UNORM_BLOCK, true},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR},
    // k_CTX1
    {{kLoadShaderIndexCTX1, VK_FORMAT_R8G8_UNORM},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG},
    // k_DXT3A_AS_1_1_1_1
    {{kLoadShaderIndexDXT3AAs1111ToARGB4, VK_FORMAT_B4G4R4A4_UNORM_PACK16},
     {kLoadShaderIndexUnknown},
     xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_8_8_8_8_GAMMA_EDRAM
    // Not usable as a texture.
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
    // k_2_10_10_10_FLOAT_EDRAM
    // Not usable as a texture.
    {{kLoadShaderIndexUnknown}, {kLoadShaderIndexUnknown}, xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA},
};

// Vulkan requires 2x1 (4:2:2) subsampled images to have an even width.
// Always decompressing them to RGBA8 UNORM for D3D12 parity.

const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatGBGRUnaligned = {
    {kLoadShaderIndexGBGR8ToRGB8, VK_FORMAT_R8G8B8A8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB,
    false};

const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatBGRGUnaligned = {
    {kLoadShaderIndexBGRG8ToRGB8, VK_FORMAT_R8G8B8A8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RGBB,
    false};

// Keep BC unaligned fallback mapping compatible with the D3D12 texture cache
// decompression mapping.
const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatDXT1Unaligned = {
    {kLoadShaderIndexDXT1ToRGBA8, VK_FORMAT_R8G8B8A8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
    false};

const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatDXT2_3Unaligned = {
    {kLoadShaderIndexDXT3ToRGBA8, VK_FORMAT_R8G8B8A8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
    false};

const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatDXT4_5Unaligned = {
    {kLoadShaderIndexDXT5ToRGBA8, VK_FORMAT_R8G8B8A8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA,
    false};

const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatDXNUnaligned = {
    {kLoadShaderIndexDXNToRG8, VK_FORMAT_R8G8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RGGG,
    false};

const VulkanTextureCache::HostFormatPair VulkanTextureCache::kHostFormatDXT5AUnaligned = {
    {kLoadShaderIndexDXT5AToR8, VK_FORMAT_R8_UNORM, false, true},
    {kLoadShaderIndexUnknown},
    xenos::XE_GPU_TEXTURE_SWIZZLE_RRRR,
    false};

VulkanTextureCache::~VulkanTextureCache() {
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  for (const std::pair<const SamplerParameters, Sampler>& sampler_pair : samplers_) {
    dfn.vkDestroySampler(device, sampler_pair.second.sampler, nullptr);
  }
  samplers_.clear();
  last_sampler_parameters_.value = UINT32_MAX;
  last_sampler_ = VK_NULL_HANDLE;
  custom_border_color_sampler_count_ = 0;
  COUNT_profile_set("gpu/texture_cache/vulkan/samplers", 0);
  sampler_used_last_ = nullptr;
  sampler_used_first_ = nullptr;

  ShutdownScaledResolveBuffer();

  if (null_image_view_3d_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, null_image_view_3d_, nullptr);
  }
  if (null_image_view_cube_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, null_image_view_cube_, nullptr);
  }
  if (null_image_view_2d_array_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, null_image_view_2d_array_, nullptr);
  }
  if (null_image_3d_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImage(device, null_image_3d_, nullptr);
  }
  if (null_image_2d_array_cube_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImage(device, null_image_2d_array_cube_, nullptr);
  }
  for (VkDeviceMemory null_images_memory : null_images_memory_) {
    if (null_images_memory != VK_NULL_HANDLE) {
      dfn.vkFreeMemory(device, null_images_memory, nullptr);
    }
  }
  for (VkPipeline load_pipeline : load_pipelines_scaled_) {
    if (load_pipeline != VK_NULL_HANDLE) {
      dfn.vkDestroyPipeline(device, load_pipeline, nullptr);
    }
  }
  for (VkPipeline load_pipeline : load_pipelines_) {
    if (load_pipeline != VK_NULL_HANDLE) {
      dfn.vkDestroyPipeline(device, load_pipeline, nullptr);
    }
  }
  if (load_pipeline_layout_ != VK_NULL_HANDLE) {
    dfn.vkDestroyPipelineLayout(device, load_pipeline_layout_, nullptr);
  }

  // Textures memory is allocated using the Vulkan Memory Allocator, destroy all
  // textures before destroying VMA.
  DestroyAllTextures(true);
  CompleteFusedImageParity(UINT64_MAX);
  pending_fused_resolve_image_.reset();
  retired_fused_resolve_images_.clear();
  resolved_images_.Clear();
  resolved_images_.CompletedSubmissionUpdated(UINT64_MAX);
  if (direct_storage_image_pipeline_) dfn.vkDestroyPipeline(device, direct_storage_image_pipeline_, nullptr);
  if (direct_storage_image_depth_pipeline_) dfn.vkDestroyPipeline(device, direct_storage_image_depth_pipeline_, nullptr);
  if (direct_storage_image_pipeline_layout_) dfn.vkDestroyPipelineLayout(device, direct_storage_image_pipeline_layout_, nullptr);
  if (direct_storage_image_descriptor_pool_) dfn.vkDestroyDescriptorPool(device, direct_storage_image_descriptor_pool_, nullptr);
  if (direct_storage_image_descriptor_set_layout_) dfn.vkDestroyDescriptorSetLayout(device, direct_storage_image_descriptor_set_layout_, nullptr);

  if (vma_allocator_ != VK_NULL_HANDLE) {
    for (auto& job : parity_readbacks_) {
      REXGPU_ERROR("[DIRECT_IMAGE_PARITY_ERROR] reason=not_observed_complete_before_shutdown frame={}", job.frame);
      vmaDestroyBuffer(vma_allocator_, job.buffer, job.allocation);
    }
    for (auto& job : direct_resolve_parity_readbacks_) {
      REXGPU_ERROR("[DIRECT_HOST_RESOLVE_PARITY_ERROR] reason=not_observed_complete_before_shutdown frame={}", job.frame);
      if (job.candidate_buffer) vmaDestroyBuffer(vma_allocator_, job.candidate_buffer, job.candidate_allocation);
      if (job.readback_buffer) vmaDestroyBuffer(vma_allocator_, job.readback_buffer, job.readback_allocation);
    }
    vmaDestroyAllocator(vma_allocator_);
  }
}

void VulkanTextureCache::ClearCache() {
  pending_fused_resolve_image_.reset();
  retired_fused_resolve_images_.clear();
  fused_image_pool_.clear();
  fused_producer_by_resource_.clear();
  resolved_images_.Clear();
  DrainResolvedImageInvalidations();
  TextureCache::ClearCache();
}

VulkanTextureCache::FusedResolveImage VulkanTextureCache::PrepareFusedResolveImage(
    uint32_t base, uint32_t length, uint32_t width, uint32_t height, xenos::Endian endian) {
  if (!REXCVAR_GET(gpu_fused_resolve_sampled_image) || pending_fused_resolve_image_) return {};
  DrainResolvedImageInvalidations();
  const TextureKey* observed = FindUniqueResolvedTextureKey(base, length);
  if (!observed) return {};
  TextureKey key = *observed;
  if (key.dimension != xenos::DataDimension::k2DOrStacked || key.format != xenos::TextureFormat::k_8_8_8_8 ||
      key.signed_separate || key.packed_mips || key.mip_max_level ||
      key.depth_or_array_size_minus_1 || !key.tiled || !key.scaled_resolve ||
      key.endianness != endian || width != 320 || height != 184 ||
      key.GetWidth() != 320 || key.GetHeight() != 180 || length != 245760) return {};
  std::unique_ptr<Texture> image;
  if (!fused_image_pool_.empty()) {
    image = std::move(fused_image_pool_.back());
    fused_image_pool_.pop_back();
  } else {
    image = CreateTextureResource(key, false);
  }
  if (!image) return {};
  auto& vk_image = static_cast<VulkanTexture&>(*image);
  if (!vk_image.direct_storage_image()) return {};
  FusedResolveImage result{vk_image.image(), vk_image.direct_storage_image_descriptor_set(),
                           key.GetWidth() * draw_resolution_scale_x(),
                           key.GetHeight() * draw_resolution_scale_y()};
  pending_fused_resolve_image_ = std::move(image);
  return result;
}

void VulkanTextureCache::PublishFusedResolveImage() {
  if (!pending_fused_resolve_image_) return;
  DrainResolvedImageInvalidations();
  auto& image = static_cast<VulkanTexture&>(*pending_fused_resolve_image_);
  image.SetUsage(VulkanTexture::Usage::kComputeStorageWrite);
  const TextureKey key = image.key();
  VmaAllocationInfo allocation_info;
  vmaGetAllocationInfo(vma_allocator_, image.image_resource_->allocation, &allocation_info);
  resolved_images_.RetainForSubmission(image.image_resource_, command_processor_.GetCurrentSubmission());
  resolved_images_.Publish(key.base_page << 12, image.GetGuestBaseSize(), key,
                           image.image_resource_, allocation_info.size);
  if (command_processor_.IsResolveChainAuditFrame()) {
    REXGPU_INFO("[FUSED_IMAGE_PUBLISHED] frame={} base=0x{:08X} guest_bytes={} image={}x{}",
                command_processor_.GetCurrentFrame(), uint32_t(key.base_page) << 12,
                image.GetGuestBaseSize(), key.GetWidth() * draw_resolution_scale_x(),
                key.GetHeight() * draw_resolution_scale_y());
  }
  fused_producer_by_resource_[image.image_resource_.get()] = &image;
  retired_fused_resolve_images_.emplace_back(command_processor_.GetCurrentSubmission(),
                                              std::move(pending_fused_resolve_image_));
}

void VulkanTextureCache::SwapTextureImage(VulkanTexture& a, VulkanTexture& b) {
  a.image_resource_.swap(b.image_resource_);
  a.views_.swap(b.views_);
  a.texture_3d_as_2d_.swap(b.texture_3d_as_2d_);
  std::swap(a.image_view_3d_as_2d_unsigned_, b.image_view_3d_as_2d_unsigned_);
  std::swap(a.image_view_3d_as_2d_signed_, b.image_view_3d_as_2d_signed_);
  std::swap(a.direct_storage_image_view_, b.direct_storage_image_view_);
  std::swap(a.direct_storage_image_descriptor_set_, b.direct_storage_image_descriptor_set_);
}

bool VulkanTextureCache::TryReuseFusedResolveImage(VulkanTexture& texture, bool load_base,
                                                   bool load_mips) {
  if (!REXCVAR_GET(gpu_fused_resolve_sampled_image) || fused_image_parity_reentry_ ||
      !load_base || load_mips) return false;
  const TextureKey key = texture.key();
  auto image = resolved_images_.Find(key.base_page << 12, texture.GetGuestBaseSize(), key);
  if (!image) return false;
  if (texture.image_resource_ != image) {
    auto producer_it = fused_producer_by_resource_.find(image.get());
    if (producer_it == fused_producer_by_resource_.end() || !producer_it->second) return false;
    VulkanTexture* producer = producer_it->second;
    fused_producer_by_resource_.erase(producer_it);
    SwapTextureImage(texture, *producer);
    fused_producer_by_resource_[producer->image_resource_.get()] = producer;
  }
  resolved_images_.RetainForSubmission(std::move(image), command_processor_.GetCurrentSubmission());
  texture.MarkAsUsed();
  if (command_processor_.IsResolveChainAuditFrame()) {
    REXGPU_INFO("[FUSED_IMAGE_ADOPTED] frame={} base=0x{:08X} guest_bytes={}",
                command_processor_.GetCurrentFrame(), uint32_t(key.base_page) << 12,
                texture.GetGuestBaseSize());
  }
  if (REXCVAR_GET(gpu_fused_resolve_image_parity) && key.GetWidth() == 320 &&
      key.GetHeight() == 180 && texture.GetGuestBaseSize() == 245760 &&
      command_processor_.IsResolveChainAuditFrame() && fused_image_parity_recorded_ < 7) {
    auto legacy = CreateTextureResource(key, false);
    bool loaded = false;
    if (legacy) {
      fused_image_parity_reentry_ = true;
      loaded = LoadTextureDataFromResidentMemoryImpl(*legacy, true, false);
      fused_image_parity_reentry_ = false;
    }
    if (!loaded || !RecordFusedImageParity(texture, static_cast<VulkanTexture&>(*legacy))) {
      REXGPU_ERROR("[FUSED_IMAGE_PARITY_ERROR] frame={} unable to compare", command_processor_.GetCurrentFrame());
    } else {
      ++fused_image_parity_recorded_;
    }
    if (legacy) retired_fused_resolve_images_.emplace_back(command_processor_.GetCurrentSubmission(),
                                                           std::move(legacy));
  }
  return true;
}

bool VulkanTextureCache::MakeFusedResolveImageWritable(VulkanTexture& texture) {
  if (texture.image_resource_.use_count() == 1) return true;
  auto replacement = CreateTextureResource(texture.key(), false);
  if (!replacement) return false;
  SwapTextureImage(texture, static_cast<VulkanTexture&>(*replacement));
  retired_fused_resolve_images_.emplace_back(command_processor_.GetCurrentSubmission(),
                                              std::move(replacement));
  return true;
}

bool VulkanTextureCache::RecordFusedImageParity(VulkanTexture& candidate, VulkanTexture& legacy) {
  const auto key = candidate.key();
  const uint32_t width = key.GetWidth() * draw_resolution_scale_x();
  const uint32_t height = key.GetHeight() * draw_resolution_scale_y();
  const VkDeviceSize bytes = VkDeviceSize(width) * height * 4;
  VkBufferCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  info.size = bytes * 2;
  info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  VmaAllocationCreateInfo allocation_info{};
  allocation_info.usage = VMA_MEMORY_USAGE_AUTO;
  allocation_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VkBuffer buffer;
  VmaAllocation allocation;
  VmaAllocationInfo mapped;
  if (vmaCreateBuffer(vma_allocator_, &info, &allocation_info, &buffer, &allocation, &mapped) != VK_SUCCESS) return false;
  std::memset(mapped.pMappedData, 0xA5, size_t(bytes));
  std::memset(static_cast<uint8_t*>(mapped.pMappedData) + bytes, 0x5A, size_t(bytes));
  if (vmaFlushAllocation(vma_allocator_, allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS) {
    vmaDestroyBuffer(vma_allocator_, buffer, allocation);
    return false;
  }
  VulkanTexture* images[] = {&candidate, &legacy};
  for (uint32_t i = 0; i < 2; ++i) {
    VkPipelineStageFlags stage;
    VkAccessFlags access;
    VkImageLayout layout;
    GetTextureUsageMasks(images[i]->image_resource_->usage, stage, access, layout);
    command_processor_.PushImageMemoryBarrier(
        images[i]->image(), ui::vulkan::util::InitializeSubresourceRange(), stage,
        VK_PIPELINE_STAGE_TRANSFER_BIT, access, VK_ACCESS_TRANSFER_READ_BIT, layout,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
    command_processor_.SubmitBarriers(true);
    VkBufferImageCopy region{};
    region.bufferOffset = bytes * i;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.layerCount = 1;
    region.imageExtent = {width, height, 1};
    command_processor_.deferred_command_buffer().CmdVkCopyImageToBuffer(
        images[i]->image(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, buffer, region);
    command_processor_.PushImageMemoryBarrier(
        images[i]->image(), ui::vulkan::util::InitializeSubresourceRange(),
        VK_PIPELINE_STAGE_TRANSFER_BIT, stage, VK_ACCESS_TRANSFER_READ_BIT, access,
        VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, layout);
  }
  command_processor_.PushBufferMemoryBarrier(buffer, 0, bytes * 2,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
  fused_image_parity_readbacks_.push_back({buffer, allocation, mapped.pMappedData,
      command_processor_.GetCurrentSubmission(), bytes, command_processor_.GetCurrentFrame(),
      uint32_t(key.base_page) << 12});
  return true;
}

void VulkanTextureCache::CompleteFusedImageParity(uint64_t completed) {
  auto it = fused_image_parity_readbacks_.begin();
  while (it != fused_image_parity_readbacks_.end()) {
    if (it->submission > completed) { ++it; continue; }
    if (vmaInvalidateAllocation(vma_allocator_, it->allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS) {
      REXGPU_ERROR("[FUSED_IMAGE_PARITY_ERROR] frame={} invalidate failed", it->frame);
    } else {
      const auto* data = static_cast<const uint8_t*>(it->mapping);
      uint64_t mismatches = 0;
      for (VkDeviceSize i = 0; i < it->bytes; ++i) mismatches += data[i] != data[it->bytes + i];
      REXGPU_INFO("[FUSED_IMAGE_PARITY] frame={} submission={} base=0x{:08X} compared_bytes={} mismatched_bytes={}",
                  it->frame, it->submission, it->base, it->bytes, mismatches);
    }
    vmaDestroyBuffer(vma_allocator_, it->buffer, it->allocation);
    it = fused_image_parity_readbacks_.erase(it);
  }
}

void VulkanTextureCache::InvalidateResolvedImages(uint32_t address, uint32_t length) {
  if (!REXCVAR_GET(gpu_fused_resolve_sampled_image) || !length) return;
  std::lock_guard<std::mutex> lock(resolved_image_invalidation_mutex_);
  if (resolved_image_invalidations_.size() >= 256) {
    resolved_image_invalidations_.assign(1, {0, 0x20000000u});
  } else {
    resolved_image_invalidations_.emplace_back(address & 0x1FFFFFFFu, length);
  }
  has_resolved_image_invalidations_.store(true, std::memory_order_release);
}

void VulkanTextureCache::DrainResolvedImageInvalidations() {
  if (!has_resolved_image_invalidations_.load(std::memory_order_acquire)) return;
  std::vector<std::pair<uint32_t, uint32_t>> pending;
  {
    std::lock_guard<std::mutex> lock(resolved_image_invalidation_mutex_);
    pending.swap(resolved_image_invalidations_);
    has_resolved_image_invalidations_.store(false, std::memory_order_release);
  }
  for (const auto& range : pending) resolved_images_.Invalidate(range.first, range.second);
}

void VulkanTextureCache::BeginSubmission(uint64_t new_submission_index) {
  TextureCache::BeginSubmission(new_submission_index);
  DrainResolvedImageInvalidations();
  const uint64_t completed = command_processor_.GetCompletedSubmission();
  CompleteFusedImageParity(completed);
  resolved_images_.CompletedSubmissionUpdated(completed);
  for (auto it = retired_fused_resolve_images_.begin(); it != retired_fused_resolve_images_.end(); ) {
    if (it->first <= completed) {
      if (it->second) {
        auto& vk_tex = static_cast<VulkanTexture&>(*it->second);
        fused_producer_by_resource_.erase(vk_tex.image_resource_.get());
        if (fused_image_pool_.size() < 8) {
          fused_image_pool_.push_back(std::move(it->second));
        }
      }
      it = retired_fused_resolve_images_.erase(it);
    } else {
      ++it;
    }
  }

  if (!null_images_cleared_ && null_image_2d_array_cube_ != VK_NULL_HANDLE &&
      null_image_3d_ != VK_NULL_HANDLE) {
    VkImage null_images[] = {null_image_2d_array_cube_, null_image_3d_};
    VkImageSubresourceRange null_image_subresource_range(
        ui::vulkan::util::InitializeSubresourceRange());
    for (size_t i = 0; i < rex::countof(null_images); ++i) {
      command_processor_.PushImageMemoryBarrier(
          null_images[i], null_image_subresource_range, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
          VK_ACCESS_TRANSFER_WRITE_BIT, VK_IMAGE_LAYOUT_UNDEFINED,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
          false);
    }
    command_processor_.SubmitBarriers(true);
    DeferredCommandBuffer& command_buffer = command_processor_.deferred_command_buffer();
    VkClearColorValue null_image_clear_color = kInvalidTextureFetchFallbackColor;
    for (size_t i = 0; i < rex::countof(null_images); ++i) {
      command_buffer.CmdVkClearColorImage(null_images[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                          &null_image_clear_color, 1,
                                          &null_image_subresource_range);
    }
    for (size_t i = 0; i < rex::countof(null_images); ++i) {
      command_processor_.PushImageMemoryBarrier(
          null_images[i], null_image_subresource_range, VK_PIPELINE_STAGE_TRANSFER_BIT,
          guest_shader_pipeline_stages_, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
          VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
          VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
    }
    null_images_cleared_ = true;
  }
}

void VulkanTextureCache::BeginFrame() {
  ProcessParityReadbacks();
  ProcessDirectResolveParityReadbacks();
  if (IsInGpuFrameAuditWindow()) {
    direct_storage_image_audit_frame_start_ = GetDirectStorageImageStats();
  }
  TextureCache::BeginFrame();

  std::memset(unsupported_format_features_used_, 0, sizeof(unsupported_format_features_used_));
}

void VulkanTextureCache::EndFrame() {
  if (IsInGpuFrameAuditWindow()) {
    const DirectStorageImageStats end_stats = GetDirectStorageImageStats();
    const DirectStorageImageStats& start_stats = direct_storage_image_audit_frame_start_;
    const uint64_t frame = command_processor_.GetCurrentFrame();
    REXGPU_INFO(
        "[DIRECT_STORAGE_IMAGE] frame={} direct_loads={} fallbacks={} copy_bytes_avoided={} "
        "legacy_copy_bytes={}",
        frame, end_stats.direct_loads - start_stats.direct_loads,
        end_stats.fallbacks - start_stats.fallbacks,
        end_stats.copy_bytes_avoided - start_stats.copy_bytes_avoided,
        end_stats.legacy_copy_bytes - start_stats.legacy_copy_bytes);
    for (size_t i = 0; i < size_t(DirectStorageImageFallback::kCount); ++i) {
      const uint64_t delta = end_stats.fallbacks_by_reason[i] - start_stats.fallbacks_by_reason[i];
      if (delta) {
        REXGPU_INFO("[DIRECT_STORAGE_IMAGE_FALLBACK] frame={} reason={} count={}", frame,
                    DirectStorageImageFallbackName(DirectStorageImageFallback(i)), delta);
      }
    }
  }
  bool unsupported_header_written = false;
  for (uint32_t i = 0; i < 64; ++i) {
    uint32_t unsupported_features = unsupported_format_features_used_[i];
    if (!unsupported_features) {
      continue;
    }
    if (!unsupported_header_written) {
      REXGPU_ERROR("Unsupported texture formats used in the frame:");
      unsupported_header_written = true;
    }
    REXGPU_ERROR("* {}{}{}{}", FormatInfo::Get(xenos::TextureFormat(i))->name,
                 unsupported_features & kUnsupportedResourceBit ? " resource" : "",
                 unsupported_features & kUnsupportedUnormBit ? " unsigned" : "",
                 unsupported_features & kUnsupportedSnormBit ? " signed" : "");
    unsupported_format_features_used_[i] = 0;
  }
}

void VulkanTextureCache::RequestTextures(uint32_t used_texture_mask) {
#if XE_GPU_FINE_GRAINED_DRAW_SCOPES
  SCOPE_profile_cpu_f("gpu");
#endif  // XE_GPU_FINE_GRAINED_DRAW_SCOPES

  TextureCache::RequestTextures(used_texture_mask);

  // Pre-create 3D-as-2D wrappers while still in texture request phase.
  if (REXCVAR_GET(gpu_3d_to_2d_texture)) {
    uint32_t textures_3d = used_texture_mask;
    uint32_t index_3d;
    while (rex::bit_scan_forward(textures_3d, &index_3d)) {
      textures_3d &= ~(uint32_t(1) << index_3d);
      const TextureBinding* binding = GetValidTextureBinding(index_3d);
      if (!binding || binding->key.dimension != xenos::DataDimension::k3D) {
        continue;
      }
      VulkanTexture* texture = static_cast<VulkanTexture*>(binding->texture);
      if (texture) {
        texture->GetOrCreate3DAs2DImageView(false, binding->host_swizzle);
      }
      VulkanTexture* texture_signed = static_cast<VulkanTexture*>(binding->texture_signed);
      if (texture_signed) {
        texture_signed->GetOrCreate3DAs2DImageView(true, binding->host_swizzle);
      }
    }
  }

  // Transition the textures into the needed usage.
  VkPipelineStageFlags dst_stage_mask;
  VkAccessFlags dst_access_mask;
  VkImageLayout new_layout;
  GetTextureUsageMasks(VulkanTexture::Usage::kGuestShaderSampled, dst_stage_mask, dst_access_mask,
                       new_layout);
  uint32_t textures_remaining = used_texture_mask;
  uint32_t index;
  while (rex::bit_scan_forward(textures_remaining, &index)) {
    textures_remaining &= ~(uint32_t(1) << index);
    const TextureBinding* binding = GetValidTextureBinding(index);
    if (!binding) {
      continue;
    }
    VulkanTexture* binding_texture = static_cast<VulkanTexture*>(binding->texture);
    if (binding_texture != nullptr) {
      // Will be referenced by the command buffer, so mark as used.
      binding_texture->MarkAsUsed();
      VulkanTexture::Usage old_usage =
          binding_texture->SetUsage(VulkanTexture::Usage::kGuestShaderSampled);
      if (old_usage != VulkanTexture::Usage::kGuestShaderSampled) {
        VkPipelineStageFlags src_stage_mask;
        VkAccessFlags src_access_mask;
        VkImageLayout old_layout;
        GetTextureUsageMasks(old_usage, src_stage_mask, src_access_mask, old_layout);
        command_processor_.PushImageMemoryBarrier(
            binding_texture->image(), ui::vulkan::util::InitializeSubresourceRange(),
            src_stage_mask, dst_stage_mask, src_access_mask, dst_access_mask, old_layout,
            new_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, true,
            VulkanCommandProcessor::BarrierRequester::kTextureCache);
      }
    }
    VulkanTexture* binding_texture_signed = static_cast<VulkanTexture*>(binding->texture_signed);
    if (binding_texture_signed != nullptr) {
      binding_texture_signed->MarkAsUsed();
      VulkanTexture::Usage old_usage =
          binding_texture_signed->SetUsage(VulkanTexture::Usage::kGuestShaderSampled);
      if (old_usage != VulkanTexture::Usage::kGuestShaderSampled) {
        VkPipelineStageFlags src_stage_mask;
        VkAccessFlags src_access_mask;
        VkImageLayout old_layout;
        GetTextureUsageMasks(old_usage, src_stage_mask, src_access_mask, old_layout);
        command_processor_.PushImageMemoryBarrier(
            binding_texture_signed->image(), ui::vulkan::util::InitializeSubresourceRange(),
            src_stage_mask, dst_stage_mask, src_access_mask, dst_access_mask, old_layout,
            new_layout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, true,
            VulkanCommandProcessor::BarrierRequester::kTextureCache);
      }
    }
  }
}

VkImageView VulkanTextureCache::GetActiveBindingOrNullImageView(uint32_t fetch_constant_index,
                                                                xenos::FetchOpDimension dimension,
                                                                bool is_signed) {
  VkImageView image_view = VK_NULL_HANDLE;
  const TextureBinding* binding = GetValidTextureBinding(fetch_constant_index);
  if (binding && AreDimensionsCompatible(dimension, binding->key.dimension)) {
    bool force_special_view =
        binding->key.dimension == xenos::DataDimension::k3D &&
        (dimension == xenos::FetchOpDimension::k1D || dimension == xenos::FetchOpDimension::k2D);
    if (force_special_view) {
      Texture* texture = nullptr;
      bool use_signed = is_signed && texture_util::IsAnySignSigned(binding->swizzled_signs);
      if (use_signed && IsSignedVersionSeparateForFormat(binding->key)) {
        texture = binding->texture_signed;
      } else {
        texture = binding->texture;
      }
      if (texture) {
        image_view = static_cast<VulkanTexture*>(texture)->GetOrCreate3DAs2DImageView(
            use_signed, binding->host_swizzle);
      }
    } else {
      const VulkanTextureBinding& vulkan_binding = vulkan_texture_bindings_[fetch_constant_index];
      image_view =
          is_signed ? vulkan_binding.image_view_signed : vulkan_binding.image_view_unsigned;
    }
  }
  // Share the latched fragment diagnostic gate, and bound both duplicates and
  // total output (at most 32 entries for this texture cache's lifetime).
  if ((fetch_constant_index == 11 || fetch_constant_index == 13 || fetch_constant_index == 14) &&
      diagnostic_descriptor_log_.ShouldLog(
          command_processor_.diagnostic_storage_enabled(),
          {fetch_constant_index, binding ? uint32_t(binding->key.base_page) : 0u,
           binding ? uint32_t(binding->key.format) : 0,
           binding ? uint32_t(binding->key.dimension) : 0, is_signed, binding != nullptr})) {
    VulkanTexture* texture_unsigned =
        binding ? static_cast<VulkanTexture*>(binding->texture) : nullptr;
    VulkanTexture* texture_signed =
        binding ? static_cast<VulkanTexture*>(binding->texture_signed) : nullptr;
    const VulkanTextureBinding& views = vulkan_texture_bindings_[fetch_constant_index];
    const HostFormatPair* host_format = binding ? &GetHostFormatPair(binding->key) : nullptr;
    REXGPU_INFO(
        "TF_DESCRIPTOR deduplicated budget=32 fetch={} signed={} dimension={} binding={} key_base=0x{:X} format={} "
        "size={}x{} mips={} mip_page=0x{:X} tiled={} packed_mips={} scaled_resolve={} "
        "host_format_unsigned={} host_format_signed={} "
        "texture_unsigned={} image_unsigned={} view_unsigned={} "
        "texture_signed={} image_signed={} view_signed={} selected_view={}",
        fetch_constant_index, is_signed, binding ? uint32_t(binding->key.dimension) : 0,
        static_cast<const void*>(binding), binding ? binding->key.base_page : 0,
        binding ? uint32_t(binding->key.format) : 0,
        binding ? binding->key.GetWidth() : 0, binding ? binding->key.GetHeight() : 0,
        binding ? uint32_t(binding->key.mip_max_level) + 1 : 0,
        binding ? binding->key.mip_page : 0,
        binding ? uint32_t(binding->key.tiled) : 0,
        binding ? uint32_t(binding->key.packed_mips) : 0,
        binding ? uint32_t(binding->key.scaled_resolve) : 0,
        host_format ? uint32_t(host_format->format_unsigned.format) : 0,
        host_format ? uint32_t(host_format->format_signed.format) : 0,
        static_cast<const void*>(texture_unsigned),
        reinterpret_cast<const void*>(texture_unsigned ? texture_unsigned->image() : VK_NULL_HANDLE),
        reinterpret_cast<const void*>(views.image_view_unsigned),
        static_cast<const void*>(texture_signed),
        reinterpret_cast<const void*>(texture_signed ? texture_signed->image() : VK_NULL_HANDLE),
        reinterpret_cast<const void*>(views.image_view_signed),
        reinterpret_cast<const void*>(image_view));
  }
  if (image_view != VK_NULL_HANDLE) {
    return image_view;
  }
  if (command_processor_.GetVulkanDevice()->properties().nullDescriptor) {
    return VK_NULL_HANDLE;
  }
  switch (dimension) {
    case xenos::FetchOpDimension::k3DOrStacked:
      return null_image_view_3d_;
    case xenos::FetchOpDimension::kCube:
      return null_image_view_cube_;
    default:
      return null_image_view_2d_array_;
  }
}

VulkanTextureCache::SamplerParameters VulkanTextureCache::GetSamplerParameters(
    const VulkanShader::SamplerBinding& binding, bool force_min_mag_linear) const {
  const auto& regs = register_file();
  xenos::xe_gpu_texture_fetch_t fetch = regs.GetTextureFetch(binding.fetch_constant);

  SamplerParameters parameters;

  xenos::ClampMode fetch_clamp_x, fetch_clamp_y, fetch_clamp_z;
  texture_util::GetClampModesForDimension(fetch, fetch_clamp_x, fetch_clamp_y, fetch_clamp_z);
  parameters.clamp_x = NormalizeClampMode(fetch_clamp_x);
  parameters.clamp_y = NormalizeClampMode(fetch_clamp_y);
  parameters.clamp_z = NormalizeClampMode(fetch_clamp_z);
  if (xenos::ClampModeUsesBorder(parameters.clamp_x) ||
      xenos::ClampModeUsesBorder(parameters.clamp_y) ||
      xenos::ClampModeUsesBorder(parameters.clamp_z)) {
    parameters.border_color = fetch.border_color;
  } else {
    parameters.border_color = xenos::BorderColor::k_ABGR_Black;
  }

  xenos::TextureFilter mag_filter = binding.mag_filter == xenos::TextureFilter::kUseFetchConst
                                        ? fetch.mag_filter
                                        : binding.mag_filter;
  parameters.mag_linear = mag_filter == xenos::TextureFilter::kLinear;
  xenos::TextureFilter min_filter = binding.min_filter == xenos::TextureFilter::kUseFetchConst
                                        ? fetch.min_filter
                                        : binding.min_filter;
  parameters.min_linear = min_filter == xenos::TextureFilter::kLinear;
  xenos::TextureFilter mip_filter = binding.mip_filter == xenos::TextureFilter::kUseFetchConst
                                        ? fetch.mip_filter
                                        : binding.mip_filter;
  parameters.mip_linear = mip_filter == xenos::TextureFilter::kLinear ||
                          ((draw_resolution_scale_x() > 1 || draw_resolution_scale_y() > 1) &&
                           mip_filter == xenos::TextureFilter::kPoint);
  xenos::AnisoFilter aniso_filter = binding.aniso_filter == xenos::AnisoFilter::kUseFetchConst
                                        ? fetch.aniso_filter
                                        : binding.aniso_filter;
  aniso_filter = std::min(aniso_filter, max_anisotropy_);
  if (force_min_mag_linear) {
    parameters.mag_linear = 1;
    parameters.min_linear = 1;
  }
  if (parameters.mag_linear || parameters.min_linear || parameters.mip_linear ||
      aniso_filter != xenos::AnisoFilter::kDisabled) {
    // Check if the texture is actually filterable on the host.
    bool linear_filterable = true;
    TextureKey texture_key;
    uint8_t texture_swizzled_signs;
    BindingInfoFromFetchConstant(fetch, texture_key, &texture_swizzled_signs);
    if (texture_key.is_valid) {
      const HostFormatPair& host_format_pair = GetHostFormatPair(texture_key);
      if ((texture_util::IsAnySignNotSigned(texture_swizzled_signs) &&
           !host_format_pair.format_unsigned.linear_filterable) ||
          (texture_util::IsAnySignSigned(texture_swizzled_signs) &&
           !host_format_pair.format_signed.linear_filterable)) {
        linear_filterable = false;
      }
    } else {
      linear_filterable = false;
    }
    if (!linear_filterable) {
      parameters.mag_linear = 0;
      parameters.min_linear = 0;
      parameters.mip_linear = 0;
      aniso_filter = xenos::AnisoFilter::kDisabled;
    }
  }
  bool min_mag_linear = parameters.mag_linear && parameters.min_linear;
  bool mip_filter_bilinear_or_trilinear =
      mip_filter == xenos::TextureFilter::kPoint || mip_filter == xenos::TextureFilter::kLinear;
  parameters.mip_base_map = mip_filter == xenos::TextureFilter::kBaseMap;
  uint32_t mip_min_level, mip_max_level;
  texture_util::GetSubresourcesFromFetchConstant(fetch, nullptr, nullptr, nullptr, nullptr, nullptr,
                                                 &mip_min_level, &mip_max_level);
  parameters.mip_min_level = mip_min_level;
  bool has_mips = mip_max_level > mip_min_level;
  int32_t anisotropic_override = REXCVAR_GET(anisotropic_override);
  if (!force_min_mag_linear && anisotropic_override > -1 && anisotropic_override < 6 && has_mips &&
      !parameters.mip_base_map && min_mag_linear && mip_filter_bilinear_or_trilinear) {
    aniso_filter = xenos::AnisoFilter(anisotropic_override);
  }
  aniso_filter = std::min(aniso_filter, max_anisotropy_);
  if (aniso_filter != xenos::AnisoFilter::kDisabled) {
    // Match D3D12 behavior: anisotropic filtering forces linear min/mag/mip.
    parameters.mag_linear = 1;
    parameters.min_linear = 1;
    parameters.mip_linear = 1;
  }
  parameters.aniso_filter = aniso_filter;

  return parameters;
}

VkSampler VulkanTextureCache::UseSampler(SamplerParameters parameters, bool& has_overflown_out) {
  assert_true(command_processor_.submission_open());
  uint64_t submission_current = command_processor_.GetCurrentSubmission();

  if (last_sampler_parameters_.value == parameters.value && last_sampler_ != VK_NULL_HANDLE) {
    has_overflown_out = false;
    return last_sampler_;
  }

  // Try to find an existing sampler.
  auto it_existing = samplers_.find(parameters);
  if (it_existing != samplers_.end()) {
    std::pair<const SamplerParameters, Sampler>& sampler = *it_existing;
    assert_true(sampler.second.last_usage_submission <= submission_current);
    // This is called very frequently, don't relink unless needed for caching.
    if (sampler.second.last_usage_submission < submission_current) {
      // Move to the front of the LRU queue.
      sampler.second.last_usage_submission = submission_current;
      if (sampler.second.used_next) {
        if (sampler.second.used_previous) {
          sampler.second.used_previous->second.used_next = sampler.second.used_next;
        } else {
          sampler_used_first_ = sampler.second.used_next;
        }
        sampler.second.used_next->second.used_previous = sampler.second.used_previous;
        sampler.second.used_previous = sampler_used_last_;
        sampler.second.used_next = nullptr;
        sampler_used_last_->second.used_next = &sampler;
        sampler_used_last_ = &sampler;
      }
    }
    has_overflown_out = false;
    last_sampler_parameters_ = parameters;
    last_sampler_ = sampler.second.sampler;
    return sampler.second.sampler;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  // See if an existing sampler can be destroyed to create space for the new
  // one.
  if (samplers_.size() >= sampler_max_count_) {
    assert_not_null(sampler_used_first_);
    if (!sampler_used_first_) {
      has_overflown_out = false;
      return VK_NULL_HANDLE;
    }
    if (sampler_used_first_->second.last_usage_submission >
        command_processor_.GetCompletedSubmission()) {
      has_overflown_out = true;
      return VK_NULL_HANDLE;
    }
    auto it_reuse = samplers_.find(sampler_used_first_->first);
    if (sampler_used_first_->second.uses_custom_border_color) {
      assert_true(custom_border_color_sampler_count_ > 0);
      if (custom_border_color_sampler_count_) {
        --custom_border_color_sampler_count_;
      }
    }
    dfn.vkDestroySampler(device, sampler_used_first_->second.sampler, nullptr);
    if (sampler_used_first_->second.used_next) {
      sampler_used_first_->second.used_next->second.used_previous =
          sampler_used_first_->second.used_previous;
    } else {
      sampler_used_last_ = sampler_used_first_->second.used_previous;
    }
    sampler_used_first_ = sampler_used_first_->second.used_next;
    assert_true(it_reuse != samplers_.end());
    if (it_reuse != samplers_.end()) {
      if (it_reuse->first.value == last_sampler_parameters_.value) {
        last_sampler_parameters_.value = UINT32_MAX;
        last_sampler_ = VK_NULL_HANDLE;
      }
      // This destroys the Sampler object.
      samplers_.erase(it_reuse);
      COUNT_profile_set("gpu/texture_cache/vulkan/samplers", samplers_.size());
    } else {
      has_overflown_out = false;
      return VK_NULL_HANDLE;
    }
  }

  // Create a new sampler and make it the least recently used.
  // The values are normalized, and unsupported ones are excluded, in
  // GetSamplerParameters.
  VkSamplerCreateInfo sampler_create_info = {};
  sampler_create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
  if (vulkan_device->properties().nonSeamlessCubeMap && REXCVAR_GET(non_seamless_cube_map)) {
    sampler_create_info.flags |= VK_SAMPLER_CREATE_NON_SEAMLESS_CUBE_MAP_BIT_EXT;
  }
  sampler_create_info.magFilter = parameters.mag_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  sampler_create_info.minFilter = parameters.min_linear ? VK_FILTER_LINEAR : VK_FILTER_NEAREST;
  sampler_create_info.mipmapMode =
      parameters.mip_linear ? VK_SAMPLER_MIPMAP_MODE_LINEAR : VK_SAMPLER_MIPMAP_MODE_NEAREST;
  static const VkSamplerAddressMode kAddressModeMap[] = {
      // kRepeat
      VK_SAMPLER_ADDRESS_MODE_REPEAT,
      // kMirroredRepeat
      VK_SAMPLER_ADDRESS_MODE_MIRRORED_REPEAT,
      // kClampToEdge
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      // kMirrorClampToEdge
      VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
      // kClampToHalfway
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
      // kMirrorClampToHalfway
      VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
      // kClampToBorder
      VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_BORDER,
      // kMirrorClampToBorder
      VK_SAMPLER_ADDRESS_MODE_MIRROR_CLAMP_TO_EDGE,
  };
  sampler_create_info.addressModeU = kAddressModeMap[uint32_t(parameters.clamp_x)];
  sampler_create_info.addressModeV = kAddressModeMap[uint32_t(parameters.clamp_y)];
  sampler_create_info.addressModeW = kAddressModeMap[uint32_t(parameters.clamp_z)];
  // LOD biasing is performed in shaders.
  if (parameters.aniso_filter != xenos::AnisoFilter::kDisabled) {
    sampler_create_info.anisotropyEnable = VK_TRUE;
    sampler_create_info.maxAnisotropy =
        float(UINT32_C(1) << (uint32_t(parameters.aniso_filter) -
                              uint32_t(xenos::AnisoFilter::kMax_1_1)));
  }
  sampler_create_info.minLod = float(parameters.mip_min_level);
  if (parameters.mip_base_map) {
    assert_false(parameters.mip_linear);
    sampler_create_info.maxLod = sampler_create_info.minLod + 0.25f;
  } else {
    sampler_create_info.maxLod = VK_LOD_CLAMP_NONE;
  }
  bool uses_custom_border_color = false;
  VkSamplerCustomBorderColorCreateInfoEXT sampler_custom_border_color_create_info = {};
  const bool custom_border_colors_supported = vulkan_device->properties().customBorderColors;
  const bool custom_border_color_slots_available =
      custom_border_color_sampler_count_ < vulkan_device->properties().maxCustomBorderColorSamplers;
  switch (parameters.border_color) {
    case xenos::BorderColor::k_ABGR_White:
      sampler_create_info.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
      break;
    case xenos::BorderColor::k_ACBYCR_Black:
    case xenos::BorderColor::k_ACBCRY_Black:
      if (custom_border_colors_supported && custom_border_color_slots_available) {
        sampler_create_info.borderColor = VK_BORDER_COLOR_FLOAT_CUSTOM_EXT;
        sampler_custom_border_color_create_info.sType =
            VK_STRUCTURE_TYPE_SAMPLER_CUSTOM_BORDER_COLOR_CREATE_INFO_EXT;
        sampler_custom_border_color_create_info.pNext = sampler_create_info.pNext;
        sampler_custom_border_color_create_info.customBorderColor.float32[0] =
            parameters.border_color == xenos::BorderColor::k_ACBYCR_Black ? 0.5f : 0.0f;
        sampler_custom_border_color_create_info.customBorderColor.float32[1] =
            parameters.border_color == xenos::BorderColor::k_ACBYCR_Black ? 0.0f : 0.5f;
        sampler_custom_border_color_create_info.customBorderColor.float32[2] = 0.5f;
        sampler_custom_border_color_create_info.customBorderColor.float32[3] = 0.0f;
        sampler_custom_border_color_create_info.format =
            vulkan_device->properties().customBorderColorWithoutFormat
                ? VK_FORMAT_UNDEFINED
                : VK_FORMAT_R32G32B32A32_SFLOAT;
        sampler_create_info.pNext = &sampler_custom_border_color_create_info;
        uses_custom_border_color = true;
      } else {
        static bool custom_border_color_fallback_logged = false;
        if (!custom_border_color_fallback_logged) {
          custom_border_color_fallback_logged = true;
          if (!custom_border_colors_supported) {
            REXGPU_WARN(
                "VulkanTextureCache: VK_EXT_custom_border_color support is unavailable; "
                "falling back to transparent black for YCbCr border color sampling");
          } else {
            REXGPU_WARN(
                "VulkanTextureCache: maxCustomBorderColorSamplers limit ({}) reached; "
                "falling back to transparent black for additional YCbCr border samplers",
                vulkan_device->properties().maxCustomBorderColorSamplers);
          }
        }
        sampler_create_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
      }
      break;
    default:
      sampler_create_info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
      break;
  }
  VkSampler vulkan_sampler;
  if (dfn.vkCreateSampler(device, &sampler_create_info, nullptr, &vulkan_sampler) != VK_SUCCESS) {
    REXGPU_ERROR(
        "VulkanTextureCache: Failed to create the sampler for parameters "
        "0x{:08X}",
        parameters.value);
    has_overflown_out = false;
    return VK_NULL_HANDLE;
  }
  std::pair<const SamplerParameters, Sampler>& new_sampler =
      *(samplers_
            .emplace(std::piecewise_construct, std::forward_as_tuple(parameters),
                     std::forward_as_tuple())
            .first);
  COUNT_profile_set("gpu/texture_cache/vulkan/samplers", samplers_.size());
  new_sampler.second.sampler = vulkan_sampler;
  new_sampler.second.uses_custom_border_color = uses_custom_border_color;
  new_sampler.second.last_usage_submission = submission_current;
  new_sampler.second.used_previous = sampler_used_last_;
  new_sampler.second.used_next = nullptr;
  if (uses_custom_border_color) {
    ++custom_border_color_sampler_count_;
  }
  if (sampler_used_last_) {
    sampler_used_last_->second.used_next = &new_sampler;
  } else {
    sampler_used_first_ = &new_sampler;
  }
  sampler_used_last_ = &new_sampler;
  last_sampler_parameters_ = parameters;
  last_sampler_ = vulkan_sampler;
  return vulkan_sampler;
}

uint64_t VulkanTextureCache::GetSubmissionToAwaitOnSamplerOverflow(
    uint32_t overflowed_sampler_count) const {
  if (!overflowed_sampler_count) {
    return 0;
  }
  std::pair<const SamplerParameters, Sampler>* sampler_used = sampler_used_first_;
  if (!sampler_used_first_) {
    return 0;
  }
  for (uint32_t samplers_remaining = overflowed_sampler_count - 1; samplers_remaining;
       --samplers_remaining) {
    std::pair<const SamplerParameters, Sampler>* sampler_used_next = sampler_used->second.used_next;
    if (!sampler_used_next) {
      break;
    }
    sampler_used = sampler_used_next;
  }
  return sampler_used->second.last_usage_submission;
}

VkImageView VulkanTextureCache::RequestSwapTexture(
    uint32_t& width_scaled_out, uint32_t& height_scaled_out, xenos::TextureFormat& format_out,
    uint32_t* width_unscaled_out, uint32_t* height_unscaled_out,
    bool* swap_source_needs_rb_swap_out, uint32_t* guest_base_address_out) {
  if (guest_base_address_out) {
    *guest_base_address_out = UINT32_MAX;
  }
  const auto& regs = register_file();
  xenos::xe_gpu_texture_fetch_t fetch = regs.GetTextureFetch(0);
  TextureKey key;
  BindingInfoFromFetchConstant(fetch, key, nullptr);
  if (!key.is_valid || key.base_page == 0 || key.dimension != xenos::DataDimension::k2DOrStacked) {
    return nullptr;
  }
  VulkanTexture* texture = static_cast<VulkanTexture*>(FindOrCreateTexture(key));
  if (!texture) {
    return VK_NULL_HANDLE;
  }
  if (guest_base_address_out) {
    *guest_base_address_out = key.base_page << 12;
  }
  uint32_t host_swizzle = GuestToHostSwizzle(fetch.swizzle, GetHostFormatSwizzle(key));
  if (swap_source_needs_rb_swap_out) {
    auto swizzle_component = [](uint32_t swizzle,
                                uint32_t component_index) -> xenos::XE_GPU_TEXTURE_SWIZZLE {
      return xenos::XE_GPU_TEXTURE_SWIZZLE((swizzle >> (3 * component_index)) & 0b111);
    };
    *swap_source_needs_rb_swap_out =
        swizzle_component(host_swizzle, 0) == xenos::XE_GPU_TEXTURE_SWIZZLE_B &&
        swizzle_component(host_swizzle, 1) == xenos::XE_GPU_TEXTURE_SWIZZLE_G &&
        swizzle_component(host_swizzle, 2) == xenos::XE_GPU_TEXTURE_SWIZZLE_R;
  }
  VkImageView texture_view = texture->GetView(false, host_swizzle, false);
  if (texture_view == VK_NULL_HANDLE) {
    return VK_NULL_HANDLE;
  }
  if (!LoadTextureData(*texture)) {
    return VK_NULL_HANDLE;
  }
  texture->MarkAsUsed();
  VulkanTexture::Usage old_usage = texture->SetUsage(VulkanTexture::Usage::kSwapSampled);
  if (old_usage != VulkanTexture::Usage::kSwapSampled) {
    VkPipelineStageFlags src_stage_mask, dst_stage_mask;
    VkAccessFlags src_access_mask, dst_access_mask;
    VkImageLayout old_layout, new_layout;
    GetTextureUsageMasks(old_usage, src_stage_mask, src_access_mask, old_layout);
    GetTextureUsageMasks(VulkanTexture::Usage::kSwapSampled, dst_stage_mask, dst_access_mask,
                         new_layout);
    command_processor_.PushImageMemoryBarrier(
        texture->image(), ui::vulkan::util::InitializeSubresourceRange(), src_stage_mask,
        dst_stage_mask, src_access_mask, dst_access_mask, old_layout, new_layout);
  }
  // Only texture->key, not the result of BindingInfoFromFetchConstant, contains
  // whether the texture is scaled.
  key = texture->key();
  if (width_unscaled_out) {
    *width_unscaled_out = key.GetWidth();
  }
  if (height_unscaled_out) {
    *height_unscaled_out = key.GetHeight();
  }
  width_scaled_out = key.GetWidth() * (key.scaled_resolve ? draw_resolution_scale_x() : 1);
  height_scaled_out = key.GetHeight() * (key.scaled_resolve ? draw_resolution_scale_y() : 1);
  format_out = key.format;
  return texture_view;
}

bool VulkanTextureCache::IsSignedVersionSeparateForFormat(TextureKey key) const {
  const HostFormatPair& host_format_pair = GetHostFormatPair(key);
  if (host_format_pair.format_unsigned.format == VK_FORMAT_UNDEFINED ||
      host_format_pair.format_signed.format == VK_FORMAT_UNDEFINED) {
    // Just one signedness.
    return false;
  }
  return !host_format_pair.unsigned_signed_compatible;
}

bool VulkanTextureCache::IsScaledResolveSupportedForFormat(TextureKey key) const {
  if (!IsDrawResolutionScaled()) {
    return false;
  }
  const HostFormatPair& host_format_pair = GetHostFormatPair(key);
  bool host_format_is_signed;
  if (IsSignedVersionSeparateForFormat(key)) {
    host_format_is_signed = bool(key.signed_separate);
  } else {
    host_format_is_signed = host_format_pair.format_unsigned.load_shader == kLoadShaderIndexUnknown;
  }
  LoadShaderIndex load_shader =
      (host_format_is_signed ? host_format_pair.format_signed : host_format_pair.format_unsigned)
          .load_shader;
  return load_shader != kLoadShaderIndexUnknown &&
         load_pipelines_scaled_[load_shader] != VK_NULL_HANDLE;
}

uint32_t VulkanTextureCache::GetHostFormatSwizzle(TextureKey key) const {
  return GetHostFormatPair(key).swizzle;
}

uint32_t VulkanTextureCache::GetMaxHostTextureWidthHeight(xenos::DataDimension dimension) const {
  const ui::vulkan::VulkanDevice::Properties& device_properties =
      command_processor_.GetVulkanDevice()->properties();
  switch (dimension) {
    case xenos::DataDimension::k1D:
    case xenos::DataDimension::k2DOrStacked:
      // 1D and 2D are emulated as 2D arrays.
      return device_properties.maxImageDimension2D;
    case xenos::DataDimension::k3D:
      return device_properties.maxImageDimension3D;
    case xenos::DataDimension::kCube:
      return device_properties.maxImageDimensionCube;
    default:
      assert_unhandled_case(dimension);
      return 0;
  }
}

uint32_t VulkanTextureCache::GetMaxHostTextureDepthOrArraySize(
    xenos::DataDimension dimension) const {
  const ui::vulkan::VulkanDevice::Properties& device_properties =
      command_processor_.GetVulkanDevice()->properties();
  switch (dimension) {
    case xenos::DataDimension::k1D:
    case xenos::DataDimension::k2DOrStacked:
      // 1D and 2D are emulated as 2D arrays.
      return device_properties.maxImageArrayLayers;
    case xenos::DataDimension::k3D:
      return device_properties.maxImageDimension3D;
    case xenos::DataDimension::kCube:
      // Not requesting the imageCubeArray feature, and the Xenos doesn't
      // support cube map arrays.
      return 6;
    default:
      assert_unhandled_case(dimension);
      return 0;
  }
}

std::unique_ptr<TextureCache::Texture> VulkanTextureCache::CreateTexture(TextureKey key) {
  return CreateTextureResource(key, true);
}

std::unique_ptr<TextureCache::Texture> VulkanTextureCache::CreateTextureResource(
    TextureKey key, bool track_usage) {
  VkFormat formats[] = {VK_FORMAT_UNDEFINED, VK_FORMAT_UNDEFINED};
  const HostFormatPair& host_format = GetHostFormatPair(key);
  if (host_format.format_signed.format == VK_FORMAT_UNDEFINED) {
    // Only the unsigned format may be available, if at all.
    formats[0] = host_format.format_unsigned.format;
  } else if (host_format.format_unsigned.format == VK_FORMAT_UNDEFINED) {
    // Only the signed format may be available, if at all.
    formats[0] = host_format.format_signed.format;
  } else {
    // Both unsigned and signed formats are available.
    if (IsSignedVersionSeparateForFormat(key)) {
      formats[0] = key.signed_separate ? host_format.format_signed.format
                                       : host_format.format_unsigned.format;
    } else {
      // Same format for unsigned and signed, or compatible formats.
      formats[0] = host_format.format_unsigned.format;
      if (host_format.format_signed.format != host_format.format_unsigned.format) {
        assert_not_zero(host_format.unsigned_signed_compatible);
        formats[1] = host_format.format_signed.format;
      }
    }
  }
  if (formats[0] == VK_FORMAT_UNDEFINED) {
    unsupported_format_features_used_[uint32_t(key.format)] |= kUnsupportedResourceBit;
    return nullptr;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  bool is_3d = key.dimension == xenos::DataDimension::k3D;
  uint32_t depth_or_array_size = key.GetDepthOrArraySize();

  VkImageCreateInfo image_create_info;
  VkImageCreateInfo* image_create_info_last = &image_create_info;
  image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  image_create_info.pNext = nullptr;
  image_create_info.flags = 0;
  if (formats[1] != VK_FORMAT_UNDEFINED) {
    image_create_info.flags |= VK_IMAGE_CREATE_MUTABLE_FORMAT_BIT;
  }
  if (key.dimension == xenos::DataDimension::kCube) {
    image_create_info.flags |= VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
  }
  image_create_info.imageType = is_3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D;
  image_create_info.format = formats[0];
  image_create_info.extent.width = key.GetWidth();
  image_create_info.extent.height = key.GetHeight();
  if (key.scaled_resolve) {
    image_create_info.extent.width *= draw_resolution_scale_x();
    image_create_info.extent.height *= draw_resolution_scale_y();
  }
  image_create_info.extent.depth = is_3d ? depth_or_array_size : 1;
  image_create_info.mipLevels = key.mip_max_level + 1;
  image_create_info.arrayLayers = is_3d ? 1 : depth_or_array_size;
  image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
  image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
  image_create_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
  image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  image_create_info.queueFamilyIndexCount = 0;
  image_create_info.pQueueFamilyIndices = nullptr;
  image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
  VkImageFormatListCreateInfo image_format_list_create_info;
  if (formats[1] != VK_FORMAT_UNDEFINED &&
      vulkan_device->extensions().ext_1_2_KHR_image_format_list) {
    image_create_info_last->pNext = &image_format_list_create_info;
    image_create_info_last = reinterpret_cast<VkImageCreateInfo*>(&image_format_list_create_info);
    image_format_list_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_LIST_CREATE_INFO;
    image_format_list_create_info.pNext = nullptr;
    image_format_list_create_info.viewFormatCount = 2;
    image_format_list_create_info.pViewFormats = formats;
  }

  VmaAllocationCreateInfo allocation_create_info = {};
  allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

  VkImage image;
  VmaAllocation allocation;

  // Default-off direct storage-image path: add VK_IMAGE_USAGE_STORAGE_BIT only
  // for images on this path (the R8G8B8A8_UNORM scaled-resolve slice), and only
  // when the device accepts STORAGE usage for these exact image parameters.
  bool direct_storage_image = false;
  if (direct_storage_image_enabled_ && direct_storage_image_pipeline_ != VK_NULL_HANDLE) {
    const bool host_format_is_signed =
        IsSignedVersionSeparateForFormat(key) && key.signed_separate;
    const HostFormat& direct_host_format =
        host_format_is_signed ? host_format.format_signed : host_format.format_unsigned;
    DirectStorageImageCriteria criteria;
    criteria.enabled = true;
    criteria.scaled_resolve = key.scaled_resolve != 0;
    criteria.tiled = key.tiled != 0;
    criteria.signed_separate = key.signed_separate != 0;
    criteria.packed_mips = key.packed_mips != 0;
    criteria.is_2d = key.dimension == xenos::DataDimension::k2DOrStacked;
    criteria.guest_format = uint32_t(key.format);
    criteria.host_format = uint32_t(direct_host_format.format);
    criteria.load_shader = uint32_t(direct_host_format.load_shader);
    criteria.mip_max_level = key.mip_max_level;
    criteria.depth_or_array_size = depth_or_array_size;
    criteria.device_supported = true;
    if (EvaluateDirectStorageImage(criteria) == DirectStorageImageFallback::kNone &&
        (formats[0] == VK_FORMAT_R8G8B8A8_UNORM ||
         (formats[0] == VK_FORMAT_R32_SFLOAT && direct_storage_image_depth_pipeline_ != VK_NULL_HANDLE))) {
      image_create_info.usage |= VK_IMAGE_USAGE_STORAGE_BIT;
      if (REXCVAR_GET(gpu_direct_storage_image_parity_loads) > 0 ||
          REXCVAR_GET(gpu_fused_resolve_image_parity))
        image_create_info.usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
      if (QueryDirectStorageImageSupport(image_create_info)) {
        direct_storage_image = true;
      } else {
        image_create_info.usage &= ~VkImageUsageFlags(VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
      }
    }
  }

  if (vmaCreateImage(vma_allocator_, &image_create_info, &allocation_create_info, &image,
                     &allocation, nullptr)) {
    return nullptr;
  }

  return std::unique_ptr<Texture>(
      new VulkanTexture(*this, key, image, allocation, track_usage, direct_storage_image));
}

bool VulkanTextureCache::EnsureScaledResolveMemoryCommitted(uint32_t start_unscaled,
                                                            uint32_t length_unscaled,
                                                            uint32_t length_scaled_alignment_log2) {
  uint64_t start_scaled, length_scaled;
  if (!GetScaledResolveRange(start_unscaled, length_unscaled, length_scaled_alignment_log2,
                             start_scaled, length_scaled)) {
    return false;
  }
  return EnsureScaledResolveBufferAllocated(start_scaled, length_scaled);
}

void VulkanTextureCache::AuditResolveProducer(const draw_util::ResolveInfo& info) {
  if (!command_processor_.IsResolveChainAuditFrame()) return;
  const uint64_t frame = command_processor_.GetCurrentFrame();
  resolve_chain_audit_.ObserveFrame(frame);
  if (audit_resolve_frame_ != frame) {
    audit_resolve_frame_ = frame;
    audit_resolve_ordinal_ = 0;
  }
  const uint32_t start = info.copy_dest_extent_start & 0x1FFFFFFF;
  const uint32_t length = std::min(info.copy_dest_extent_length, 0x20000000u - start);
  const uint64_t producer = resolve_chain_audit_.Write(start, length);
  const auto source = info.IsCopyingDepth() ? info.depth_edram_info : info.color_edram_info;
  REXGPU_INFO("[CHAIN_PRODUCER] frame={} submission={} producer={} ordinal={} "
              "start={} length={} dest_base={} dest_format={} source_depth={} source_format={} "
              "source_base={} msaa_enum={} guest_width={} guest_height={} scale_x={} scale_y={}",
              frame, command_processor_.GetCurrentSubmission(), producer, ++audit_resolve_ordinal_,
              start, length, info.copy_dest_base,
              uint32_t(info.copy_dest_info.copy_dest_format), info.IsCopyingDepth(),
              uint32_t(source.format), info.IsCopyingDepth() ? info.depth_original_base : info.color_original_base,
              uint32_t(source.msaa_samples), uint32_t(info.coordinate_info.width_div_8) * 8,
              info.height_div_8 * 8, draw_resolution_scale_x(), draw_resolution_scale_y());
}

bool VulkanTextureCache::LoadTextureDataFromResidentMemoryImpl(Texture& texture, bool load_base,
                                                               bool load_mips) {
  VulkanTexture& vulkan_texture = static_cast<VulkanTexture&>(texture);
  DrainResolvedImageInvalidations();
  if (TryReuseFusedResolveImage(vulkan_texture, load_base, load_mips)) return true;
  if (!MakeFusedResolveImageWritable(vulkan_texture)) return false;
  TextureKey texture_key = vulkan_texture.key();

  // Get the pipeline.
  const HostFormatPair& host_format_pair = GetHostFormatPair(texture_key);
  bool host_format_is_signed;
  if (IsSignedVersionSeparateForFormat(texture_key)) {
    host_format_is_signed = bool(texture_key.signed_separate);
  } else {
    host_format_is_signed = host_format_pair.format_unsigned.load_shader == kLoadShaderIndexUnknown;
  }
  const HostFormat& host_format =
      host_format_is_signed ? host_format_pair.format_signed : host_format_pair.format_unsigned;
  LoadShaderIndex load_shader = host_format.load_shader;
  if (load_shader == kLoadShaderIndexUnknown) {
    return false;
  }
  VkPipeline pipeline = texture_key.scaled_resolve ? load_pipelines_scaled_[load_shader]
                                                   : load_pipelines_[load_shader];
  if (pipeline == VK_NULL_HANDLE) {
    return false;
  }
  const LoadShaderInfo& load_shader_info = GetLoadShaderInfo(load_shader);
  LoadShaderIndex load_shader_float_convert = kLoadShaderIndexUnknown;
  if (host_format.format == VK_FORMAT_R16G16B16A16_SFLOAT) {
    switch (load_shader) {
      case kLoadShaderIndexR10G11B11ToRGBA16:
      case kLoadShaderIndexR11G11B10ToRGBA16:
        load_shader_float_convert = kLoadShaderIndexRGBA16UNormToFloat;
        break;
      case kLoadShaderIndexR10G11B11ToRGBA16SNorm:
      case kLoadShaderIndexR11G11B10ToRGBA16SNorm:
        load_shader_float_convert = kLoadShaderIndexRGBA16SNormToFloat;
        break;
      default:
        break;
    }
  }
  const LoadShaderInfo* load_shader_info_float_convert = nullptr;
  VkPipeline pipeline_float_convert = VK_NULL_HANDLE;
  if (load_shader_float_convert != kLoadShaderIndexUnknown) {
    pipeline_float_convert = texture_key.scaled_resolve
                                 ? load_pipelines_scaled_[load_shader_float_convert]
                                 : load_pipelines_[load_shader_float_convert];
    if (pipeline_float_convert == VK_NULL_HANDLE) {
      return false;
    }
    load_shader_info_float_convert = &GetLoadShaderInfo(load_shader_float_convert);
  }

  // Get the guest layout.
  const texture_util::TextureGuestLayout& guest_layout = vulkan_texture.guest_layout();
  xenos::DataDimension dimension = texture_key.dimension;
  bool is_3d = dimension == xenos::DataDimension::k3D;
  bool is_3d_tiling = is_3d || vulkan_texture.force_load_3d_tiling();
  uint32_t width = texture_key.GetWidth();
  uint32_t height = texture_key.GetHeight();
  uint32_t depth_or_array_size = texture_key.GetDepthOrArraySize();
  uint32_t depth = is_3d ? depth_or_array_size : 1;
  uint32_t array_size = is_3d ? 1 : depth_or_array_size;
  xenos::TextureFormat guest_format = texture_key.format;
  const FormatInfo* guest_format_info = FormatInfo::Get(guest_format);
  uint32_t block_width = guest_format_info->block_width;
  uint32_t block_height = guest_format_info->block_height;
  uint32_t bytes_per_block = guest_format_info->bytes_per_block();
  uint32_t level_first = load_base ? 0 : 1;
  uint32_t level_last = load_mips ? texture_key.mip_max_level : 0;
  assert_true(level_first <= level_last);
  uint32_t level_packed = guest_layout.packed_level;
  uint32_t level_stored_first = std::min(level_first, level_packed);
  uint32_t level_stored_last = std::min(level_last, level_packed);
  uint32_t texture_resolution_scale_x = texture_key.scaled_resolve ? draw_resolution_scale_x() : 1;
  uint32_t texture_resolution_scale_y = texture_key.scaled_resolve ? draw_resolution_scale_y() : 1;

  // The loop counter can mean two things depending on whether the packed mip
  // tail is stored as mip 0, because in this case, it would be ambiguous since
  // both the base and the mips would be on "level 0", but stored in separate
  // places.
  uint32_t loop_level_first, loop_level_last;
  if (level_packed == 0) {
    // Packed mip tail is the level 0 - may need to load mip tails for the base,
    // the mips, or both.
    // Loop iteration 0 - base packed mip tail.
    // Loop iteration 1 - mips packed mip tail.
    loop_level_first = uint32_t(level_first != 0);
    loop_level_last = uint32_t(level_last != 0);
  } else {
    // Packed mip tail is not the level 0.
    // Loop iteration is the actual level being loaded.
    loop_level_first = level_stored_first;
    loop_level_last = level_stored_last;
  }

  // Get the host layout and the buffer.
  uint32_t host_block_width = host_format.block_compressed ? block_width : 1;
  uint32_t host_block_height = host_format.block_compressed ? block_height : 1;
  uint32_t host_x_blocks_per_thread = UINT32_C(1)
                                      << load_shader_info.guest_x_blocks_per_thread_log2;
  if (!host_format.block_compressed) {
    // Decompressing guest blocks.
    host_x_blocks_per_thread *= block_width;
  }
  VkDeviceSize host_buffer_size = 0;
  struct HostLayout {
    VkDeviceSize offset_bytes;
    VkDeviceSize slice_size_bytes;
    uint32_t x_pitch_blocks;
    uint32_t y_pitch_blocks;
  };
  HostLayout host_layout_base;
  // Indexing is the same as for guest stored mips:
  // 1...min(level_last, level_packed) if level_packed is not 0, or only 0 if
  // level_packed == 0.
  HostLayout host_layout_mips[xenos::kTextureMaxMips];
  for (uint32_t loop_level = loop_level_first; loop_level <= loop_level_last; ++loop_level) {
    bool is_base = loop_level == 0;
    uint32_t level = (level_packed == 0) ? 0 : loop_level;
    HostLayout& level_host_layout = is_base ? host_layout_base : host_layout_mips[level];
    level_host_layout.offset_bytes = host_buffer_size;
    uint32_t level_guest_x_extent_texels_unscaled;
    uint32_t level_guest_y_extent_texels_unscaled;
    uint32_t level_guest_z_extent_texels;
    if (level == level_packed) {
      // Loading the packed tail for the base or the mips - load the whole tail
      // to copy regions out of it.
      const texture_util::TextureGuestLayout::Level& guest_layout_packed =
          is_base ? guest_layout.base : guest_layout.mips[level];
      level_guest_x_extent_texels_unscaled = guest_layout_packed.x_extent_blocks * block_width;
      level_guest_y_extent_texels_unscaled = guest_layout_packed.y_extent_blocks * block_height;
      level_guest_z_extent_texels = guest_layout_packed.z_extent;
    } else {
      level_guest_x_extent_texels_unscaled = std::max(width >> level, UINT32_C(1));
      level_guest_y_extent_texels_unscaled = std::max(height >> level, UINT32_C(1));
      level_guest_z_extent_texels = std::max(depth >> level, UINT32_C(1));
    }
    level_host_layout.x_pitch_blocks =
        rex::round_up((level_guest_x_extent_texels_unscaled * texture_resolution_scale_x +
                       (host_block_width - 1)) /
                          host_block_width,
                      host_x_blocks_per_thread);
    level_host_layout.y_pitch_blocks =
        (level_guest_y_extent_texels_unscaled * texture_resolution_scale_y +
         (host_block_height - 1)) /
        host_block_height;
    level_host_layout.slice_size_bytes =
        VkDeviceSize(load_shader_info.bytes_per_host_block) * level_host_layout.x_pitch_blocks *
        level_host_layout.y_pitch_blocks * level_guest_z_extent_texels;
    host_buffer_size += level_host_layout.slice_size_bytes * array_size;
  }
  uint64_t legacy_copy_bytes = 0;
  for (uint32_t level = level_first; level <= level_last; ++level) {
    const HostLayout& level_host_layout =
        level != 0 ? host_layout_mips[std::min(level, level_packed)] : host_layout_base;
    const uint32_t level_width = std::max((width * texture_resolution_scale_x) >> level, UINT32_C(1));
    const uint32_t level_height =
        std::max((height * texture_resolution_scale_y) >> level, UINT32_C(1));
    const uint32_t level_depth = std::max(depth >> level, UINT32_C(1));
    legacy_copy_bytes +=
        uint64_t((level_width + host_block_width - 1) / host_block_width) *
        ((level_height + host_block_height - 1) / host_block_height) * level_depth * array_size *
        load_shader_info.bytes_per_host_block;
  }

  DirectStorageImageFallback direct_storage_image_fallback =
      DirectStorageImageFallback::kPathDisabled;
  bool use_direct_storage_image = false;
  if (direct_storage_image_enabled_ && !parity_reentry_ && direct_storage_image_pipeline_ != VK_NULL_HANDLE &&
      pipeline_float_convert == VK_NULL_HANDLE && !vulkan_texture.force_load_3d_tiling()) {
    DirectStorageImageCriteria criteria;
    criteria.enabled = true;
    criteria.scaled_resolve = texture_key.scaled_resolve != 0;
    criteria.tiled = texture_key.tiled != 0;
    criteria.signed_separate = texture_key.signed_separate != 0;
    criteria.packed_mips = texture_key.packed_mips != 0;
    criteria.is_2d = texture_key.dimension == xenos::DataDimension::k2DOrStacked;
    criteria.guest_format = uint32_t(texture_key.format);
    criteria.host_format = uint32_t(host_format.format);
    criteria.load_shader = uint32_t(load_shader);
    criteria.mip_max_level = texture_key.mip_max_level;
    criteria.depth_or_array_size = depth_or_array_size;
    criteria.device_supported = vulkan_texture.direct_storage_image();
    direct_storage_image_fallback = EvaluateDirectStorageImage(criteria);
    if (direct_storage_image_fallback == DirectStorageImageFallback::kNone) {
      if (level_first != 0 || level_last != 0) {
        direct_storage_image_fallback = DirectStorageImageFallback::kMipLevels;
      } else if (!vulkan_texture.direct_storage_image()) {
        direct_storage_image_fallback = DirectStorageImageFallback::kDeviceUnsupported;
      } else {
        use_direct_storage_image = true;
      }
    }
  }

  const size_t parity_index = use_direct_storage_image
      ? BeginParityReadback(texture, load_base, load_mips) : SIZE_MAX;

  VulkanCommandProcessor::ScratchBufferAcquisition scratch_buffer_acquisition(
      command_processor_.AcquireScratchGpuBuffer(
          use_direct_storage_image ? VkDeviceSize(0)
                                   : host_buffer_size,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT));
  VkBuffer scratch_buffer = scratch_buffer_acquisition.buffer();
  if (!use_direct_storage_image) {
    if (scratch_buffer == VK_NULL_HANDLE) {
      return false;
    }
  }

  // Begin loading.
  // TODO(Triang3l): Going from one descriptor to another on per-array-layer
  // or even per-8-depth-slices level to stay within maxStorageBufferRange.
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VulkanSharedMemory& vulkan_shared_memory = static_cast<VulkanSharedMemory&>(shared_memory());
  std::array<VkWriteDescriptorSet, 3> write_descriptor_sets;
  uint32_t write_descriptor_set_count = 0;
  VkDescriptorSet descriptor_set_dest = VK_NULL_HANDLE;
  VkDescriptorBufferInfo write_descriptor_set_dest_buffer_info;
  if (!use_direct_storage_image) {
    descriptor_set_dest = command_processor_.AllocateSingleTransientDescriptor(
        VulkanCommandProcessor::SingleTransientDescriptorLayout ::kStorageBufferCompute);
    if (!descriptor_set_dest) {
      return false;
    }
    write_descriptor_set_dest_buffer_info.buffer = scratch_buffer;
    write_descriptor_set_dest_buffer_info.offset = 0;
    write_descriptor_set_dest_buffer_info.range = host_buffer_size;
    VkWriteDescriptorSet& write_descriptor_set_dest =
        write_descriptor_sets[write_descriptor_set_count++];
    write_descriptor_set_dest.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set_dest.pNext = nullptr;
    write_descriptor_set_dest.dstSet = descriptor_set_dest;
    write_descriptor_set_dest.dstBinding = 0;
    write_descriptor_set_dest.dstArrayElement = 0;
    write_descriptor_set_dest.descriptorCount = 1;
    write_descriptor_set_dest.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set_dest.pImageInfo = nullptr;
    write_descriptor_set_dest.pBufferInfo = &write_descriptor_set_dest_buffer_info;
    write_descriptor_set_dest.pTexelBufferView = nullptr;
  }
  // TODO(Triang3l): Use a single 512 MB shared memory binding if possible.
  // TODO(Triang3l): Scaled resolve buffer bindings.
  // Aligning because if the data for a vector in a storage buffer is provided
  // partially, the value read may still be (0, 0, 0, 0), and small (especially
  // linear) textures won't be loaded correctly.
  uint32_t source_length_alignment = UINT32_C(1) << load_shader_info.source_bpe_log2;
  VkDescriptorSet descriptor_set_source_base = VK_NULL_HANDLE;
  VkDescriptorSet descriptor_set_source_mips = VK_NULL_HANDLE;
  VkDescriptorBufferInfo write_descriptor_set_source_base_buffer_info;
  VkDescriptorBufferInfo write_descriptor_set_source_mips_buffer_info;
  if (level_first == 0) {
    descriptor_set_source_base = command_processor_.AllocateSingleTransientDescriptor(
        VulkanCommandProcessor::SingleTransientDescriptorLayout ::kStorageBufferCompute);
    if (!descriptor_set_source_base) {
      return false;
    }
    uint32_t source_base_start_unscaled = texture_key.base_page << 12;
    uint64_t source_base_start;
    uint64_t source_base_range;
    if (texture_key.scaled_resolve) {
      if (!GetScaledResolveRange(source_base_start_unscaled, vulkan_texture.GetGuestBaseSize(),
                                 load_shader_info.source_bpe_log2, source_base_start,
                                 source_base_range)) {
        return false;
      }
      write_descriptor_set_source_base_buffer_info.buffer = scaled_resolve_buffer_;
    } else {
      source_base_start = source_base_start_unscaled;
      source_base_range = rex::align(uint64_t(vulkan_texture.GetGuestBaseSize()),
                                     uint64_t(source_length_alignment));
      if (source_base_range > uint64_t(SharedMemory::kBufferSize) - source_base_start) {
        return false;
      }
      write_descriptor_set_source_base_buffer_info.buffer = vulkan_shared_memory.buffer();
    }
    write_descriptor_set_source_base_buffer_info.offset = source_base_start;
    write_descriptor_set_source_base_buffer_info.range = source_base_range;
    VkWriteDescriptorSet& write_descriptor_set_source_base =
        write_descriptor_sets[write_descriptor_set_count++];
    write_descriptor_set_source_base.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set_source_base.pNext = nullptr;
    write_descriptor_set_source_base.dstSet = descriptor_set_source_base;
    write_descriptor_set_source_base.dstBinding = 0;
    write_descriptor_set_source_base.dstArrayElement = 0;
    write_descriptor_set_source_base.descriptorCount = 1;
    write_descriptor_set_source_base.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set_source_base.pImageInfo = nullptr;
    write_descriptor_set_source_base.pBufferInfo = &write_descriptor_set_source_base_buffer_info;
    write_descriptor_set_source_base.pTexelBufferView = nullptr;
  }
  if (level_last != 0) {
    descriptor_set_source_mips = command_processor_.AllocateSingleTransientDescriptor(
        VulkanCommandProcessor::SingleTransientDescriptorLayout ::kStorageBufferCompute);
    if (!descriptor_set_source_mips) {
      return false;
    }
    uint32_t source_mips_start_unscaled = texture_key.mip_page << 12;
    uint64_t source_mips_start;
    uint64_t source_mips_range;
    if (texture_key.scaled_resolve) {
      if (!GetScaledResolveRange(source_mips_start_unscaled, vulkan_texture.GetGuestMipsSize(),
                                 load_shader_info.source_bpe_log2, source_mips_start,
                                 source_mips_range)) {
        return false;
      }
      write_descriptor_set_source_mips_buffer_info.buffer = scaled_resolve_buffer_;
    } else {
      source_mips_start = source_mips_start_unscaled;
      source_mips_range = rex::align(uint64_t(vulkan_texture.GetGuestMipsSize()),
                                     uint64_t(source_length_alignment));
      if (source_mips_range > uint64_t(SharedMemory::kBufferSize) - source_mips_start) {
        return false;
      }
      write_descriptor_set_source_mips_buffer_info.buffer = vulkan_shared_memory.buffer();
    }
    write_descriptor_set_source_mips_buffer_info.offset = source_mips_start;
    write_descriptor_set_source_mips_buffer_info.range = source_mips_range;
    VkWriteDescriptorSet& write_descriptor_set_source_mips =
        write_descriptor_sets[write_descriptor_set_count++];
    write_descriptor_set_source_mips.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write_descriptor_set_source_mips.pNext = nullptr;
    write_descriptor_set_source_mips.dstSet = descriptor_set_source_mips;
    write_descriptor_set_source_mips.dstBinding = 0;
    write_descriptor_set_source_mips.dstArrayElement = 0;
    write_descriptor_set_source_mips.descriptorCount = 1;
    write_descriptor_set_source_mips.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write_descriptor_set_source_mips.pImageInfo = nullptr;
    write_descriptor_set_source_mips.pBufferInfo = &write_descriptor_set_source_mips_buffer_info;
    write_descriptor_set_source_mips.pTexelBufferView = nullptr;
  }
  if (write_descriptor_set_count) {
    dfn.vkUpdateDescriptorSets(device, write_descriptor_set_count, write_descriptor_sets.data(), 0,
                               nullptr);
  }
  if (texture_key.scaled_resolve) {
    UseScaledResolveBufferForRead();
  } else {
    vulkan_shared_memory.Use(VulkanSharedMemory::Usage::kRead);
  }

  // Submit the copy buffer population commands.

  DeferredCommandBuffer& command_buffer = command_processor_.deferred_command_buffer();

  const VkPipeline active_load_pipeline =
      use_direct_storage_image ? (host_format.format == VK_FORMAT_R32_SFLOAT
          ? direct_storage_image_depth_pipeline_ : direct_storage_image_pipeline_) : pipeline;
  const VkPipelineLayout active_load_pipeline_layout =
      use_direct_storage_image ? direct_storage_image_pipeline_layout_ : load_pipeline_layout_;

  if (use_direct_storage_image) {
    vulkan_texture.MarkAsUsed();
    VulkanTexture::Usage texture_old_usage =
        vulkan_texture.SetUsage(VulkanTexture::Usage::kComputeStorageWrite);
    if (texture_old_usage != VulkanTexture::Usage::kComputeStorageWrite) {
      VkPipelineStageFlags texture_src_stage_mask, texture_dst_stage_mask;
      VkAccessFlags texture_src_access_mask, texture_dst_access_mask;
      VkImageLayout texture_old_layout, texture_new_layout;
      GetTextureUsageMasks(texture_old_usage, texture_src_stage_mask, texture_src_access_mask,
                           texture_old_layout);
      GetTextureUsageMasks(VulkanTexture::Usage::kComputeStorageWrite, texture_dst_stage_mask,
                           texture_dst_access_mask, texture_new_layout);
      command_processor_.PushImageMemoryBarrier(
          vulkan_texture.image(), ui::vulkan::util::InitializeSubresourceRange(),
          texture_src_stage_mask, texture_dst_stage_mask, texture_src_access_mask,
          texture_dst_access_mask, texture_old_layout, texture_new_layout);
    }
    command_processor_.SubmitBarriers(true);
  }

  command_processor_.BindExternalComputePipeline(active_load_pipeline);

  if (use_direct_storage_image) {
    VkDescriptorSet descriptor_set_dest_image = vulkan_texture.direct_storage_image_descriptor_set();
    command_buffer.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, active_load_pipeline_layout,
                                           kLoadDescriptorSetIndexDestination, 1,
                                           &descriptor_set_dest_image, 0, nullptr);
  } else {
    command_buffer.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, active_load_pipeline_layout,
                                           kLoadDescriptorSetIndexDestination, 1, &descriptor_set_dest,
                                           0, nullptr);
  }

  VkDescriptorSet descriptor_set_source_current = VK_NULL_HANDLE;

  LoadConstants load_constants;
  // 3 bits for each.
  assert_true(texture_resolution_scale_x <= 7);
  assert_true(texture_resolution_scale_y <= 7);
  load_constants.is_tiled_3d_endian_scale =
      uint32_t(texture_key.tiled) | (uint32_t(is_3d_tiling) << 1) |
      (uint32_t(texture_key.endianness) << 2) | (texture_resolution_scale_x << 4) |
      (texture_resolution_scale_y << 7);

  uint32_t guest_x_blocks_per_group_log2 = load_shader_info.GetGuestXBlocksPerGroupLog2();
  for (uint32_t loop_level = loop_level_first; loop_level <= loop_level_last; ++loop_level) {
    bool is_base = loop_level == 0;
    uint32_t level = (level_packed == 0) ? 0 : loop_level;

    VkDescriptorSet descriptor_set_source =
        is_base ? descriptor_set_source_base : descriptor_set_source_mips;
    if (descriptor_set_source_current != descriptor_set_source) {
      descriptor_set_source_current = descriptor_set_source;
      command_buffer.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE,
                                             active_load_pipeline_layout,
                                             kLoadDescriptorSetIndexSource, 1, &descriptor_set_source,
                                             0, nullptr);
    }

    // TODO(Triang3l): guest_offset relative to the storage buffer origin.
    load_constants.guest_offset = 0;
    if (!is_base) {
      load_constants.guest_offset += guest_layout.mip_offsets_bytes[level] *
                                     (texture_resolution_scale_x * texture_resolution_scale_y);
    }
    const texture_util::TextureGuestLayout::Level& level_guest_layout =
        is_base ? guest_layout.base : guest_layout.mips[level];
    uint32_t level_guest_pitch = level_guest_layout.row_pitch_bytes;
    if (texture_key.tiled) {
      // Shaders expect pitch in blocks for tiled textures.
      level_guest_pitch /= bytes_per_block;
      assert_zero(level_guest_pitch & (xenos::kTextureTileWidthHeight - 1));
    }
    load_constants.guest_pitch_aligned = level_guest_pitch;
    load_constants.guest_z_stride_block_rows_aligned = level_guest_layout.z_slice_stride_block_rows;
    assert_true(!is_3d_tiling || !(load_constants.guest_z_stride_block_rows_aligned &
                                   (xenos::kTextureTileWidthHeight - 1)));

    uint32_t level_width, level_height, level_depth;
    if (level == level_packed) {
      // This is the packed mip tail, containing not only the specified level,
      // but also other levels at different offsets - load the entire needed
      // extents.
      level_width = level_guest_layout.x_extent_blocks * block_width;
      level_height = level_guest_layout.y_extent_blocks * block_height;
      level_depth = level_guest_layout.z_extent;
    } else {
      level_width = std::max(width >> level, UINT32_C(1));
      level_height = std::max(height >> level, UINT32_C(1));
      level_depth = std::max(depth >> level, UINT32_C(1));
    }
    load_constants.size_blocks[0] =
        (level_width + (block_width - 1)) / block_width * texture_resolution_scale_x;
    load_constants.size_blocks[1] =
        (level_height + (block_height - 1)) / block_height * texture_resolution_scale_y;
    load_constants.size_blocks[2] = level_depth;
    load_constants.height_texels = level_height;

    uint32_t group_count_x =
        (load_constants.size_blocks[0] + ((UINT32_C(1) << guest_x_blocks_per_group_log2) - 1)) >>
        guest_x_blocks_per_group_log2;
    uint32_t group_count_y =
        (load_constants.size_blocks[1] + ((UINT32_C(1) << kLoadGuestYBlocksPerGroupLog2) - 1)) >>
        kLoadGuestYBlocksPerGroupLog2;

    // TODO(Triang3l): host_offset relative to the storage buffer origin.
    const HostLayout& level_host_layout = is_base ? host_layout_base : host_layout_mips[level];
    load_constants.host_offset = uint32_t(level_host_layout.offset_bytes);
    load_constants.host_pitch =
        load_shader_info.bytes_per_host_block * level_host_layout.x_pitch_blocks;

    command_buffer.CmdVkPushConstants(active_load_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                      sizeof(load_constants), &load_constants);

    uint32_t level_array_slice_stride_bytes_scaled =
        level_guest_layout.array_slice_stride_bytes *
        (texture_resolution_scale_x * texture_resolution_scale_y);
    for (uint32_t slice = 0; slice < array_size; ++slice) {
      if (slice != 0) {
        command_buffer.CmdVkPushConstants(active_load_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                          offsetof(LoadConstants, guest_offset),
                                          sizeof(load_constants.guest_offset),
                                          &load_constants.guest_offset);
        command_buffer.CmdVkPushConstants(active_load_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                          offsetof(LoadConstants, host_offset),
                                          sizeof(load_constants.host_offset),
                                          &load_constants.host_offset);
      }
      command_processor_.SubmitBarriers(true);
      command_buffer.CmdVkDispatch(group_count_x, group_count_y, load_constants.size_blocks[2],
                                   texture_key.scaled_resolve ? WorkPurpose::kTextureScaledResolve
                                                              : WorkPurpose::kTextureGuest);
      load_constants.guest_offset += level_array_slice_stride_bytes_scaled;
      load_constants.host_offset += uint32_t(level_host_layout.slice_size_bytes);
    }
  }

  if (use_direct_storage_image) {
    const bool chain_audit = command_processor_.IsResolveChainAuditFrame();
    const uint64_t audit_frame = command_processor_.GetCurrentFrame();
    const uint64_t load_id = chain_audit ? ++audit_next_load_id_ : 0;
    if (chain_audit) {
      resolve_chain_audit_.ObserveFrame(audit_frame);
      if (!vulkan_texture.audit_texture_id_) {
        vulkan_texture.audit_texture_id_ = ++audit_next_texture_id_;
      }
      REXGPU_INFO(
          "[CHAIN_LOAD] frame={} submission={} load={} texture={} base_page={} mip_page={} "
          "guest_format={} vk_format={} signed_separate={} scaled={} tiled={} packed_mips={} "
          "pitch={} guest_width={} guest_height={} depth_or_layers={} load_base={} load_mips={} "
          "level_first={} level_last={} load_shader={} float_conversion={} "
          "direct_storage_image=1 optimal_storage_bit=1",
          audit_frame, command_processor_.GetCurrentSubmission(), load_id,
          vulkan_texture.audit_texture_id_, uint32_t(texture_key.base_page),
          uint32_t(texture_key.mip_page), uint32_t(texture_key.format), uint32_t(host_format.format),
          uint32_t(texture_key.signed_separate), uint32_t(texture_key.scaled_resolve),
          uint32_t(texture_key.tiled), uint32_t(texture_key.packed_mips), uint32_t(texture_key.pitch),
          width, height, depth_or_array_size, load_base, load_mips, level_first, level_last,
          uint32_t(load_shader), false);
      for (uint32_t part = 0; part < 2; ++part) {
        if (!(part ? load_mips : load_base)) {
          continue;
        }
        const uint32_t start = (part ? texture_key.mip_page : texture_key.base_page) << 12;
        const uint32_t length = part ? texture.GetGuestMipsSize() : texture.GetGuestBaseSize();
        const auto sample =
            resolve_chain_audit_.Load(vulkan_texture.audit_texture_id_, part != 0, start, length);
        const auto invalidations = texture.audit_invalidations(part != 0);
        const auto previous = vulkan_texture.audit_last_invalidations_[part];
        REXGPU_INFO(
            "[CHAIN_SOURCE] frame={} load={} texture={} mip={} start={} length={} valid={} "
            "previous_load={} same_observed_writers={} unknown_bytes={} cpu_watch_delta={} "
            "gpu_watch_delta={}",
            audit_frame, load_id, vulkan_texture.audit_texture_id_, part, start, length,
            sample.valid, sample.previous, sample.same_observed_writers, sample.unknown_bytes,
            invalidations.first - previous.first, invalidations.second - previous.second);
        vulkan_texture.audit_last_invalidations_[part] = invalidations;
        for (const auto& [producer, bytes] : sample.producer_bytes) {
          REXGPU_INFO(
              "[CHAIN_EDGE] frame={} load={} texture={} mip={} producer={} bytes={} "
              "provenance=resolve_extent",
              audit_frame, load_id, vulkan_texture.audit_texture_id_, part, producer, bytes);
        }
      }
    }
    if (direct_storage_image_enabled_) {
      CountDirectStorageImageResult(DirectStorageImageFallback::kNone, legacy_copy_bytes);
    }
    if (parity_index != SIZE_MAX)
      CaptureParityImage(vulkan_texture, parity_readbacks_[parity_index], true);
    return true;
  }

  if (pipeline_float_convert != VK_NULL_HANDLE) {
    assert_not_null(load_shader_info_float_convert);
    static std::array<uint8_t, 64> float_fallback_log_mask = {};
    uint8_t float_fallback_log_bit = host_format_is_signed ? uint8_t(1 << 1) : uint8_t(1 << 0);
    uint8_t& float_fallback_log_state = float_fallback_log_mask[uint32_t(texture_key.format)];
    if ((float_fallback_log_state & float_fallback_log_bit) == 0) {
      float_fallback_log_state |= float_fallback_log_bit;
      REXGPU_INFO("VulkanTextureCache: Using two-pass float fallback for format {} ({})",
                  FormatInfo::Get(texture_key.format)->name,
                  host_format_is_signed ? "signed" : "unsigned");
    }
    // Make the normalized intermediate data visible for conversion and keep it
    // writable for in-place conversion to float16.
    command_processor_.PushBufferMemoryBarrier(
        scratch_buffer, 0, VK_WHOLE_SIZE,
        scratch_buffer_acquisition.SetStageMask(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT),
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        scratch_buffer_acquisition.SetAccessMask(VK_ACCESS_SHADER_READ_BIT |
                                                 VK_ACCESS_SHADER_WRITE_BIT),
        VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);

    command_processor_.BindExternalComputePipeline(pipeline_float_convert);
    command_buffer.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, load_pipeline_layout_,
                                           kLoadDescriptorSetIndexDestination, 1,
                                           &descriptor_set_dest, 0, nullptr);
    command_buffer.CmdVkBindDescriptorSets(VK_PIPELINE_BIND_POINT_COMPUTE, load_pipeline_layout_,
                                           kLoadDescriptorSetIndexSource, 1, &descriptor_set_dest,
                                           0, nullptr);

    LoadConstants load_constants_float_convert;
    // Linear source and destination in the same scratch buffer, 1x1 scaling.
    load_constants_float_convert.is_tiled_3d_endian_scale =
        (uint32_t(is_3d) << 1) | (UINT32_C(1) << 4) | (UINT32_C(1) << 7);
    uint32_t guest_x_blocks_per_group_log2_float_convert =
        load_shader_info_float_convert->GetGuestXBlocksPerGroupLog2();

    for (uint32_t loop_level = loop_level_first; loop_level <= loop_level_last; ++loop_level) {
      bool is_base = loop_level == 0;
      uint32_t level = (level_packed == 0) ? 0 : loop_level;

      const texture_util::TextureGuestLayout::Level& level_guest_layout =
          is_base ? guest_layout.base : guest_layout.mips[level];
      uint32_t level_width, level_height, level_depth;
      if (level == level_packed) {
        level_width = level_guest_layout.x_extent_blocks * block_width;
        level_height = level_guest_layout.y_extent_blocks * block_height;
        level_depth = level_guest_layout.z_extent;
      } else {
        level_width = std::max(width >> level, UINT32_C(1));
        level_height = std::max(height >> level, UINT32_C(1));
        level_depth = std::max(depth >> level, UINT32_C(1));
      }
      load_constants_float_convert.size_blocks[0] =
          (level_width + (block_width - 1)) / block_width * texture_resolution_scale_x;
      load_constants_float_convert.size_blocks[1] =
          (level_height + (block_height - 1)) / block_height * texture_resolution_scale_y;
      load_constants_float_convert.size_blocks[2] = level_depth;
      load_constants_float_convert.height_texels = level_height;

      uint32_t group_count_x =
          (load_constants_float_convert.size_blocks[0] +
           ((UINT32_C(1) << guest_x_blocks_per_group_log2_float_convert) - 1)) >>
          guest_x_blocks_per_group_log2_float_convert;
      uint32_t group_count_y = (load_constants_float_convert.size_blocks[1] +
                                ((UINT32_C(1) << kLoadGuestYBlocksPerGroupLog2) - 1)) >>
                               kLoadGuestYBlocksPerGroupLog2;

      const HostLayout& level_host_layout = is_base ? host_layout_base : host_layout_mips[level];
      load_constants_float_convert.guest_offset = uint32_t(level_host_layout.offset_bytes);
      load_constants_float_convert.guest_pitch_aligned =
          load_shader_info_float_convert->bytes_per_host_block * level_host_layout.x_pitch_blocks;
      load_constants_float_convert.guest_z_stride_block_rows_aligned =
          level_host_layout.y_pitch_blocks;
      load_constants_float_convert.host_offset = uint32_t(level_host_layout.offset_bytes);
      load_constants_float_convert.host_pitch =
          load_shader_info_float_convert->bytes_per_host_block * level_host_layout.x_pitch_blocks;

      command_buffer.CmdVkPushConstants(load_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                                        sizeof(load_constants_float_convert),
                                        &load_constants_float_convert);

      uint32_t level_array_slice_stride_bytes = uint32_t(level_host_layout.slice_size_bytes);
      for (uint32_t slice = 0; slice < array_size; ++slice) {
        if (slice != 0) {
          command_buffer.CmdVkPushConstants(load_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                            offsetof(LoadConstants, guest_offset),
                                            sizeof(load_constants_float_convert.guest_offset),
                                            &load_constants_float_convert.guest_offset);
          command_buffer.CmdVkPushConstants(load_pipeline_layout_, VK_SHADER_STAGE_COMPUTE_BIT,
                                            offsetof(LoadConstants, host_offset),
                                            sizeof(load_constants_float_convert.host_offset),
                                            &load_constants_float_convert.host_offset);
        }
        command_processor_.SubmitBarriers(true);
        command_buffer.CmdVkDispatch(group_count_x, group_count_y,
                                     load_constants_float_convert.size_blocks[2],
                                     WorkPurpose::kTextureConvert);
        load_constants_float_convert.guest_offset += level_array_slice_stride_bytes;
        load_constants_float_convert.host_offset += level_array_slice_stride_bytes;
      }
    }
  }

  // Submit copying from the copy buffer to the host texture.
  command_processor_.PushBufferMemoryBarrier(
      scratch_buffer, 0, VK_WHOLE_SIZE,
      scratch_buffer_acquisition.SetStageMask(VK_PIPELINE_STAGE_TRANSFER_BIT),
      VK_PIPELINE_STAGE_TRANSFER_BIT,
      scratch_buffer_acquisition.SetAccessMask(VK_ACCESS_TRANSFER_READ_BIT),
      VK_ACCESS_TRANSFER_READ_BIT);
  vulkan_texture.MarkAsUsed();
  VulkanTexture::Usage texture_old_usage =
      vulkan_texture.SetUsage(VulkanTexture::Usage::kTransferDestination);
  if (texture_old_usage != VulkanTexture::Usage::kTransferDestination) {
    VkPipelineStageFlags texture_src_stage_mask, texture_dst_stage_mask;
    VkAccessFlags texture_src_access_mask, texture_dst_access_mask;
    VkImageLayout texture_old_layout, texture_new_layout;
    GetTextureUsageMasks(texture_old_usage, texture_src_stage_mask, texture_src_access_mask,
                         texture_old_layout);
    GetTextureUsageMasks(VulkanTexture::Usage::kTransferDestination, texture_dst_stage_mask,
                         texture_dst_access_mask, texture_new_layout);
    command_processor_.PushImageMemoryBarrier(
        vulkan_texture.image(), ui::vulkan::util::InitializeSubresourceRange(),
        texture_src_stage_mask, texture_dst_stage_mask, texture_src_access_mask,
        texture_dst_access_mask, texture_old_layout, texture_new_layout);
  }
  command_processor_.SubmitBarriers(true);
  VkBufferImageCopy* copy_regions = command_buffer.CmdCopyBufferToImageEmplace(
      scratch_buffer, vulkan_texture.image(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
      level_last - level_first + 1, host_block_width, host_block_height,
      load_shader_info.bytes_per_host_block);
  const bool chain_audit = command_processor_.IsResolveChainAuditFrame();
  const uint64_t audit_frame = command_processor_.GetCurrentFrame();
  const uint64_t load_id = chain_audit ? ++audit_next_load_id_ : 0;
  if (chain_audit) {
    resolve_chain_audit_.ObserveFrame(audit_frame);
    if (!vulkan_texture.audit_texture_id_) vulkan_texture.audit_texture_id_ = ++audit_next_texture_id_;
    VkFormatProperties audit_format_properties{};
    vulkan_device->vulkan_instance()->functions().vkGetPhysicalDeviceFormatProperties(
        vulkan_device->physical_device(), host_format.format, &audit_format_properties);
    REXGPU_INFO("[CHAIN_LOAD] frame={} submission={} load={} texture={} base_page={} mip_page={} "
                "guest_format={} vk_format={} signed_separate={} scaled={} tiled={} packed_mips={} "
                "pitch={} guest_width={} guest_height={} depth_or_layers={} load_base={} load_mips={} "
                "level_first={} level_last={} load_shader={} float_conversion={} optimal_storage_bit={}",
                audit_frame, command_processor_.GetCurrentSubmission(), load_id,
                vulkan_texture.audit_texture_id_, uint32_t(texture_key.base_page),
                uint32_t(texture_key.mip_page), uint32_t(texture_key.format), uint32_t(host_format.format),
                uint32_t(texture_key.signed_separate), uint32_t(texture_key.scaled_resolve),
                uint32_t(texture_key.tiled), uint32_t(texture_key.packed_mips), uint32_t(texture_key.pitch),
                width, height, depth_or_array_size, load_base, load_mips, level_first, level_last,
                uint32_t(load_shader), pipeline_float_convert != VK_NULL_HANDLE,
                (audit_format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_STORAGE_IMAGE_BIT) != 0);
    for (uint32_t part = 0; part < 2; ++part) {
      if (!(part ? load_mips : load_base)) continue;
      const uint32_t start = (part ? texture_key.mip_page : texture_key.base_page) << 12;
      const uint32_t length = part ? texture.GetGuestMipsSize() : texture.GetGuestBaseSize();
      const auto sample = resolve_chain_audit_.Load(vulkan_texture.audit_texture_id_, part != 0, start, length);
      const auto invalidations = texture.audit_invalidations(part != 0);
      const auto previous = vulkan_texture.audit_last_invalidations_[part];
      REXGPU_INFO("[CHAIN_SOURCE] frame={} load={} texture={} mip={} start={} length={} valid={} "
                  "previous_load={} same_observed_writers={} unknown_bytes={} cpu_watch_delta={} gpu_watch_delta={}",
                  audit_frame, load_id, vulkan_texture.audit_texture_id_, part, start, length,
                  sample.valid, sample.previous, sample.same_observed_writers, sample.unknown_bytes,
                  invalidations.first - previous.first, invalidations.second - previous.second);
      vulkan_texture.audit_last_invalidations_[part] = invalidations;
      for (const auto& [producer, bytes] : sample.producer_bytes)
        REXGPU_INFO("[CHAIN_EDGE] frame={} load={} texture={} mip={} producer={} bytes={} provenance=resolve_extent",
                    audit_frame, load_id, vulkan_texture.audit_texture_id_, part, producer, bytes);
    }
  }
  for (uint32_t level = level_first; level <= level_last; ++level) {
    VkBufferImageCopy& copy_region = copy_regions[level - level_first];
    const HostLayout& level_host_layout =
        level != 0 ? host_layout_mips[std::min(level, level_packed)] : host_layout_base;
    copy_region.bufferOffset = level_host_layout.offset_bytes;
    if (level >= level_packed) {
      uint32_t level_offset_blocks_x, level_offset_blocks_y, level_offset_z;
      texture_util::GetPackedMipOffset(width, height, depth, guest_format, level,
                                       level_offset_blocks_x, level_offset_blocks_y,
                                       level_offset_z);
      uint32_t level_offset_host_blocks_x = texture_resolution_scale_x * level_offset_blocks_x;
      uint32_t level_offset_host_blocks_y = texture_resolution_scale_y * level_offset_blocks_y;
      if (!host_format.block_compressed) {
        level_offset_host_blocks_x *= block_width;
        level_offset_host_blocks_y *= block_height;
      }
      copy_region.bufferOffset +=
          load_shader_info.bytes_per_host_block *
          (level_offset_host_blocks_x +
           level_host_layout.x_pitch_blocks *
               (level_offset_host_blocks_y +
                level_host_layout.y_pitch_blocks * VkDeviceSize(level_offset_z)));
    }
    copy_region.bufferRowLength = level_host_layout.x_pitch_blocks * host_block_width;
    copy_region.bufferImageHeight = level_host_layout.y_pitch_blocks * host_block_height;
    copy_region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copy_region.imageSubresource.mipLevel = level;
    copy_region.imageSubresource.baseArrayLayer = 0;
    copy_region.imageSubresource.layerCount = array_size;
    copy_region.imageOffset.x = 0;
    copy_region.imageOffset.y = 0;
    copy_region.imageOffset.z = 0;
    copy_region.imageExtent.width =
        std::max((width * texture_resolution_scale_x) >> level, UINT32_C(1));
    copy_region.imageExtent.height =
        std::max((height * texture_resolution_scale_y) >> level, UINT32_C(1));
    copy_region.imageExtent.depth = std::max(depth >> level, UINT32_C(1));
    if (chain_audit) {
      const uint64_t bytes = uint64_t((copy_region.imageExtent.width + host_block_width - 1) / host_block_width) *
          ((copy_region.imageExtent.height + host_block_height - 1) / host_block_height) *
          copy_region.imageExtent.depth * array_size * load_shader_info.bytes_per_host_block;
      REXGPU_INFO("[CHAIN_COPY] frame={} load={} texture={} level={} width={} height={} depth={} "
                  "layers={} vk_format={} bytes={} block_width={} block_height={} block_bytes={}",
                  audit_frame, load_id, vulkan_texture.audit_texture_id_, level,
                  copy_region.imageExtent.width, copy_region.imageExtent.height, copy_region.imageExtent.depth,
                  array_size, uint32_t(host_format.format), bytes, host_block_width, host_block_height,
                  load_shader_info.bytes_per_host_block);
    }
  }

  if (direct_storage_image_enabled_) {
    CountDirectStorageImageResult(direct_storage_image_fallback, legacy_copy_bytes);
  }
  return true;
}

void VulkanTextureCache::UpdateTextureBindingsImpl(uint32_t fetch_constant_mask) {
  uint32_t bindings_remaining = fetch_constant_mask;
  uint32_t binding_index;
  while (rex::bit_scan_forward(bindings_remaining, &binding_index)) {
    bindings_remaining &= ~(UINT32_C(1) << binding_index);
    VulkanTextureBinding& vulkan_binding = vulkan_texture_bindings_[binding_index];
    vulkan_binding.Reset();
    const TextureBinding* binding = GetValidTextureBinding(binding_index);
    if (!binding) {
      continue;
    }
    const HostFormatPair& host_format_pair = GetHostFormatPair(binding->key);
    bool uses_unsigned = texture_util::IsAnySignNotSigned(binding->swizzled_signs);
    bool uses_signed = texture_util::IsAnySignSigned(binding->swizzled_signs);
    xenos::TextureFormat format = binding->key.format;
    if (uses_unsigned && host_format_pair.format_unsigned.format == VK_FORMAT_UNDEFINED) {
      unsupported_format_features_used_[uint32_t(format)] |= kUnsupportedUnormBit;
    }
    if (uses_signed && host_format_pair.format_signed.format == VK_FORMAT_UNDEFINED) {
      unsupported_format_features_used_[uint32_t(format)] |= kUnsupportedSnormBit;
    }
    if (IsSignedVersionSeparateForFormat(binding->key)) {
      if (binding->texture && uses_unsigned &&
          host_format_pair.format_unsigned.format != VK_FORMAT_UNDEFINED) {
        vulkan_binding.image_view_unsigned =
            static_cast<VulkanTexture*>(binding->texture)->GetView(false, binding->host_swizzle);
      }
      if (binding->texture_signed && uses_signed &&
          host_format_pair.format_signed.format != VK_FORMAT_UNDEFINED) {
        vulkan_binding.image_view_signed = static_cast<VulkanTexture*>(binding->texture_signed)
                                               ->GetView(true, binding->host_swizzle);
      }
    } else {
      VulkanTexture* texture = static_cast<VulkanTexture*>(binding->texture);
      if (texture) {
        if (uses_unsigned && host_format_pair.format_unsigned.format != VK_FORMAT_UNDEFINED) {
          vulkan_binding.image_view_unsigned = texture->GetView(false, binding->host_swizzle);
        }
        if (uses_signed && host_format_pair.format_signed.format != VK_FORMAT_UNDEFINED) {
          vulkan_binding.image_view_signed = texture->GetView(true, binding->host_swizzle);
        }
        // Formats such as BC1 have no signed host representation, so the signed
        // view above is never created. Binding that slot would leave the shader
        // sampling the null image (all zeros), which reads as an unlit black
        // surface, so fall back to the unsigned view of the same resource.
        if (uses_signed && vulkan_binding.image_view_signed == VK_NULL_HANDLE &&
            vulkan_binding.image_view_unsigned != VK_NULL_HANDLE) {
          vulkan_binding.image_view_signed = vulkan_binding.image_view_unsigned;
        }
      }
    }
  }
}

VulkanTextureCache::VulkanTexture::VulkanTexture(VulkanTextureCache& texture_cache,
                                                 const TextureKey& key, VkImage image,
                                                 VmaAllocation allocation, bool track_usage,
                                                 bool direct_storage_image)
    : Texture(texture_cache, key, track_usage),
      image_resource_(std::make_shared<ImageResource>(texture_cache.vma_allocator_, image,
                                                       allocation)) {
  VmaAllocationInfo allocation_info;
  vmaGetAllocationInfo(texture_cache.vma_allocator_, allocation, &allocation_info);
  SetHostMemoryUsage(uint64_t(allocation_info.size));

  // Default-off direct storage-image path: a dedicated 2D view plus a stable
  // descriptor set, so the load shader can write the image without touching the
  // command processor's transient descriptor plumbing.
  if (direct_storage_image &&
      texture_cache.direct_storage_image_descriptor_set_layout_ != VK_NULL_HANDLE &&
      texture_cache.direct_storage_image_descriptor_pool_ != VK_NULL_HANDLE) {
    assert_zero(key.dimension != xenos::DataDimension::k2DOrStacked);
    assert_zero(key.GetDepthOrArraySize() != 1);
    const ui::vulkan::VulkanDevice* const vulkan_device =
        texture_cache.command_processor_.GetVulkanDevice();
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();
    VkImageViewCreateInfo view_create_info;
    view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view_create_info.pNext = nullptr;
    view_create_info.flags = 0;
    view_create_info.image = image;
    view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_create_info.format = key.format == xenos::TextureFormat::k_24_8
        ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
    view_create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    view_create_info.subresourceRange.baseMipLevel = 0;
    view_create_info.subresourceRange.levelCount = 1;
    view_create_info.subresourceRange.baseArrayLayer = 0;
    view_create_info.subresourceRange.layerCount = 1;
    if (dfn.vkCreateImageView(device, &view_create_info, nullptr,
                              &direct_storage_image_view_) == VK_SUCCESS) {
      VkDescriptorSetAllocateInfo descriptor_set_allocate_info;
      descriptor_set_allocate_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
      descriptor_set_allocate_info.pNext = nullptr;
      descriptor_set_allocate_info.descriptorPool =
          texture_cache.direct_storage_image_descriptor_pool_;
      descriptor_set_allocate_info.descriptorSetCount = 1;
      descriptor_set_allocate_info.pSetLayouts =
          &texture_cache.direct_storage_image_descriptor_set_layout_;
      if (dfn.vkAllocateDescriptorSets(device, &descriptor_set_allocate_info,
                                       &direct_storage_image_descriptor_set_) == VK_SUCCESS) {
        VkDescriptorImageInfo descriptor_image_info;
        descriptor_image_info.sampler = VK_NULL_HANDLE;
        descriptor_image_info.imageView = direct_storage_image_view_;
        descriptor_image_info.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        VkWriteDescriptorSet write_descriptor_set;
        write_descriptor_set.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write_descriptor_set.pNext = nullptr;
        write_descriptor_set.dstSet = direct_storage_image_descriptor_set_;
        write_descriptor_set.dstBinding = 0;
        write_descriptor_set.dstArrayElement = 0;
        write_descriptor_set.descriptorCount = 1;
        write_descriptor_set.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write_descriptor_set.pImageInfo = &descriptor_image_info;
        write_descriptor_set.pBufferInfo = nullptr;
        write_descriptor_set.pTexelBufferView = nullptr;
        dfn.vkUpdateDescriptorSets(device, 1, &write_descriptor_set, 0, nullptr);
      } else {
        dfn.vkDestroyImageView(device, direct_storage_image_view_, nullptr);
        direct_storage_image_view_ = VK_NULL_HANDLE;
      }
    }
  }
}

VulkanTextureCache::VulkanTexture::~VulkanTexture() {
  VulkanTextureCache& vulkan_texture_cache =
      static_cast<VulkanTextureCache&>(texture_cache());
  if (audit_texture_id_) vulkan_texture_cache.resolve_chain_audit_.ForgetTexture(audit_texture_id_);
  const ui::vulkan::VulkanDevice* const vulkan_device =
      vulkan_texture_cache.command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  for (const auto& view_pair : views_) {
    dfn.vkDestroyImageView(device, view_pair.second, nullptr);
  }
  if (image_view_3d_as_2d_unsigned_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, image_view_3d_as_2d_unsigned_, nullptr);
  }
  if (image_view_3d_as_2d_signed_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, image_view_3d_as_2d_signed_, nullptr);
  }
  if (direct_storage_image_descriptor_set_ != VK_NULL_HANDLE &&
      vulkan_texture_cache.vk_free_descriptor_sets_ != nullptr) {
    vulkan_texture_cache.vk_free_descriptor_sets_(
        device, vulkan_texture_cache.direct_storage_image_descriptor_pool_, 1,
        &direct_storage_image_descriptor_set_);
  }
  if (direct_storage_image_view_ != VK_NULL_HANDLE) {
    dfn.vkDestroyImageView(device, direct_storage_image_view_, nullptr);
  }
}

VkImageView VulkanTextureCache::VulkanTexture::GetView(bool is_signed, uint32_t host_swizzle,
                                                       bool is_array) {
  xenos::DataDimension dimension = key().dimension;
  if (dimension == xenos::DataDimension::k3D || dimension == xenos::DataDimension::kCube) {
    is_array = false;
  }

  const VulkanTextureCache& vulkan_texture_cache =
      static_cast<const VulkanTextureCache&>(texture_cache());

  ViewKey view_key;

  const HostFormatPair& host_format_pair = vulkan_texture_cache.GetHostFormatPair(key());
  VkFormat format =
      (is_signed ? host_format_pair.format_signed : host_format_pair.format_unsigned).format;
  if (format == VK_FORMAT_UNDEFINED) {
    return VK_NULL_HANDLE;
  }
  // If not distinguishing between unsigned and signed formats for the same
  // image, don't create two views. As this happens within an image, no need to
  // care about whether unsigned and signed images are separate - if they are
  // (or if there are only unsigned or only signed images), this image will have
  // either all views unsigned or all views signed.
  view_key.is_signed_separate_view = is_signed && (host_format_pair.format_signed.format !=
                                                   host_format_pair.format_unsigned.format);

  const ui::vulkan::VulkanDevice* const vulkan_device =
      vulkan_texture_cache.command_processor_.GetVulkanDevice();

  if (!vulkan_device->properties().imageViewFormatSwizzle) {
    host_swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
  }
  view_key.host_swizzle = host_swizzle;

  view_key.is_array = uint32_t(is_array);

  // Try to find an existing view.
  auto it = views_.find(view_key);
  if (it != views_.end()) {
    return it->second;
  }

  // Create a new view.
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VkImageViewCreateInfo view_create_info;
  view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_create_info.pNext = nullptr;
  view_create_info.flags = 0;
  view_create_info.image = image();
  view_create_info.format = format;
  view_create_info.components.r = GetComponentSwizzle(host_swizzle, 0);
  view_create_info.components.g = GetComponentSwizzle(host_swizzle, 1);
  view_create_info.components.b = GetComponentSwizzle(host_swizzle, 2);
  view_create_info.components.a = GetComponentSwizzle(host_swizzle, 3);
  view_create_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange();
  switch (dimension) {
    case xenos::DataDimension::k3D:
      view_create_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
      break;
    case xenos::DataDimension::kCube:
      view_create_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
      break;
    default:
      if (is_array) {
        view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
      } else {
        view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_create_info.subresourceRange.layerCount = 1;
      }
      break;
  }
  VkImageView view;
  if (dfn.vkCreateImageView(device, &view_create_info, nullptr, &view) != VK_SUCCESS) {
    REXGPU_ERROR(
        "VulkanTextureCache: Failed to create an image view for Vulkan format "
        "{} ({}signed) with swizzle 0x{:3X}",
        uint32_t(format), is_signed ? "" : "un", host_swizzle);
    return VK_NULL_HANDLE;
  }
  views_.emplace(view_key, view);
  return view;
}

VkImageView VulkanTextureCache::VulkanTexture::GetOrCreate3DAs2DImageView(bool is_signed,
                                                                          uint32_t host_swizzle) {
  if (!REXCVAR_GET(gpu_3d_to_2d_texture)) {
    return VK_NULL_HANDLE;
  }

  VkImageView& cached_view =
      is_signed ? image_view_3d_as_2d_signed_ : image_view_3d_as_2d_unsigned_;
  if (cached_view != VK_NULL_HANDLE) {
    return cached_view;
  }

  VulkanTextureCache& vulkan_texture_cache = static_cast<VulkanTextureCache&>(texture_cache());

  if (!texture_3d_as_2d_) {
    const HostFormatPair& host_format_pair = vulkan_texture_cache.GetHostFormatPair(key());
    VkFormat wrapper_format = host_format_pair.format_unsigned.format;
    if (wrapper_format == VK_FORMAT_UNDEFINED) {
      wrapper_format = host_format_pair.format_signed.format;
    }
    if (wrapper_format == VK_FORMAT_UNDEFINED) {
      return VK_NULL_HANDLE;
    }

    VkImageCreateInfo image_create_info = {};
    image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image_create_info.pNext = nullptr;
    image_create_info.flags = 0;
    image_create_info.imageType = VK_IMAGE_TYPE_2D;
    image_create_info.format = wrapper_format;
    image_create_info.extent.width = key().GetWidth();
    image_create_info.extent.height = key().GetHeight();
    image_create_info.extent.depth = 1;
    image_create_info.mipLevels = 1;
    image_create_info.arrayLayers = 1;
    image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_create_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    image_create_info.queueFamilyIndexCount = 0;
    image_create_info.pQueueFamilyIndices = nullptr;
    image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocation_create_info = {};
    allocation_create_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkImage image_2d;
    VmaAllocation allocation_2d;
    if (vmaCreateImage(vulkan_texture_cache.vma_allocator_, &image_create_info,
                       &allocation_create_info, &image_2d, &allocation_2d, nullptr) != VK_SUCCESS) {
      REXGPU_ERROR("VulkanTextureCache: Failed to create 3D-as-2D wrapper image");
      return VK_NULL_HANDLE;
    }

    TextureKey key_2d = key();
    key_2d.depth_or_array_size_minus_1 = 0;
    key_2d.mip_max_level = 0;
    texture_3d_as_2d_.reset(
        new VulkanTexture(vulkan_texture_cache, key_2d, image_2d, allocation_2d, false));
    texture_3d_as_2d_->SetForceLoad3DTiling(true);

    if (!vulkan_texture_cache.LoadTextureData(*texture_3d_as_2d_)) {
      REXGPU_ERROR("VulkanTextureCache: Failed to load 3D-as-2D wrapper data");
      texture_3d_as_2d_.reset();
      return VK_NULL_HANDLE;
    }

    if (!vulkan_texture_cache.command_processor_.PushImageMemoryBarrier(
            texture_3d_as_2d_->image(), ui::vulkan::util::InitializeSubresourceRange(),
            VK_PIPELINE_STAGE_TRANSFER_BIT, vulkan_texture_cache.guest_shader_pipeline_stages_,
            VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)) {
      texture_3d_as_2d_.reset();
      return VK_NULL_HANDLE;
    }
    texture_3d_as_2d_->SetUsage(Usage::kGuestShaderSampled);
  }

  const HostFormatPair& host_format_pair = vulkan_texture_cache.GetHostFormatPair(key());
  VkFormat view_format =
      (is_signed ? host_format_pair.format_signed : host_format_pair.format_unsigned).format;
  if (view_format == VK_FORMAT_UNDEFINED) {
    return VK_NULL_HANDLE;
  }

  const ui::vulkan::VulkanDevice* const vulkan_device =
      vulkan_texture_cache.command_processor_.GetVulkanDevice();
  if (!vulkan_device->properties().imageViewFormatSwizzle) {
    host_swizzle = xenos::XE_GPU_TEXTURE_SWIZZLE_RGBA;
  }

  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  VkImageViewCreateInfo view_create_info = {};
  view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_create_info.pNext = nullptr;
  view_create_info.flags = 0;
  view_create_info.image = texture_3d_as_2d_->image();
  view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
  view_create_info.format = view_format;
  view_create_info.components.r = GetComponentSwizzle(host_swizzle, 0);
  view_create_info.components.g = GetComponentSwizzle(host_swizzle, 1);
  view_create_info.components.b = GetComponentSwizzle(host_swizzle, 2);
  view_create_info.components.a = GetComponentSwizzle(host_swizzle, 3);
  view_create_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange();
  view_create_info.subresourceRange.layerCount = 1;

  if (dfn.vkCreateImageView(device, &view_create_info, nullptr, &cached_view) != VK_SUCCESS) {
    REXGPU_ERROR("VulkanTextureCache: Failed to create 3D-as-2D wrapper image view");
    return VK_NULL_HANDLE;
  }

  return cached_view;
}

VulkanTextureCache::VulkanTextureCache(const RegisterFile& register_file,
                                       VulkanSharedMemory& shared_memory,
                                       uint32_t draw_resolution_scale_x,
                                       uint32_t draw_resolution_scale_y,
                                       VulkanCommandProcessor& command_processor,
                                       VkPipelineStageFlags guest_shader_pipeline_stages)
    : TextureCache(register_file, shared_memory, draw_resolution_scale_x, draw_resolution_scale_y),
      command_processor_(command_processor),
      guest_shader_pipeline_stages_(guest_shader_pipeline_stages) {}

bool VulkanTextureCache::InitializeScaledResolveBuffer() {
  assert_true(IsDrawResolutionScaled());

  uint64_t scale_area = uint64_t(draw_resolution_scale_x()) * uint64_t(draw_resolution_scale_y());
  scaled_resolve_buffer_size_ = uint64_t(SharedMemory::kBufferSize) * scale_area;

  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  const VkBufferCreateFlags sparse_flags =
      VK_BUFFER_CREATE_SPARSE_BINDING_BIT | VK_BUFFER_CREATE_SPARSE_RESIDENCY_BIT;

  VkBufferCreateInfo buffer_create_info;
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.pNext = nullptr;
  buffer_create_info.flags = sparse_flags;
  buffer_create_info.size = scaled_resolve_buffer_size_;
  buffer_create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  buffer_create_info.queueFamilyIndexCount = 0;
  buffer_create_info.pQueueFamilyIndices = nullptr;

  if (vulkan_device->properties().sparseResidencyBuffer &&
      dfn.vkCreateBuffer(device, &buffer_create_info, nullptr, &scaled_resolve_buffer_) ==
          VK_SUCCESS) {
    VkMemoryRequirements memory_requirements;
    dfn.vkGetBufferMemoryRequirements(device, scaled_resolve_buffer_, &memory_requirements);
    if (rex::bit_scan_forward(
            memory_requirements.memoryTypeBits & vulkan_device->memory_types().device_local,
            &scaled_resolve_buffer_memory_type_)) {
      uint32_t alignment_log2;
      rex::bit_scan_forward(std::max(memory_requirements.alignment, VkDeviceSize(1)),
                            &alignment_log2);
      uint32_t size_log2 = rex::log2_ceil(scaled_resolve_buffer_size_);
      uint32_t preferred_log2 = std::max(uint32_t(0), size_log2 - uint32_t(10));
      scaled_resolve_sparse_granularity_log2_ =
          std::max(alignment_log2, std::max(uint32_t(22), preferred_log2));
      scaled_resolve_sparse_granularity_log2_ =
          std::min(scaled_resolve_sparse_granularity_log2_, size_log2);
      uint32_t allocation_count_log2 =
          std::max(uint32_t(6), size_log2 - scaled_resolve_sparse_granularity_log2_);
      scaled_resolve_sparse_allocated_.resize(size_t(1) << (allocation_count_log2 - uint32_t(6)));
      scaled_resolve_buffer_sparse_ = true;
    } else {
      dfn.vkDestroyBuffer(device, scaled_resolve_buffer_, nullptr);
      scaled_resolve_buffer_ = VK_NULL_HANDLE;
    }
  }

  if (scaled_resolve_buffer_ == VK_NULL_HANDLE) {
    buffer_create_info.flags = 0;
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
            vulkan_device, scaled_resolve_buffer_size_, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            ui::vulkan::util::MemoryPurpose::kDeviceLocal, scaled_resolve_buffer_,
            scaled_resolve_buffer_memory_.emplace_back(), &scaled_resolve_buffer_memory_type_)) {
      REXGPU_ERROR(
          "VulkanTextureCache: Failed to create a scaled resolve buffer with "
          "{} MB address space",
          scaled_resolve_buffer_size_ >> 20);
      return false;
    }
    scaled_resolve_buffer_sparse_ = false;
    scaled_resolve_sparse_granularity_log2_ = UINT32_MAX;
    scaled_resolve_sparse_allocated_.clear();
  }

  scaled_resolve_last_usage_write_ = false;
  scaled_resolve_last_written_range_ = std::make_pair(uint64_t(0), uint64_t(0));
  return true;
}

void VulkanTextureCache::ShutdownScaledResolveBuffer() {
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();

  ui::vulkan::util::DestroyAndNullHandle(dfn.vkDestroyBuffer, device, scaled_resolve_buffer_);
  for (VkDeviceMemory memory : scaled_resolve_buffer_memory_) {
    dfn.vkFreeMemory(device, memory, nullptr);
  }
  scaled_resolve_buffer_memory_.clear();
  scaled_resolve_sparse_allocated_.clear();
  scaled_resolve_buffer_size_ = 0;
  scaled_resolve_buffer_sparse_ = false;
  scaled_resolve_buffer_memory_type_ = UINT32_MAX;
  scaled_resolve_sparse_granularity_log2_ = UINT32_MAX;
  scaled_resolve_last_usage_write_ = false;
  scaled_resolve_last_written_range_ = std::make_pair(uint64_t(0), uint64_t(0));
}

bool VulkanTextureCache::GetScaledResolveRange(uint32_t start_unscaled, uint32_t length_unscaled,
                                               uint32_t length_scaled_alignment_log2,
                                               uint64_t& start_scaled_out,
                                               uint64_t& length_scaled_out) const {
  if (!IsDrawResolutionScaled()) {
    return false;
  }
  if (length_unscaled == 0) {
    start_scaled_out = 0;
    length_scaled_out = 0;
    return true;
  }
  if (start_unscaled > SharedMemory::kBufferSize ||
      (SharedMemory::kBufferSize - start_unscaled) < length_unscaled) {
    return false;
  }

  uint64_t scale_area = uint64_t(draw_resolution_scale_x()) * uint64_t(draw_resolution_scale_y());
  uint64_t start_scaled = uint64_t(start_unscaled) * scale_area;
  uint64_t length_scaled = uint64_t(length_unscaled) * scale_area;
  uint64_t alignment = uint64_t(1) << length_scaled_alignment_log2;
  length_scaled = rex::align(length_scaled, alignment);

  if (start_scaled > scaled_resolve_buffer_size_ ||
      length_scaled > scaled_resolve_buffer_size_ - start_scaled) {
    return false;
  }

  start_scaled_out = start_scaled;
  length_scaled_out = length_scaled;
  return true;
}

bool VulkanTextureCache::EnsureScaledResolveBufferAllocated(uint64_t start_scaled,
                                                            uint64_t length_scaled) {
  if (!length_scaled || !scaled_resolve_buffer_sparse_) {
    return true;
  }
  uint32_t allocation_first = uint32_t(start_scaled >> scaled_resolve_sparse_granularity_log2_);
  uint32_t allocation_last =
      uint32_t((start_scaled + (length_scaled - 1)) >> scaled_resolve_sparse_granularity_log2_);
  while (true) {
    std::pair<size_t, size_t> allocation_range =
        rex::bit::GetNextRangeUnset(scaled_resolve_sparse_allocated_.data(), allocation_first,
                                    allocation_last - allocation_first + 1);
    if (!allocation_range.second) {
      break;
    }

    const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
    const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
    const VkDevice device = vulkan_device->device();

    VkMemoryAllocateInfo memory_allocate_info;
    memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    memory_allocate_info.pNext = nullptr;
    memory_allocate_info.allocationSize = VkDeviceSize(allocation_range.second)
                                          << scaled_resolve_sparse_granularity_log2_;
    memory_allocate_info.memoryTypeIndex = scaled_resolve_buffer_memory_type_;
    VkDeviceMemory allocation_memory;
    if (dfn.vkAllocateMemory(device, &memory_allocate_info, nullptr, &allocation_memory) !=
        VK_SUCCESS) {
      REXGPU_ERROR(
          "VulkanTextureCache: Failed to allocate sparse memory for the "
          "scaled resolve buffer");
      return false;
    }
    scaled_resolve_buffer_memory_.push_back(allocation_memory);

    VkSparseMemoryBind bind;
    bind.resourceOffset = VkDeviceSize(allocation_range.first)
                          << scaled_resolve_sparse_granularity_log2_;
    bind.size = memory_allocate_info.allocationSize;
    bind.memory = allocation_memory;
    bind.memoryOffset = 0;
    bind.flags = 0;
    command_processor_.SparseBindBuffer(
        scaled_resolve_buffer_, 1, &bind,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT);

    rex::bit::SetRange(scaled_resolve_sparse_allocated_.data(), allocation_range.first,
                       allocation_range.second);
    allocation_first = uint32_t(allocation_range.first + allocation_range.second);
  }
  return true;
}

void VulkanTextureCache::GetScaledResolveUsageMasks(VkPipelineStageFlags& stage_mask_out,
                                                    VkAccessFlags& access_mask_out,
                                                    bool write) const {
  stage_mask_out = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
  access_mask_out = VK_ACCESS_SHADER_READ_BIT;
  if (write) {
    access_mask_out |= VK_ACCESS_SHADER_WRITE_BIT;
  }
}

void VulkanTextureCache::UseScaledResolveBufferForRead() {
  if (!scaled_resolve_buffer_) {
    return;
  }
  if (!scaled_resolve_last_usage_write_ && !scaled_resolve_last_written_range_.second) {
    return;
  }

  VkPipelineStageFlags src_stage_mask, dst_stage_mask;
  VkAccessFlags src_access_mask, dst_access_mask;
  GetScaledResolveUsageMasks(src_stage_mask, src_access_mask, scaled_resolve_last_usage_write_);
  GetScaledResolveUsageMasks(dst_stage_mask, dst_access_mask, false);
  VkDeviceSize offset;
  VkDeviceSize size;
  if (!scaled_resolve_last_usage_write_) {
    offset = VkDeviceSize(scaled_resolve_last_written_range_.first);
    size = VkDeviceSize(scaled_resolve_last_written_range_.second);
  } else {
    offset = 0;
    size = VK_WHOLE_SIZE;
    scaled_resolve_last_usage_write_ = false;
  }
  command_processor_.PushBufferMemoryBarrier(
      scaled_resolve_buffer_, offset, size, src_stage_mask, dst_stage_mask, src_access_mask,
      dst_access_mask, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
  scaled_resolve_last_written_range_ = std::make_pair(uint64_t(0), uint64_t(0));
}

void VulkanTextureCache::UseScaledResolveBufferForWrite(uint64_t written_start_scaled,
                                                        uint64_t written_length_scaled) {
  if (!scaled_resolve_buffer_) {
    return;
  }
  if (written_start_scaled > scaled_resolve_buffer_size_) {
    written_start_scaled = scaled_resolve_buffer_size_;
  }
  written_length_scaled =
      std::min(written_length_scaled, scaled_resolve_buffer_size_ - written_start_scaled);

  if (scaled_resolve_last_usage_write_ && scaled_resolve_last_written_range_.second) {
    VkPipelineStageFlags stage_mask;
    VkAccessFlags access_mask;
    GetScaledResolveUsageMasks(stage_mask, access_mask, true);
    command_processor_.PushBufferMemoryBarrier(
        scaled_resolve_buffer_, VkDeviceSize(scaled_resolve_last_written_range_.first),
        VkDeviceSize(scaled_resolve_last_written_range_.second), stage_mask, stage_mask,
        access_mask, access_mask, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
  } else if (!scaled_resolve_last_usage_write_) {
    VkPipelineStageFlags src_stage_mask, dst_stage_mask;
    VkAccessFlags src_access_mask, dst_access_mask;
    GetScaledResolveUsageMasks(src_stage_mask, src_access_mask, false);
    GetScaledResolveUsageMasks(dst_stage_mask, dst_access_mask, true);
    command_processor_.PushBufferMemoryBarrier(
        scaled_resolve_buffer_, 0, VK_WHOLE_SIZE, src_stage_mask, dst_stage_mask, src_access_mask,
        dst_access_mask, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
    scaled_resolve_last_usage_write_ = true;
  }

  scaled_resolve_last_written_range_ = std::make_pair(written_start_scaled, written_length_scaled);
}

bool VulkanTextureCache::Initialize() {
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanInstance::Functions& ifn = vulkan_device->vulkan_instance()->functions();
  const VkPhysicalDevice physical_device = vulkan_device->physical_device();
  const ui::vulkan::VulkanDevice::Functions& dfn = vulkan_device->functions();
  const VkDevice device = vulkan_device->device();
  const ui::vulkan::VulkanDevice::Properties& device_properties = vulkan_device->properties();

  // Vulkan Memory Allocator.

  vma_allocator_ = ui::vulkan::CreateVmaAllocator(vulkan_device, true);
  if (vma_allocator_ == VK_NULL_HANDLE) {
    return false;
  }

  if (IsDrawResolutionScaled() && !InitializeScaledResolveBuffer()) {
    return false;
  }

  // Image formats.

  // Initialize to the best formats.
  for (size_t i = 0; i < rex::countof(host_formats_); ++i) {
    host_formats_[i] = kBestHostFormats[i];
  }

  // Check format support and switch to fallbacks if needed.
  constexpr VkFormatFeatureFlags kLinearFilterFeatures =
      VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT;
  VkFormatProperties r16_unorm_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16_UNORM,
                                          &r16_unorm_properties);
  VkFormatProperties r16_snorm_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16_SNORM,
                                          &r16_snorm_properties);
  VkFormatProperties r16g16_unorm_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16G16_UNORM,
                                          &r16g16_unorm_properties);
  VkFormatProperties r16g16_snorm_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16G16_SNORM,
                                          &r16g16_snorm_properties);
  VkFormatProperties r16g16b16a16_unorm_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16G16B16A16_UNORM,
                                          &r16g16b16a16_unorm_properties);
  VkFormatProperties r16g16b16a16_snorm_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16G16B16A16_SNORM,
                                          &r16g16b16a16_snorm_properties);
  VkFormatProperties r16g16b16a16_sfloat_properties;
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R16G16B16A16_SFLOAT,
                                          &r16g16b16a16_sfloat_properties);
  VkFormatProperties format_properties;
  // k_Cr_Y1_Cb_Y0_REP, k_Y1_Cr_Y0_Cb_REP.
  HostFormatPair& host_format_gbgr =
      host_formats_[uint32_t(xenos::TextureFormat::k_Cr_Y1_Cb_Y0_REP)];
  assert_true(host_format_gbgr.format_unsigned.format == VK_FORMAT_G8B8G8R8_422_UNORM_KHR);
  assert_true(host_format_gbgr.format_signed.format == VK_FORMAT_UNDEFINED);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_G8B8G8R8_422_UNORM_KHR,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_gbgr.format_unsigned.load_shader = kLoadShaderIndexGBGR8ToRGB8;
    host_format_gbgr.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_gbgr.format_unsigned.block_compressed = false;
    host_format_gbgr.unsigned_signed_compatible = false;
  }
  HostFormatPair& host_format_bgrg =
      host_formats_[uint32_t(xenos::TextureFormat::k_Y1_Cr_Y0_Cb_REP)];
  assert_true(host_format_bgrg.format_unsigned.format == VK_FORMAT_B8G8R8G8_422_UNORM_KHR);
  assert_true(host_format_bgrg.format_signed.format == VK_FORMAT_UNDEFINED);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_B8G8R8G8_422_UNORM_KHR,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_bgrg.format_unsigned.load_shader = kLoadShaderIndexBGRG8ToRGB8;
    host_format_bgrg.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_bgrg.format_unsigned.block_compressed = false;
    host_format_bgrg.unsigned_signed_compatible = false;
  }
  // Expand R5G6B5 while loading if the native packed format can't be sampled.
  HostFormatPair& host_format_r5g6b5 = host_formats_[uint32_t(xenos::TextureFormat::k_5_6_5)];
  assert_true(host_format_r5g6b5.format_unsigned.format == VK_FORMAT_R5G6B5_UNORM_PACK16);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R5G6B5_UNORM_PACK16,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_r5g6b5.format_unsigned.load_shader = kLoadShaderIndexR5G6B5ToRGBA8;
    host_format_r5g6b5.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_r5g6b5.format_unsigned.block_compressed = false;
    host_format_r5g6b5.unsigned_signed_compatible = false;
  }
  // MoltenVK doesn't expose B4G4R4A4 sampling on all Metal devices. Expand
  // guest RGBA4444 texels to the universally supported RGBA8 format while
  // loading them instead of dropping the texture binding entirely.
  HostFormatPair& host_format_rgba4 = host_formats_[uint32_t(xenos::TextureFormat::k_4_4_4_4)];
  assert_true(host_format_rgba4.format_unsigned.format == VK_FORMAT_B4G4R4A4_UNORM_PACK16);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_B4G4R4A4_UNORM_PACK16,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_rgba4.format_unsigned.load_shader = kLoadShaderIndexRGBA4ToRGBA8;
    host_format_rgba4.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_rgba4.format_unsigned.block_compressed = false;
    host_format_rgba4.unsigned_signed_compatible = false;
  }
  // k_10_11_11, k_11_11_10 (+ _AS_16_16_16_16 variants) - if normalized
  // R16G16B16A16 is not linearly filterable, decode to normalized 16-bit in a
  // scratch buffer first, then convert it to float16 in-place in a second pass
  // before copying to the image.
  bool rgba16_norm_filterable =
      (r16g16b16a16_unorm_properties.optimalTilingFeatures & kLinearFilterFeatures) ==
          kLinearFilterFeatures &&
      (r16g16b16a16_snorm_properties.optimalTilingFeatures & kLinearFilterFeatures) ==
          kLinearFilterFeatures;
  bool rgba16_float_filterable = (r16g16b16a16_sfloat_properties.optimalTilingFeatures &
                                  kLinearFilterFeatures) == kLinearFilterFeatures;
  if (!rgba16_norm_filterable) {
    if (!rgba16_float_filterable) {
      REXGPU_ERROR(
          "VulkanTextureCache: Neither normalized nor float16 RGBA16 formats are linearly "
          "filterable for 10_11_11/11_11_10 texture fallback");
      return false;
    }
    REXGPU_WARN("VulkanTextureCache: Using RGBA16 float fallback for 10_11_11 / 11_11_10 textures");
    constexpr std::array<xenos::TextureFormat, 4> kR10G11B11Formats = {
        xenos::TextureFormat::k_10_11_11,
        xenos::TextureFormat::k_11_11_10,
        xenos::TextureFormat::k_10_11_11_AS_16_16_16_16,
        xenos::TextureFormat::k_11_11_10_AS_16_16_16_16,
    };
    for (xenos::TextureFormat format : kR10G11B11Formats) {
      HostFormatPair& host_format = host_formats_[uint32_t(format)];
      assert_true(host_format.format_unsigned.format == VK_FORMAT_R16G16B16A16_UNORM);
      assert_true(host_format.format_signed.format == VK_FORMAT_R16G16B16A16_SNORM);
      host_format.format_unsigned.format = VK_FORMAT_R16G16B16A16_SFLOAT;
      host_format.format_signed.format = VK_FORMAT_R16G16B16A16_SFLOAT;
      // Different source signednesses still need separate decoded data.
      host_format.unsigned_signed_compatible = false;
    }
  }
  // S3TC.
  // Not checking the textureCompressionBC feature because its availability
  // means that all BC formats are supported, however, the device may expose
  // some BC formats without this feature. Xenia doesn't use BC6H and BC7 at
  // all, and has fallbacks for each used format.
  // TODO(Triang3l): Raise the host texture memory usage limit if S3TC has to be
  // decompressed.
  // TODO(Triang3l): S3TC -> 5551 or 4444 as an option.
  // TODO(Triang3l): S3TC -> ETC2 / EAC (a huge research topic).
  HostFormatPair& host_format_dxt1 = host_formats_[uint32_t(xenos::TextureFormat::k_DXT1)];
  assert_true(host_format_dxt1.format_unsigned.format == VK_FORMAT_BC1_RGBA_UNORM_BLOCK);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_BC1_RGBA_UNORM_BLOCK,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_dxt1.format_unsigned.load_shader = kLoadShaderIndexDXT1ToRGBA8;
    host_format_dxt1.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_dxt1.format_unsigned.block_compressed = false;
    host_formats_[uint32_t(xenos::TextureFormat::k_DXT1_AS_16_16_16_16)] = host_format_dxt1;
  }
  HostFormatPair& host_format_dxt2_3 = host_formats_[uint32_t(xenos::TextureFormat::k_DXT2_3)];
  assert_true(host_format_dxt2_3.format_unsigned.format == VK_FORMAT_BC2_UNORM_BLOCK);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_BC2_UNORM_BLOCK,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_dxt2_3.format_unsigned.load_shader = kLoadShaderIndexDXT3ToRGBA8;
    host_format_dxt2_3.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_dxt2_3.format_unsigned.block_compressed = false;
    host_formats_[uint32_t(xenos::TextureFormat::k_DXT2_3_AS_16_16_16_16)] = host_format_dxt2_3;
  }
  HostFormatPair& host_format_dxt4_5 = host_formats_[uint32_t(xenos::TextureFormat::k_DXT4_5)];
  assert_true(host_format_dxt4_5.format_unsigned.format == VK_FORMAT_BC3_UNORM_BLOCK);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_BC3_UNORM_BLOCK,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_dxt4_5.format_unsigned.load_shader = kLoadShaderIndexDXT5ToRGBA8;
    host_format_dxt4_5.format_unsigned.format = VK_FORMAT_R8G8B8A8_UNORM;
    host_format_dxt4_5.format_unsigned.block_compressed = false;
    host_formats_[uint32_t(xenos::TextureFormat::k_DXT4_5_AS_16_16_16_16)] = host_format_dxt4_5;
  }
  HostFormatPair& host_format_dxn = host_formats_[uint32_t(xenos::TextureFormat::k_DXN)];
  assert_true(host_format_dxn.format_unsigned.format == VK_FORMAT_BC5_UNORM_BLOCK);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_BC5_UNORM_BLOCK,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_dxn.format_unsigned.load_shader = kLoadShaderIndexDXNToRG8;
    host_format_dxn.format_unsigned.format = VK_FORMAT_R8G8_UNORM;
    host_format_dxn.format_unsigned.block_compressed = false;
  }
  HostFormatPair& host_format_dxt5a = host_formats_[uint32_t(xenos::TextureFormat::k_DXT5A)];
  assert_true(host_format_dxt5a.format_unsigned.format == VK_FORMAT_BC4_UNORM_BLOCK);
  ifn.vkGetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_BC4_UNORM_BLOCK,
                                          &format_properties);
  if ((format_properties.optimalTilingFeatures & kLinearFilterFeatures) != kLinearFilterFeatures) {
    host_format_dxt5a.format_unsigned.load_shader = kLoadShaderIndexDXT5AToR8;
    host_format_dxt5a.format_unsigned.format = VK_FORMAT_R8_UNORM;
    host_format_dxt5a.format_unsigned.block_compressed = false;
  }
  // Keep signed format support aligned with D3D12 for texture formats where
  // D3D12 doesn't expose signed sampling paths.
  constexpr std::array<xenos::TextureFormat, 19> kD3D12SignedUnsupportedFormats = {
      xenos::TextureFormat::k_1_5_5_5,
      xenos::TextureFormat::k_5_6_5,
      xenos::TextureFormat::k_6_5_5,
      xenos::TextureFormat::k_2_10_10_10,
      xenos::TextureFormat::k_Cr_Y1_Cb_Y0_REP,
      xenos::TextureFormat::k_Y1_Cr_Y0_Cb_REP,
      xenos::TextureFormat::k_4_4_4_4,
      xenos::TextureFormat::k_DXT1,
      xenos::TextureFormat::k_DXT2_3,
      xenos::TextureFormat::k_DXT4_5,
      xenos::TextureFormat::k_DXN,
      xenos::TextureFormat::k_DXT5A,
      xenos::TextureFormat::k_CTX1,
      xenos::TextureFormat::k_DXT1_AS_16_16_16_16,
      xenos::TextureFormat::k_DXT2_3_AS_16_16_16_16,
      xenos::TextureFormat::k_DXT4_5_AS_16_16_16_16,
      xenos::TextureFormat::k_2_10_10_10_AS_16_16_16_16,
      xenos::TextureFormat::k_DXT3A,
      xenos::TextureFormat::k_DXT3A_AS_1_1_1_1,
  };
  for (xenos::TextureFormat format : kD3D12SignedUnsupportedFormats) {
    HostFormatPair& host_format = host_formats_[uint32_t(format)];
    host_format.format_signed.load_shader = kLoadShaderIndexUnknown;
    host_format.format_signed.format = VK_FORMAT_UNDEFINED;
    host_format.format_signed.linear_filterable = false;
    host_format.unsigned_signed_compatible = false;
  }
  // k_16, k_16_16, k_16_16_16_16 - UNORM / SNORM are optional, fall back to
  // SFLOAT, which is mandatory and is always filterable.
  // For D3D12 parity, prefer keeping normalized-format sampling filterable over
  // keeping fixed-point precision if the host lacks linear filtering support on
  // UNORM/SNORM variants.
  VkFormatFeatureFlags norm16_required_features = kLinearFilterFeatures;
  HostFormatPair& host_format_16 = host_formats_[uint32_t(xenos::TextureFormat::k_16)];
  assert_true(host_format_16.format_unsigned.format == VK_FORMAT_R16_UNORM);
  if ((r16_unorm_properties.optimalTilingFeatures & norm16_required_features) !=
      norm16_required_features) {
    host_format_16.format_unsigned.load_shader = kLoadShaderIndexR16UNormToFloat;
    host_format_16.format_unsigned.format = VK_FORMAT_R16_SFLOAT;
  }
  assert_true(host_format_16.format_signed.format == VK_FORMAT_R16_SNORM);
  if ((r16_snorm_properties.optimalTilingFeatures & norm16_required_features) !=
      norm16_required_features) {
    host_format_16.format_signed.load_shader = kLoadShaderIndexR16SNormToFloat;
    host_format_16.format_signed.format = VK_FORMAT_R16_SFLOAT;
  }
  host_format_16.unsigned_signed_compatible =
      (host_format_16.format_unsigned.format == VK_FORMAT_R16_UNORM &&
       host_format_16.format_signed.format == VK_FORMAT_R16_SNORM) ||
      (host_format_16.format_unsigned.format == VK_FORMAT_R16_SFLOAT &&
       host_format_16.format_signed.format == VK_FORMAT_R16_SFLOAT);
  HostFormatPair& host_format_16_16 = host_formats_[uint32_t(xenos::TextureFormat::k_16_16)];
  assert_true(host_format_16_16.format_unsigned.format == VK_FORMAT_R16G16_UNORM);
  if ((r16g16_unorm_properties.optimalTilingFeatures & norm16_required_features) !=
      norm16_required_features) {
    host_format_16_16.format_unsigned.load_shader = kLoadShaderIndexRG16UNormToFloat;
    host_format_16_16.format_unsigned.format = VK_FORMAT_R16G16_SFLOAT;
  }
  assert_true(host_format_16_16.format_signed.format == VK_FORMAT_R16G16_SNORM);
  if ((r16g16_snorm_properties.optimalTilingFeatures & norm16_required_features) !=
      norm16_required_features) {
    host_format_16_16.format_signed.load_shader = kLoadShaderIndexRG16SNormToFloat;
    host_format_16_16.format_signed.format = VK_FORMAT_R16G16_SFLOAT;
  }
  host_format_16_16.unsigned_signed_compatible =
      (host_format_16_16.format_unsigned.format == VK_FORMAT_R16G16_UNORM &&
       host_format_16_16.format_signed.format == VK_FORMAT_R16G16_SNORM) ||
      (host_format_16_16.format_unsigned.format == VK_FORMAT_R16G16_SFLOAT &&
       host_format_16_16.format_signed.format == VK_FORMAT_R16G16_SFLOAT);
  HostFormatPair& host_format_16_16_16_16 =
      host_formats_[uint32_t(xenos::TextureFormat::k_16_16_16_16)];
  assert_true(host_format_16_16_16_16.format_unsigned.format == VK_FORMAT_R16G16B16A16_UNORM);
  if ((r16g16b16a16_unorm_properties.optimalTilingFeatures & norm16_required_features) !=
      norm16_required_features) {
    host_format_16_16_16_16.format_unsigned.load_shader = kLoadShaderIndexRGBA16UNormToFloat;
    host_format_16_16_16_16.format_unsigned.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  }
  assert_true(host_format_16_16_16_16.format_signed.format == VK_FORMAT_R16G16B16A16_SNORM);
  if ((r16g16b16a16_snorm_properties.optimalTilingFeatures & norm16_required_features) !=
      norm16_required_features) {
    host_format_16_16_16_16.format_signed.load_shader = kLoadShaderIndexRGBA16SNormToFloat;
    host_format_16_16_16_16.format_signed.format = VK_FORMAT_R16G16B16A16_SFLOAT;
  }
  host_format_16_16_16_16.unsigned_signed_compatible =
      (host_format_16_16_16_16.format_unsigned.format == VK_FORMAT_R16G16B16A16_UNORM &&
       host_format_16_16_16_16.format_signed.format == VK_FORMAT_R16G16B16A16_SNORM) ||
      (host_format_16_16_16_16.format_unsigned.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
       host_format_16_16_16_16.format_signed.format == VK_FORMAT_R16G16B16A16_SFLOAT);

  // Normalize format information structures.
  for (size_t i = 0; i < rex::countof(host_formats_); ++i) {
    HostFormatPair& host_format = host_formats_[i];
    // load_shader_index is left uninitialized for the tail (non-existent
    // formats), kLoadShaderIndexUnknown may be non-zero, and format support may
    // be disabled by setting the format to VK_FORMAT_UNDEFINED.
    if (host_format.format_unsigned.format == VK_FORMAT_UNDEFINED) {
      host_format.format_unsigned.load_shader = kLoadShaderIndexUnknown;
    }
    assert_false(host_format.format_unsigned.load_shader == kLoadShaderIndexUnknown &&
                 host_format.format_unsigned.format != VK_FORMAT_UNDEFINED);
    if (host_format.format_unsigned.load_shader == kLoadShaderIndexUnknown) {
      host_format.format_unsigned.format = VK_FORMAT_UNDEFINED;
      // Surely known it's unsupported with these two conditions.
      host_format.format_unsigned.linear_filterable = false;
    }
    if (host_format.format_signed.format == VK_FORMAT_UNDEFINED) {
      host_format.format_signed.load_shader = kLoadShaderIndexUnknown;
    }
    assert_false(host_format.format_signed.load_shader == kLoadShaderIndexUnknown &&
                 host_format.format_signed.format != VK_FORMAT_UNDEFINED);
    if (host_format.format_signed.load_shader == kLoadShaderIndexUnknown) {
      host_format.format_signed.format = VK_FORMAT_UNDEFINED;
      // Surely known it's unsupported with these two conditions.
      host_format.format_signed.linear_filterable = false;
    }

    // Check if the formats are supported and are linear-filterable.
    if (host_format.format_unsigned.format != VK_FORMAT_UNDEFINED) {
      ifn.vkGetPhysicalDeviceFormatProperties(physical_device, host_format.format_unsigned.format,
                                              &format_properties);
      if (format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
        host_format.format_unsigned.linear_filterable =
            (format_properties.optimalTilingFeatures &
             VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
      } else {
        host_format.format_unsigned.format = VK_FORMAT_UNDEFINED;
        host_format.format_unsigned.load_shader = kLoadShaderIndexUnknown;
        host_format.format_unsigned.linear_filterable = false;
      }
    }
    if (host_format.format_signed.format != VK_FORMAT_UNDEFINED) {
      ifn.vkGetPhysicalDeviceFormatProperties(physical_device, host_format.format_signed.format,
                                              &format_properties);
      if (format_properties.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT) {
        host_format.format_signed.linear_filterable =
            (format_properties.optimalTilingFeatures &
             VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT) != 0;
      } else {
        host_format.format_signed.format = VK_FORMAT_UNDEFINED;
        host_format.format_signed.load_shader = kLoadShaderIndexUnknown;
        host_format.format_signed.linear_filterable = false;
      }
    }

    // Log which formats are not supported or supported via fallbacks.
    const HostFormatPair& best_host_format = kBestHostFormats[i];
    const char* guest_format_name = FormatInfo::Get(xenos::TextureFormat(i))->name;
    if (best_host_format.format_unsigned.format != VK_FORMAT_UNDEFINED) {
      assert_not_null(guest_format_name);
      if (host_format.format_unsigned.format != VK_FORMAT_UNDEFINED) {
        if (host_format.format_unsigned.format != best_host_format.format_unsigned.format) {
          REXGPU_INFO(
              "VulkanTextureCache: Format {} (unsigned) is supported via a "
              "fallback format (using the Vulkan format {} instead of the "
              "preferred {})",
              guest_format_name, uint32_t(host_format.format_unsigned.format),
              uint32_t(best_host_format.format_unsigned.format));
        }
      } else {
        REXGPU_INFO(
            "VulkanTextureCache: Format {} (unsigned) is not supported by the "
            "device (preferred Vulkan format is {})",
            guest_format_name, uint32_t(best_host_format.format_unsigned.format));
      }
    }
    if (best_host_format.format_signed.format != VK_FORMAT_UNDEFINED) {
      assert_not_null(guest_format_name);
      if (host_format.format_signed.format != VK_FORMAT_UNDEFINED) {
        if (host_format.format_signed.format != best_host_format.format_signed.format) {
          REXGPU_INFO(
              "VulkanTextureCache: Format {} (signed) is supported via a "
              "fallback format (using the Vulkan format {} instead of the "
              "preferred {})",
              guest_format_name, uint32_t(host_format.format_signed.format),
              uint32_t(best_host_format.format_signed.format));
        }
      } else {
        REXGPU_INFO(
            "VulkanTextureCache: Format {} (signed) is not supported by the "
            "device (preferred Vulkan format is {})",
            guest_format_name, uint32_t(best_host_format.format_signed.format));
      }
    }

    // Signednesses with different load shaders must have the data loaded
    // differently, therefore can't share the image even if the format is the
    // same. Also, if there's only one version, simplify the logic - there can't
    // be compatibility between two formats when one of them is undefined.
    if (host_format.format_unsigned.format != VK_FORMAT_UNDEFINED &&
        host_format.format_signed.format != VK_FORMAT_UNDEFINED) {
      if (host_format.format_unsigned.load_shader == host_format.format_signed.load_shader) {
        if (host_format.format_unsigned.format == host_format.format_signed.format) {
          // Same format after all the fallbacks - force compatibilty.
          host_format.unsigned_signed_compatible = true;
        }
      } else {
        host_format.unsigned_signed_compatible = false;
      }
      // Formats within the same compatibility class must have the same block
      // size, though the fallbacks are configured incorrectly if that's not the
      // case (since such formats just can't be in one compatibility class).
      assert_false(host_format.unsigned_signed_compatible &&
                   host_format.format_unsigned.block_compressed !=
                       host_format.format_signed.block_compressed);
      if (host_format.unsigned_signed_compatible &&
          host_format.format_unsigned.block_compressed !=
              host_format.format_signed.block_compressed) {
        host_format.unsigned_signed_compatible = false;
      }
    } else {
      host_format.unsigned_signed_compatible = false;
    }
  }

  // Load pipeline layout.

  VkDescriptorSetLayout load_descriptor_set_layouts[kLoadDescriptorSetCount] = {};
  VkDescriptorSetLayout load_descriptor_set_layout_storage_buffer =
      command_processor_.GetSingleTransientDescriptorLayout(
          VulkanCommandProcessor::SingleTransientDescriptorLayout ::kStorageBufferCompute);
  assert_true(load_descriptor_set_layout_storage_buffer != VK_NULL_HANDLE);
  load_descriptor_set_layouts[kLoadDescriptorSetIndexDestination] =
      load_descriptor_set_layout_storage_buffer;
  load_descriptor_set_layouts[kLoadDescriptorSetIndexSource] =
      load_descriptor_set_layout_storage_buffer;
  VkPushConstantRange load_pipeline_layout_push_constant_range;
  load_pipeline_layout_push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
  load_pipeline_layout_push_constant_range.offset = 0;
  load_pipeline_layout_push_constant_range.size = sizeof(LoadConstants);
  VkPipelineLayoutCreateInfo load_pipeline_layout_create_info;
  load_pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  load_pipeline_layout_create_info.pNext = nullptr;
  load_pipeline_layout_create_info.flags = 0;
  load_pipeline_layout_create_info.setLayoutCount = kLoadDescriptorSetCount;
  load_pipeline_layout_create_info.pSetLayouts = load_descriptor_set_layouts;
  load_pipeline_layout_create_info.pushConstantRangeCount = 1;
  load_pipeline_layout_create_info.pPushConstantRanges = &load_pipeline_layout_push_constant_range;
  if (dfn.vkCreatePipelineLayout(device, &load_pipeline_layout_create_info, nullptr,
                                 &load_pipeline_layout_)) {
    REXGPU_ERROR("VulkanTexture: Failed to create the texture load pipeline layout");
    return false;
  }

  // Default-off direct storage-image load path: destination storage image (set
  // 0) + shared source storage buffer (set 1) + the same load push constants.
  direct_storage_image_enabled_ = REXCVAR_GET(gpu_direct_storage_image_texture_load);
  if (direct_storage_image_enabled_) {
    const ui::vulkan::VulkanInstance::Functions& ifn =
        vulkan_device->vulkan_instance()->functions();
    vk_free_descriptor_sets_ = reinterpret_cast<PFN_vkFreeDescriptorSets>(
        ifn.vkGetDeviceProcAddr(device, "vkFreeDescriptorSets"));
    static_assert(uint32_t(kDirectStorageImageWriteUsage.stage_mask) ==
                      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                  "Direct storage-image write stage mismatch");
    static_assert(uint32_t(kDirectStorageImageWriteUsage.access_mask) == VK_ACCESS_SHADER_WRITE_BIT,
                  "Direct storage-image write access mismatch");
    static_assert(kDirectStorageImageWriteUsage.layout == VK_IMAGE_LAYOUT_GENERAL,
                  "Direct storage-image write layout mismatch");
    static_assert(uint32_t(kDirectStorageImageSampledReadUsage.access_mask) ==
                      VK_ACCESS_SHADER_READ_BIT,
                  "Direct storage-image sampled access mismatch");
    static_assert(kDirectStorageImageSampledReadUsage.layout ==
                      VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                  "Direct storage-image sampled layout mismatch");
    static_assert(kDirectStorageImageGuestFormat == uint32_t(xenos::TextureFormat::k_8_8_8_8),
                  "Direct storage-image guest format mismatch");
    static_assert(kDirectStorageImageHostFormat == uint32_t(VK_FORMAT_R8G8B8A8_UNORM),
                  "Direct storage-image host format mismatch");
    static_assert(kDirectStorageImageDepthLoadShader == uint32_t(kLoadShaderIndexDepthUnorm));
    static_assert(kDirectStorageImageDepthHostFormat == uint32_t(VK_FORMAT_R32_SFLOAT));
    static_assert(kDirectStorageImageDepthGuestFormat == uint32_t(xenos::TextureFormat::k_24_8));
    static_assert(kDirectStorageImageLoadShader == uint32_t(kLoadShaderIndex32bpb),
                  "Direct storage-image load shader mismatch");

    VkDescriptorSetLayoutBinding direct_image_binding;
    direct_image_binding.binding = 0;
    direct_image_binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    direct_image_binding.descriptorCount = 1;
    direct_image_binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    direct_image_binding.pImmutableSamplers = nullptr;
    VkDescriptorSetLayoutCreateInfo direct_image_layout_create_info;
    direct_image_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    direct_image_layout_create_info.pNext = nullptr;
    direct_image_layout_create_info.flags = 0;
    direct_image_layout_create_info.bindingCount = 1;
    direct_image_layout_create_info.pBindings = &direct_image_binding;
    VkDescriptorPoolSize direct_image_pool_size;
    direct_image_pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    direct_image_pool_size.descriptorCount = 256;
    VkDescriptorPoolCreateInfo direct_image_pool_create_info;
    direct_image_pool_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    direct_image_pool_create_info.pNext = nullptr;
    direct_image_pool_create_info.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    direct_image_pool_create_info.maxSets = 256;
    direct_image_pool_create_info.poolSizeCount = 1;
    direct_image_pool_create_info.pPoolSizes = &direct_image_pool_size;
    VkDescriptorSetLayout direct_image_descriptor_set_layouts[kLoadDescriptorSetCount] = {
        VK_NULL_HANDLE, load_descriptor_set_layout_storage_buffer};
    VkPipelineLayoutCreateInfo direct_image_pipeline_layout_create_info =
        load_pipeline_layout_create_info;
    direct_image_pipeline_layout_create_info.pSetLayouts = direct_image_descriptor_set_layouts;
    bool direct_image_ready = false;
    if (dfn.vkCreateDescriptorSetLayout(device, &direct_image_layout_create_info, nullptr,
                                        &direct_storage_image_descriptor_set_layout_) ||
        dfn.vkCreateDescriptorPool(device, &direct_image_pool_create_info, nullptr,
                                   &direct_storage_image_descriptor_pool_)) {
      REXGPU_ERROR("VulkanTexture: Failed to create the direct storage-image descriptors");
    } else {
      direct_image_descriptor_set_layouts[kLoadDescriptorSetIndexDestination] =
          direct_storage_image_descriptor_set_layout_;
      if (dfn.vkCreatePipelineLayout(device, &direct_image_pipeline_layout_create_info, nullptr,
                                     &direct_storage_image_pipeline_layout_)) {
        REXGPU_ERROR("VulkanTexture: Failed to create the direct storage-image pipeline layout");
      } else {
        std::string direct_image_compile_error;
        std::vector<uint32_t> direct_image_spirv;
        if (!command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT,
                                                  kDirectStorageImageLoadShaderSource,
                                                  direct_image_spirv, direct_image_compile_error)) {
          REXGPU_ERROR("VulkanTexture: Failed to compile the direct storage-image load shader: {}",
                       direct_image_compile_error);
        } else {
          direct_storage_image_pipeline_ = ui::vulkan::util::CreateComputePipeline(
              vulkan_device, direct_storage_image_pipeline_layout_, direct_image_spirv.data(),
              direct_image_spirv.size() * sizeof(uint32_t));
          if (direct_storage_image_pipeline_ == VK_NULL_HANDLE) {
            REXGPU_ERROR("VulkanTexture: Failed to create the direct storage-image pipeline");
          } else {
            direct_image_ready = true;
            direct_image_spirv.clear();
            if (command_processor_.CompileGlslToSpirv(VK_SHADER_STAGE_COMPUTE_BIT,
                    kDirectStorageImageDepthShaderSource, direct_image_spirv,
                    direct_image_compile_error)) {
              direct_storage_image_depth_pipeline_ = ui::vulkan::util::CreateComputePipeline(
                  vulkan_device, direct_storage_image_pipeline_layout_, direct_image_spirv.data(),
                  direct_image_spirv.size() * sizeof(uint32_t));
            } else {
              REXGPU_ERROR("Direct depth image shader compilation failed: {}", direct_image_compile_error);
            }
            REXGPU_INFO(
                "VulkanTexture: Direct storage-image load path enabled (R8G8B8A8_UNORM scaled "
                "resolves)");
          }
        }
      }
    }
    if (!direct_image_ready) {
      // Keep the cvar state for accounting, but never dispatch the direct path
      // without a pipeline - every load falls back to the buffer path.
      direct_storage_image_pipeline_ = VK_NULL_HANDLE;
      REXGPU_WARN(
          "VulkanTexture: Direct storage-image load path unavailable, using the buffer-and-copy "
          "path");
    }
  }

  // Load pipelines, only the ones needed for the formats that will be used.

  bool load_shaders_needed[kLoadShaderCount] = {};
  auto mark_host_format_pair_load_shaders_needed = [&](const HostFormatPair& host_format_pair) {
    if (host_format_pair.format_unsigned.load_shader != kLoadShaderIndexUnknown) {
      load_shaders_needed[host_format_pair.format_unsigned.load_shader] = true;
    }
    if (host_format_pair.format_signed.load_shader != kLoadShaderIndexUnknown) {
      load_shaders_needed[host_format_pair.format_signed.load_shader] = true;
    }
  };
  for (size_t i = 0; i < rex::countof(host_formats_); ++i) {
    mark_host_format_pair_load_shaders_needed(host_formats_[i]);
  }
  mark_host_format_pair_load_shaders_needed(kHostFormatGBGRUnaligned);
  mark_host_format_pair_load_shaders_needed(kHostFormatBGRGUnaligned);
  mark_host_format_pair_load_shaders_needed(kHostFormatDXT1Unaligned);
  mark_host_format_pair_load_shaders_needed(kHostFormatDXT2_3Unaligned);
  mark_host_format_pair_load_shaders_needed(kHostFormatDXT4_5Unaligned);
  mark_host_format_pair_load_shaders_needed(kHostFormatDXNUnaligned);
  mark_host_format_pair_load_shaders_needed(kHostFormatDXT5AUnaligned);

  std::pair<const uint32_t*, size_t> load_shader_code[kLoadShaderCount] = {};
  load_shader_code[kLoadShaderIndex8bpb] =
      std::make_pair(shaders::texture_load_8bpb_cs, sizeof(shaders::texture_load_8bpb_cs));
  load_shader_code[kLoadShaderIndex16bpb] =
      std::make_pair(shaders::texture_load_16bpb_cs, sizeof(shaders::texture_load_16bpb_cs));
  load_shader_code[kLoadShaderIndex32bpb] =
      std::make_pair(shaders::texture_load_32bpb_cs, sizeof(shaders::texture_load_32bpb_cs));
  load_shader_code[kLoadShaderIndex64bpb] =
      std::make_pair(shaders::texture_load_64bpb_cs, sizeof(shaders::texture_load_64bpb_cs));
  load_shader_code[kLoadShaderIndex128bpb] =
      std::make_pair(shaders::texture_load_128bpb_cs, sizeof(shaders::texture_load_128bpb_cs));
  load_shader_code[kLoadShaderIndexR5G5B5A1ToB5G5R5A1] =
      std::make_pair(shaders::texture_load_r5g5b5a1_b5g5r5a1_cs,
                     sizeof(shaders::texture_load_r5g5b5a1_b5g5r5a1_cs));
  load_shader_code[kLoadShaderIndexR5G6B5ToB5G6R5] = std::make_pair(
      shaders::texture_load_r5g6b5_b5g6r5_cs, sizeof(shaders::texture_load_r5g6b5_b5g6r5_cs));
  load_shader_code[kLoadShaderIndexR5G6B5ToRGBA8] = std::make_pair(
      shaders::texture_load_r5g6b5_rgba8_cs, sizeof(shaders::texture_load_r5g6b5_rgba8_cs));
  load_shader_code[kLoadShaderIndexR5G5B6ToB5G6R5WithRBGASwizzle] =
      std::make_pair(shaders::texture_load_r5g5b6_b5g6r5_swizzle_rbga_cs,
                     sizeof(shaders::texture_load_r5g5b6_b5g6r5_swizzle_rbga_cs));
  load_shader_code[kLoadShaderIndexRGBA4ToARGB4] =
      std::make_pair(shaders::texture_load_r4g4b4a4_a4r4g4b4_cs,
                     sizeof(shaders::texture_load_r4g4b4a4_a4r4g4b4_cs));
  load_shader_code[kLoadShaderIndexRGBA4ToRGBA8] = std::make_pair(
      shaders::texture_load_r4g4b4a4_rgba8_cs, sizeof(shaders::texture_load_r4g4b4a4_rgba8_cs));
  load_shader_code[kLoadShaderIndexGBGR8ToRGB8] = std::make_pair(
      shaders::texture_load_gbgr8_rgb8_cs, sizeof(shaders::texture_load_gbgr8_rgb8_cs));
  load_shader_code[kLoadShaderIndexBGRG8ToRGB8] = std::make_pair(
      shaders::texture_load_bgrg8_rgb8_cs, sizeof(shaders::texture_load_bgrg8_rgb8_cs));
  load_shader_code[kLoadShaderIndexR10G11B11ToRGBA16] = std::make_pair(
      shaders::texture_load_r10g11b11_rgba16_cs, sizeof(shaders::texture_load_r10g11b11_rgba16_cs));
  load_shader_code[kLoadShaderIndexR10G11B11ToRGBA16SNorm] =
      std::make_pair(shaders::texture_load_r10g11b11_rgba16_snorm_cs,
                     sizeof(shaders::texture_load_r10g11b11_rgba16_snorm_cs));
  load_shader_code[kLoadShaderIndexR11G11B10ToRGBA16] = std::make_pair(
      shaders::texture_load_r11g11b10_rgba16_cs, sizeof(shaders::texture_load_r11g11b10_rgba16_cs));
  load_shader_code[kLoadShaderIndexR11G11B10ToRGBA16SNorm] =
      std::make_pair(shaders::texture_load_r11g11b10_rgba16_snorm_cs,
                     sizeof(shaders::texture_load_r11g11b10_rgba16_snorm_cs));
  load_shader_code[kLoadShaderIndexR16UNormToFloat] = std::make_pair(
      shaders::texture_load_r16_unorm_float_cs, sizeof(shaders::texture_load_r16_unorm_float_cs));
  load_shader_code[kLoadShaderIndexR16SNormToFloat] = std::make_pair(
      shaders::texture_load_r16_snorm_float_cs, sizeof(shaders::texture_load_r16_snorm_float_cs));
  load_shader_code[kLoadShaderIndexRG16UNormToFloat] = std::make_pair(
      shaders::texture_load_rg16_unorm_float_cs, sizeof(shaders::texture_load_rg16_unorm_float_cs));
  load_shader_code[kLoadShaderIndexRG16SNormToFloat] = std::make_pair(
      shaders::texture_load_rg16_snorm_float_cs, sizeof(shaders::texture_load_rg16_snorm_float_cs));
  load_shader_code[kLoadShaderIndexRGBA16UNormToFloat] =
      std::make_pair(shaders::texture_load_rgba16_unorm_float_cs,
                     sizeof(shaders::texture_load_rgba16_unorm_float_cs));
  load_shader_code[kLoadShaderIndexRGBA16SNormToFloat] =
      std::make_pair(shaders::texture_load_rgba16_snorm_float_cs,
                     sizeof(shaders::texture_load_rgba16_snorm_float_cs));
  load_shader_code[kLoadShaderIndexDXT1ToRGBA8] = std::make_pair(
      shaders::texture_load_dxt1_rgba8_cs, sizeof(shaders::texture_load_dxt1_rgba8_cs));
  load_shader_code[kLoadShaderIndexDXT3ToRGBA8] = std::make_pair(
      shaders::texture_load_dxt3_rgba8_cs, sizeof(shaders::texture_load_dxt3_rgba8_cs));
  load_shader_code[kLoadShaderIndexDXT5ToRGBA8] = std::make_pair(
      shaders::texture_load_dxt5_rgba8_cs, sizeof(shaders::texture_load_dxt5_rgba8_cs));
  load_shader_code[kLoadShaderIndexDXNToRG8] =
      std::make_pair(shaders::texture_load_dxn_rg8_cs, sizeof(shaders::texture_load_dxn_rg8_cs));
  load_shader_code[kLoadShaderIndexDXT3A] =
      std::make_pair(shaders::texture_load_dxt3a_cs, sizeof(shaders::texture_load_dxt3a_cs));
  load_shader_code[kLoadShaderIndexDXT3AAs1111ToARGB4] =
      std::make_pair(shaders::texture_load_dxt3aas1111_argb4_cs,
                     sizeof(shaders::texture_load_dxt3aas1111_argb4_cs));
  load_shader_code[kLoadShaderIndexDXT5AToR8] =
      std::make_pair(shaders::texture_load_dxt5a_r8_cs, sizeof(shaders::texture_load_dxt5a_r8_cs));
  load_shader_code[kLoadShaderIndexCTX1] =
      std::make_pair(shaders::texture_load_ctx1_cs, sizeof(shaders::texture_load_ctx1_cs));
  load_shader_code[kLoadShaderIndexDepthUnorm] = std::make_pair(
      shaders::texture_load_depth_unorm_cs, sizeof(shaders::texture_load_depth_unorm_cs));
  load_shader_code[kLoadShaderIndexDepthFloat] = std::make_pair(
      shaders::texture_load_depth_float_cs, sizeof(shaders::texture_load_depth_float_cs));
  std::pair<const uint32_t*, size_t> load_shader_code_scaled[kLoadShaderCount] = {};
  if (IsDrawResolutionScaled()) {
    load_shader_code_scaled[kLoadShaderIndex8bpb] = std::make_pair(
        shaders::texture_load_8bpb_scaled_cs, sizeof(shaders::texture_load_8bpb_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndex16bpb] = std::make_pair(
        shaders::texture_load_16bpb_scaled_cs, sizeof(shaders::texture_load_16bpb_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndex32bpb] = std::make_pair(
        shaders::texture_load_32bpb_scaled_cs, sizeof(shaders::texture_load_32bpb_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndex64bpb] = std::make_pair(
        shaders::texture_load_64bpb_scaled_cs, sizeof(shaders::texture_load_64bpb_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndex128bpb] = std::make_pair(
        shaders::texture_load_128bpb_scaled_cs, sizeof(shaders::texture_load_128bpb_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR5G5B5A1ToB5G5R5A1] =
        std::make_pair(shaders::texture_load_r5g5b5a1_b5g5r5a1_scaled_cs,
                       sizeof(shaders::texture_load_r5g5b5a1_b5g5r5a1_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR5G6B5ToB5G6R5] =
        std::make_pair(shaders::texture_load_r5g6b5_b5g6r5_scaled_cs,
                       sizeof(shaders::texture_load_r5g6b5_b5g6r5_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR5G6B5ToRGBA8] =
        std::make_pair(shaders::texture_load_r5g6b5_rgba8_scaled_cs,
                       sizeof(shaders::texture_load_r5g6b5_rgba8_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR5G5B6ToB5G6R5WithRBGASwizzle] =
        std::make_pair(shaders::texture_load_r5g5b6_b5g6r5_swizzle_rbga_scaled_cs,
                       sizeof(shaders::texture_load_r5g5b6_b5g6r5_swizzle_rbga_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexRGBA4ToARGB4] =
        std::make_pair(shaders::texture_load_r4g4b4a4_a4r4g4b4_scaled_cs,
                       sizeof(shaders::texture_load_r4g4b4a4_a4r4g4b4_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexRGBA4ToRGBA8] =
        std::make_pair(shaders::texture_load_r4g4b4a4_rgba8_scaled_cs,
                       sizeof(shaders::texture_load_r4g4b4a4_rgba8_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR10G11B11ToRGBA16] =
        std::make_pair(shaders::texture_load_r10g11b11_rgba16_scaled_cs,
                       sizeof(shaders::texture_load_r10g11b11_rgba16_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR10G11B11ToRGBA16SNorm] =
        std::make_pair(shaders::texture_load_r10g11b11_rgba16_snorm_scaled_cs,
                       sizeof(shaders::texture_load_r10g11b11_rgba16_snorm_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR11G11B10ToRGBA16] =
        std::make_pair(shaders::texture_load_r11g11b10_rgba16_scaled_cs,
                       sizeof(shaders::texture_load_r11g11b10_rgba16_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR11G11B10ToRGBA16SNorm] =
        std::make_pair(shaders::texture_load_r11g11b10_rgba16_snorm_scaled_cs,
                       sizeof(shaders::texture_load_r11g11b10_rgba16_snorm_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR16UNormToFloat] =
        std::make_pair(shaders::texture_load_r16_unorm_float_scaled_cs,
                       sizeof(shaders::texture_load_r16_unorm_float_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexR16SNormToFloat] =
        std::make_pair(shaders::texture_load_r16_snorm_float_scaled_cs,
                       sizeof(shaders::texture_load_r16_snorm_float_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexRG16UNormToFloat] =
        std::make_pair(shaders::texture_load_rg16_unorm_float_scaled_cs,
                       sizeof(shaders::texture_load_rg16_unorm_float_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexRG16SNormToFloat] =
        std::make_pair(shaders::texture_load_rg16_snorm_float_scaled_cs,
                       sizeof(shaders::texture_load_rg16_snorm_float_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexRGBA16UNormToFloat] =
        std::make_pair(shaders::texture_load_rgba16_unorm_float_scaled_cs,
                       sizeof(shaders::texture_load_rgba16_unorm_float_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexRGBA16SNormToFloat] =
        std::make_pair(shaders::texture_load_rgba16_snorm_float_scaled_cs,
                       sizeof(shaders::texture_load_rgba16_snorm_float_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexDepthUnorm] =
        std::make_pair(shaders::texture_load_depth_unorm_scaled_cs,
                       sizeof(shaders::texture_load_depth_unorm_scaled_cs));
    load_shader_code_scaled[kLoadShaderIndexDepthFloat] =
        std::make_pair(shaders::texture_load_depth_float_scaled_cs,
                       sizeof(shaders::texture_load_depth_float_scaled_cs));
  }

  for (size_t i = 0; i < kLoadShaderCount; ++i) {
    if (!load_shaders_needed[i]) {
      continue;
    }
    const std::pair<const uint32_t*, size_t>& current_load_shader_code = load_shader_code[i];
    assert_not_null(current_load_shader_code.first);
    load_pipelines_[i] = ui::vulkan::util::CreateComputePipeline(
        vulkan_device, load_pipeline_layout_, current_load_shader_code.first,
        current_load_shader_code.second);
    if (load_pipelines_[i] == VK_NULL_HANDLE) {
      REXGPU_ERROR(
          "VulkanTextureCache: Failed to create the texture loading pipeline "
          "for shader {}",
          i);
      return false;
    }
    if (IsDrawResolutionScaled()) {
      const std::pair<const uint32_t*, size_t>& current_load_shader_code_scaled =
          load_shader_code_scaled[i];
      if (current_load_shader_code_scaled.first) {
        load_pipelines_scaled_[i] = ui::vulkan::util::CreateComputePipeline(
            vulkan_device, load_pipeline_layout_, current_load_shader_code_scaled.first,
            current_load_shader_code_scaled.second);
        if (load_pipelines_scaled_[i] == VK_NULL_HANDLE) {
          REXGPU_ERROR(
              "VulkanTextureCache: Failed to create the resolution-scaled "
              "texture loading pipeline for shader {}",
              i);
          return false;
        }
      }
    }
  }

  // Use true null descriptors when supported; otherwise, keep compatibility
  // fallback null images.
  if (device_properties.nullDescriptor) {
    null_images_cleared_ = true;
  } else {
    // Null images as a replacement for unneeded bindings and for bindings for
    // which the real image hasn't been created.
    VkImageCreateInfo null_image_create_info;
    null_image_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    null_image_create_info.pNext = nullptr;
    null_image_create_info.flags = VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    null_image_create_info.imageType = VK_IMAGE_TYPE_2D;
    // Match D3D12 null SRV behavior: invalid fetch constants read (0, 0, 0, 0).
    null_image_create_info.format = kInvalidTextureFetchFallbackFormat;
    null_image_create_info.extent.width = 1;
    null_image_create_info.extent.height = 1;
    null_image_create_info.extent.depth = 1;
    null_image_create_info.mipLevels = 1;
    null_image_create_info.arrayLayers = 6;
    null_image_create_info.samples = VK_SAMPLE_COUNT_1_BIT;
    null_image_create_info.tiling = VK_IMAGE_TILING_OPTIMAL;
    null_image_create_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    null_image_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    null_image_create_info.queueFamilyIndexCount = 0;
    null_image_create_info.pQueueFamilyIndices = nullptr;
    null_image_create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    if (dfn.vkCreateImage(device, &null_image_create_info, nullptr, &null_image_2d_array_cube_) !=
        VK_SUCCESS) {
      REXGPU_ERROR(
          "VulkanTextureCache: Failed to create the null 2D array and cube "
          "image");
      return false;
    }

    null_image_create_info.flags &= ~VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT;
    null_image_create_info.imageType = VK_IMAGE_TYPE_3D;
    null_image_create_info.arrayLayers = 1;
    if (dfn.vkCreateImage(device, &null_image_create_info, nullptr, &null_image_3d_) !=
        VK_SUCCESS) {
      REXGPU_ERROR("VulkanTextureCache: Failed to create the null 3D image");
      return false;
    }

    VkMemoryRequirements null_image_memory_requirements_2d_array_cube_;
    dfn.vkGetImageMemoryRequirements(device, null_image_2d_array_cube_,
                                     &null_image_memory_requirements_2d_array_cube_);
    VkMemoryRequirements null_image_memory_requirements_3d_;
    dfn.vkGetImageMemoryRequirements(device, null_image_3d_, &null_image_memory_requirements_3d_);
    uint32_t null_image_memory_type_common = ui::vulkan::util::ChooseMemoryType(
        vulkan_device->memory_types(),
        null_image_memory_requirements_2d_array_cube_.memoryTypeBits &
            null_image_memory_requirements_3d_.memoryTypeBits,
        ui::vulkan::util::MemoryPurpose::kDeviceLocal);
    if (null_image_memory_type_common != UINT32_MAX) {
      // Place both null images in one memory allocation because maximum total
      // memory allocation count is limited.
      VkDeviceSize null_image_memory_offset_3d_ =
          rex::align(null_image_memory_requirements_2d_array_cube_.size,
                     std::max(null_image_memory_requirements_3d_.alignment, VkDeviceSize(1)));
      VkMemoryAllocateInfo null_image_memory_allocate_info;
      null_image_memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      null_image_memory_allocate_info.pNext = nullptr;
      null_image_memory_allocate_info.allocationSize =
          null_image_memory_offset_3d_ + null_image_memory_requirements_3d_.size;
      null_image_memory_allocate_info.memoryTypeIndex = null_image_memory_type_common;
      if (dfn.vkAllocateMemory(device, &null_image_memory_allocate_info, nullptr,
                               &null_images_memory_[0]) != VK_SUCCESS) {
        REXGPU_ERROR(
            "VulkanTextureCache: Failed to allocate the memory for the null "
            "images");
        return false;
      }
      if (dfn.vkBindImageMemory(device, null_image_2d_array_cube_, null_images_memory_[0], 0) !=
          VK_SUCCESS) {
        REXGPU_ERROR(
            "VulkanTextureCache: Failed to bind the memory to the null 2D array "
            "and cube image");
        return false;
      }
      if (dfn.vkBindImageMemory(device, null_image_3d_, null_images_memory_[0],
                                null_image_memory_offset_3d_) != VK_SUCCESS) {
        REXGPU_ERROR("VulkanTextureCache: Failed to bind the memory to the null 3D image");
        return false;
      }
    } else {
      // Place each null image in separate allocations.
      const uint32_t null_image_memory_type_2d_array_cube = ui::vulkan::util::ChooseMemoryType(
          vulkan_device->memory_types(),
          null_image_memory_requirements_2d_array_cube_.memoryTypeBits,
          ui::vulkan::util::MemoryPurpose::kDeviceLocal);
      const uint32_t null_image_memory_type_3d = ui::vulkan::util::ChooseMemoryType(
          vulkan_device->memory_types(), null_image_memory_requirements_3d_.memoryTypeBits,
          ui::vulkan::util::MemoryPurpose::kDeviceLocal);
      if (null_image_memory_type_2d_array_cube == UINT32_MAX ||
          null_image_memory_type_3d == UINT32_MAX) {
        REXGPU_ERROR(
            "VulkanTextureCache: Failed to get the memory types for the null "
            "images");
        return false;
      }

      VkMemoryAllocateInfo null_image_memory_allocate_info;
      VkMemoryAllocateInfo* null_image_memory_allocate_info_last = &null_image_memory_allocate_info;
      null_image_memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
      null_image_memory_allocate_info.pNext = nullptr;
      null_image_memory_allocate_info.allocationSize =
          null_image_memory_requirements_2d_array_cube_.size;
      null_image_memory_allocate_info.memoryTypeIndex = null_image_memory_type_2d_array_cube;
      VkMemoryDedicatedAllocateInfo null_image_memory_dedicated_allocate_info;
      if (vulkan_device->extensions().ext_1_1_KHR_dedicated_allocation) {
        null_image_memory_allocate_info_last->pNext = &null_image_memory_dedicated_allocate_info;
        null_image_memory_allocate_info_last =
            reinterpret_cast<VkMemoryAllocateInfo*>(&null_image_memory_dedicated_allocate_info);
        null_image_memory_dedicated_allocate_info.sType =
            VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO;
        null_image_memory_dedicated_allocate_info.pNext = nullptr;
        null_image_memory_dedicated_allocate_info.image = null_image_2d_array_cube_;
        null_image_memory_dedicated_allocate_info.buffer = VK_NULL_HANDLE;
      }
      if (dfn.vkAllocateMemory(device, &null_image_memory_allocate_info, nullptr,
                               &null_images_memory_[0]) != VK_SUCCESS) {
        REXGPU_ERROR(
            "VulkanTextureCache: Failed to allocate the memory for the null 2D "
            "array and cube image");
        return false;
      }
      if (dfn.vkBindImageMemory(device, null_image_2d_array_cube_, null_images_memory_[0], 0) !=
          VK_SUCCESS) {
        REXGPU_ERROR(
            "VulkanTextureCache: Failed to bind the memory to the null 2D array "
            "and cube image");
        return false;
      }

      null_image_memory_allocate_info.allocationSize = null_image_memory_requirements_3d_.size;
      null_image_memory_allocate_info.memoryTypeIndex = null_image_memory_type_3d;
      null_image_memory_dedicated_allocate_info.image = null_image_3d_;
      if (dfn.vkAllocateMemory(device, &null_image_memory_allocate_info, nullptr,
                               &null_images_memory_[1]) != VK_SUCCESS) {
        REXGPU_ERROR(
            "VulkanTextureCache: Failed to allocate the memory for the null 3D "
            "image");
        return false;
      }
      if (dfn.vkBindImageMemory(device, null_image_3d_, null_images_memory_[1], 0) != VK_SUCCESS) {
        REXGPU_ERROR("VulkanTextureCache: Failed to bind the memory to the null 3D image");
        return false;
      }
    }

    VkImageViewCreateInfo null_image_view_create_info;
    null_image_view_create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    null_image_view_create_info.pNext = nullptr;
    null_image_view_create_info.flags = 0;
    null_image_view_create_info.image = null_image_2d_array_cube_;
    null_image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    null_image_view_create_info.format = null_image_create_info.format;
    // Micro-optimization if this has any effect on the host GPU at all, use only
    // constant components instead of the real texels.
    VkComponentSwizzle null_image_view_swizzle = device_properties.imageViewFormatSwizzle
                                                     ? VK_COMPONENT_SWIZZLE_ZERO
                                                     : VK_COMPONENT_SWIZZLE_IDENTITY;
    null_image_view_create_info.components.r = null_image_view_swizzle;
    null_image_view_create_info.components.g = null_image_view_swizzle;
    null_image_view_create_info.components.b = null_image_view_swizzle;
    null_image_view_create_info.components.a = null_image_view_swizzle;
    null_image_view_create_info.subresourceRange = ui::vulkan::util::InitializeSubresourceRange(
        VK_IMAGE_ASPECT_COLOR_BIT, 0, VK_REMAINING_MIP_LEVELS, 0, 1);
    if (dfn.vkCreateImageView(device, &null_image_view_create_info, nullptr,
                              &null_image_view_2d_array_) != VK_SUCCESS) {
      REXGPU_ERROR("VulkanTextureCache: Failed to create the null 2D array image view");
      return false;
    }
    null_image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_CUBE;
    null_image_view_create_info.subresourceRange.layerCount = 6;
    if (dfn.vkCreateImageView(device, &null_image_view_create_info, nullptr,
                              &null_image_view_cube_) != VK_SUCCESS) {
      REXGPU_ERROR("VulkanTextureCache: Failed to create the null cube image view");
      return false;
    }
    null_image_view_create_info.image = null_image_3d_;
    null_image_view_create_info.viewType = VK_IMAGE_VIEW_TYPE_3D;
    null_image_view_create_info.subresourceRange.layerCount = 1;
    if (dfn.vkCreateImageView(device, &null_image_view_create_info, nullptr,
                              &null_image_view_3d_) != VK_SUCCESS) {
      REXGPU_ERROR("VulkanTextureCache: Failed to create the null 3D image view");
      return false;
    }

    null_images_cleared_ = false;
  }

  // Samplers.

  // Some MoltenVK devices have a maximum of 2048, 1024, or even 96 samplers,
  // below Vulkan's minimum requirement of 4000.
  // Assuming that the current VulkanTextureCache is the only one on this
  // VkDevice (true in a regular emulation scenario), so taking over all the
  // allocation slots exclusively.
  // Also leaving a few slots for use by things like overlay applications.
  sampler_max_count_ =
      device_properties.maxSamplerAllocationCount - ui::vulkan::UISamplers::kSamplerCount - 16;

  if (device_properties.samplerAnisotropy) {
    max_anisotropy_ = xenos::AnisoFilter(
        uint32_t(xenos::AnisoFilter::kMax_1_1) +
        (31 - rex::lzcnt(uint32_t(
                  std::min(16.0f, std::max(1.0f, device_properties.maxSamplerAnisotropy))))));
  } else {
    max_anisotropy_ = xenos::AnisoFilter::kDisabled;
  }

  return true;
}

const VulkanTextureCache::HostFormatPair& VulkanTextureCache::GetHostFormatPair(
    TextureKey key) const {
  const HostFormatPair& host_format_pair = host_formats_[uint32_t(key.format)];
  if (!host_format_pair.format_unsigned.block_compressed &&
      !host_format_pair.format_signed.block_compressed) {
    return host_format_pair;
  }

  // Keep unaligned decompression selection aligned with D3D12 fallback behavior:
  // only select unaligned fallback mappings when block alignment is actually
  // violated for the current guest format.
  const FormatInfo* format_info = FormatInfo::Get(key.format);
  if (!(key.GetWidth() & (format_info->block_width - 1)) &&
      !(key.GetHeight() & (format_info->block_height - 1))) {
    return host_format_pair;
  }

  switch (key.format) {
    case xenos::TextureFormat::k_Cr_Y1_Cb_Y0_REP:
      return kHostFormatGBGRUnaligned;
    case xenos::TextureFormat::k_Y1_Cr_Y0_Cb_REP:
      return kHostFormatBGRGUnaligned;
    case xenos::TextureFormat::k_DXT1:
    case xenos::TextureFormat::k_DXT1_AS_16_16_16_16:
      return kHostFormatDXT1Unaligned;
    case xenos::TextureFormat::k_DXT2_3:
    case xenos::TextureFormat::k_DXT2_3_AS_16_16_16_16:
      return kHostFormatDXT2_3Unaligned;
    case xenos::TextureFormat::k_DXT4_5:
    case xenos::TextureFormat::k_DXT4_5_AS_16_16_16_16:
      return kHostFormatDXT4_5Unaligned;
    case xenos::TextureFormat::k_DXN:
      return kHostFormatDXNUnaligned;
    case xenos::TextureFormat::k_DXT5A:
      return kHostFormatDXT5AUnaligned;
    default:
      return host_format_pair;
  }
}

void VulkanTextureCache::GetTextureUsageMasks(VulkanTexture::Usage usage,
                                              VkPipelineStageFlags& stage_mask,
                                              VkAccessFlags& access_mask, VkImageLayout& layout) {
  stage_mask = 0;
  access_mask = 0;
  layout = VK_IMAGE_LAYOUT_UNDEFINED;
  switch (usage) {
    case VulkanTexture::Usage::kUndefined:
      break;
    case VulkanTexture::Usage::kParityTransferSource:
      stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;
      access_mask = VK_ACCESS_TRANSFER_READ_BIT;
      layout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
      break;
    case VulkanTexture::Usage::kTransferDestination:
      stage_mask = VK_PIPELINE_STAGE_TRANSFER_BIT;
      access_mask = VK_ACCESS_TRANSFER_WRITE_BIT;
      layout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
      break;
    case VulkanTexture::Usage::kComputeStorageWrite:
      // Default-off direct storage-image load: the load compute shader owns the
      // image in GENERAL, so the next consumer (guest fragment/compute
      // sampling) must wait for the compute writes and transition to
      // SHADER_READ_ONLY_OPTIMAL.
      stage_mask = VkPipelineStageFlags(kDirectStorageImageWriteUsage.stage_mask);
      access_mask = VkAccessFlags(kDirectStorageImageWriteUsage.access_mask);
      layout = VkImageLayout(kDirectStorageImageWriteUsage.layout);
      break;
    case VulkanTexture::Usage::kGuestShaderSampled:
      stage_mask = guest_shader_pipeline_stages_;
      access_mask = VK_ACCESS_SHADER_READ_BIT;
      layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      break;
    case VulkanTexture::Usage::kSwapSampled:
      // The swap texture is likely to be used only for the presentation
      // post-processing passes, and not during emulation where it'd be used in
      // other stages.
      stage_mask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
      access_mask = VK_ACCESS_SHADER_READ_BIT;
      layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
      break;
  }
}

bool VulkanTextureCache::IsInGpuFrameAuditWindow() const {
  return command_processor_.IsGpuFrameAuditWindow();
}

bool VulkanTextureCache::QueryDirectStorageImageSupport(
    const VkImageCreateInfo& image_create_info) {
  const ui::vulkan::VulkanDevice* const vulkan_device = command_processor_.GetVulkanDevice();
  const ui::vulkan::VulkanInstance* const vulkan_instance = vulkan_device->vulkan_instance();
  const ui::vulkan::VulkanInstance::Functions& ifn = vulkan_instance->functions();
  PFN_vkGetPhysicalDeviceImageFormatProperties get_image_format_properties =
      reinterpret_cast<PFN_vkGetPhysicalDeviceImageFormatProperties>(ifn.vkGetInstanceProcAddr(
          vulkan_instance->instance(), "vkGetPhysicalDeviceImageFormatProperties"));
  if (get_image_format_properties == nullptr) {
    return false;
  }
  VkImageFormatProperties image_format_properties;
  const VkResult result = get_image_format_properties(
      vulkan_device->physical_device(), image_create_info.format, image_create_info.imageType,
      image_create_info.tiling, image_create_info.usage, image_create_info.flags,
      &image_format_properties);
  return result == VK_SUCCESS;
}

void VulkanTextureCache::CountDirectStorageImageResult(DirectStorageImageFallback fallback,
                                                       uint64_t copy_bytes) {
  if (fallback == DirectStorageImageFallback::kNone) {
    direct_storage_image_loads_.fetch_add(1, std::memory_order_relaxed);
    direct_storage_image_copy_bytes_avoided_.fetch_add(copy_bytes, std::memory_order_relaxed);
  } else {
    direct_storage_image_fallbacks_.fetch_add(1, std::memory_order_relaxed);
    direct_storage_image_fallbacks_by_reason_[size_t(fallback)].fetch_add(
        1, std::memory_order_relaxed);
    direct_storage_image_legacy_copy_bytes_.fetch_add(copy_bytes, std::memory_order_relaxed);
  }
}

VulkanTextureCache::DirectStorageImageStats VulkanTextureCache::GetDirectStorageImageStats()
    const {
  DirectStorageImageStats stats;
  stats.direct_loads = direct_storage_image_loads_.load(std::memory_order_relaxed);
  stats.fallbacks = direct_storage_image_fallbacks_.load(std::memory_order_relaxed);
  stats.copy_bytes_avoided =
      direct_storage_image_copy_bytes_avoided_.load(std::memory_order_relaxed);
  stats.legacy_copy_bytes =
      direct_storage_image_legacy_copy_bytes_.load(std::memory_order_relaxed);
  for (size_t i = 0; i < size_t(DirectStorageImageFallback::kCount); ++i) {
    stats.fallbacks_by_reason[i] =
        direct_storage_image_fallbacks_by_reason_[i].load(std::memory_order_relaxed);
  }
  return stats;
}

xenos::ClampMode VulkanTextureCache::NormalizeClampMode(xenos::ClampMode clamp_mode) const {
  if (clamp_mode == xenos::ClampMode::kClampToHalfway) {
    // No GL_CLAMP (clamp to half edge, half border) equivalent in Vulkan, but
    // there's no Direct3D 9 equivalent anyway, and too weird to be suitable for
    // intentional real usage.
    return xenos::ClampMode::kClampToEdge;
  }
  if (clamp_mode == xenos::ClampMode::kMirrorClampToEdge ||
      clamp_mode == xenos::ClampMode::kMirrorClampToHalfway ||
      clamp_mode == xenos::ClampMode::kMirrorClampToBorder) {
    // No equivalents for anything other than kMirrorClampToEdge in Vulkan.
    return command_processor_.GetVulkanDevice()->properties().samplerMirrorClampToEdge
               ? xenos::ClampMode::kMirrorClampToEdge
               : xenos::ClampMode::kMirroredRepeat;
  }
  return clamp_mode;
}

}  // namespace rex::graphics::vulkan
