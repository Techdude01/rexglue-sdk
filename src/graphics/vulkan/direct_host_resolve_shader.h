/**
 ******************************************************************************
 * Based on the Xenia Xbox 360 Emulator Research Project's resolve shader
 * addressing and conversion semantics.
 * Copyright 2022 Ben Vanik. All rights reserved.
 * Released under the BSD license - see the Xenia LICENSE for details.
 * ReXGlue adaptation: direct host-render-target source and scaled destination.
 ******************************************************************************
 */
#pragma once

namespace rex::graphics::vulkan {

// Narrow 32bpp scaled-resolve path. The source RT format and the resolve
// shader are selected by the C++ eligibility check; all other resolves retain
// the existing EDRAM-buffer path.
inline constexpr char kDirectHostResolveShader[] = R"GLSL(
#version 450
#extension GL_EXT_samplerless_texture_functions : require

#define DIRECT_SOURCE_MULTISAMPLED 0
#define DIRECT_SAMPLED_IMAGE_OUTPUT 0
#define DIRECT_SKIP_GUEST_BYTE_STORE 0
#if DIRECT_SOURCE_MULTISAMPLED
layout(set = 1, binding = 0) uniform texture2DMS source_image;
#else
layout(set = 1, binding = 0) uniform texture2D source_image;
#endif
layout(set = 0, binding = 0, std430) buffer Destination { uint data[]; } destination;
#if DIRECT_SAMPLED_IMAGE_OUTPUT
layout(set = 2, binding = 0, rgba8) uniform writeonly image2D sampled_destination;
#endif
layout(push_constant) uniform PushConstants {
  uint edram_info;
  uint coordinate_info;
  uint dest_info;
  uint dest_coordinate_info;
  uint dest_base;
  uint source_base_tiles;
  uint source_pitch_tiles;
  uint dispatch_first_tile;
  uint source_msaa_2x_supported;
} pc;
layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

const uint kEdramTileCount = 2048u;
const uint kEdramTileWidth = 80u;
const uint kEdramTileHeight = 16u;

uint TiledAddress2D(uvec2 p, uint pitch_macro_tiles) {
  uint outer_blocks = (((p.y >> 5u) * pitch_macro_tiles) + (p.x >> 5u)) << 6u;
  uint inner_blocks = (((p.y >> 1u) & 7u) << 3u) | (p.x & 7u);
  uint outer_inner_bytes = (outer_blocks | inner_blocks) << 2u;
  uint bank = (p.y >> 4u) & 1u;
  uint pipe = ((p.x >> 3u) & 3u) ^ (((p.y >> 3u) & 1u) << 1u);
  return ((p.y & 1u) << 4u) | (pipe << 6u) | (bank << 11u) |
         (outer_inner_bytes & 15u) | (((outer_inner_bytes >> 4u) & 1u) << 5u) |
         (((outer_inner_bytes >> 5u) & 7u) << 8u) | ((outer_inner_bytes >> 8u) << 12u);
}

uint ScaledDestinationAddress(uvec2 host_position, uvec2 scale) {
  uint pitch_macro_tiles = pc.dest_coordinate_info & 1023u;
  // Match this SDK's shipped scaled resolve and texture-load shaders:
  // RGBA8 uses 4x1 host-pixel sub-units, not newer upstream 16x2 groups.
  uvec2 group_blocks_log2 = uvec2(2u, 0u);
  uvec2 host_group_id = host_position >> group_blocks_log2;
  uvec2 guest_group_id = host_group_id / scale;
  uvec2 host_group_in_guest = host_group_id - scale * guest_group_id;
  uvec2 guest_group_origin = guest_group_id << group_blocks_log2;
  uint host_group_index = host_group_in_guest.x * scale.y + host_group_in_guest.y;
  uvec2 position_in_host_group = host_position & ((uvec2(1u) << group_blocks_log2) - 1u);
  uint group_width_bytes_log2 = group_blocks_log2.x + 2u;
  uint guest_address = TiledAddress2D(guest_group_origin, pitch_macro_tiles);
  uint scaled_group_address = guest_address * scale.x * scale.y;
  uint host_group_offset = (host_group_index << (group_width_bytes_log2 + group_blocks_log2.y)) |
                           (position_in_host_group.y << group_width_bytes_log2) |
                           (position_in_host_group.x << 2u);
  return scaled_group_address + host_group_offset;
}

uvec2 SourceCoordinate(uvec2 pixel, uint sample_index, uvec2 scale, out uint image_sample) {
  uint msaa = (pc.edram_info >> 10u) & 3u;
  uvec2 sample_shift = uvec2(uint(msaa >= 2u), uint(msaa >= 1u));
  uvec2 rt_sample = pixel << sample_shift;
  rt_sample += (uvec2(sample_index) >> uvec2(1u, 0u)) & 1u;
  uvec2 tile_size = uvec2(kEdramTileWidth, kEdramTileHeight) * scale;
  uvec2 tile_offset = rt_sample / tile_size;
  rt_sample -= tile_offset * tile_size;
  uint resolve_base = (pc.edram_info >> 13u) & 2047u;
  uint resolve_pitch = pc.edram_info & 1023u;
  uint absolute_tile = (resolve_base + tile_offset.y * resolve_pitch + tile_offset.x) & (kEdramTileCount - 1u);
  uint relative_tile = (absolute_tile - pc.source_base_tiles) & (kEdramTileCount - 1u);
  uint source_tile_y = relative_tile / pc.source_pitch_tiles;
  uint source_tile_x = relative_tile - source_tile_y * pc.source_pitch_tiles;
  uvec2 source_sample = uvec2(source_tile_x * tile_size.x, source_tile_y * tile_size.y) + rt_sample;
  if (msaa == 1u) {
    uint guest_sample = source_sample.y & 1u;
    image_sample = pc.source_msaa_2x_supported != 0u ? (guest_sample != 0u ? 0u : 1u)
                                                       : (guest_sample != 0u ? 3u : 0u);
    source_sample.y >>= 1u;
  } else {
    image_sample = 0u;
  }
  return source_sample;
}

uint PackUNorm8x4(vec4 value) {
  uvec4 c = uvec4(floor(value * 255.0 + 0.5));
  return c.x | (c.y << 8u) | (c.z << 16u) | (c.w << 24u);
}

uint LoadSourcePixel(uvec2 coordinate, uint image_sample, uvec2 image_size) {
  if (any(greaterThanEqual(coordinate, image_size))) {
    return 0u;
  }
#if DIRECT_SOURCE_MULTISAMPLED
  return PackUNorm8x4(texelFetch(source_image, ivec2(coordinate), int(image_sample)));
#else
  return PackUNorm8x4(texelFetch(source_image, ivec2(coordinate), 0));
#endif
}

uint SwapRedBlue(uint value) {
  return (value & 0xFF00FF00u) | ((value & 0x000000FFu) << 16u) | ((value >> 16u) & 0xFFu);
}

uvec4 EndianSwap32(uvec4 value, uint endian) {
  if (endian == 1u || endian == 2u) {
    value = ((value & 0x00FF00FFu) << 8u) | ((value & 0xFF00FF00u) >> 8u);
  }
  if (endian == 2u || endian == 3u) {
    value = (value << 16u) | (value >> 16u);
  }
  return value;
}

void StoreFour(uint address, uvec4 value, uint endian) {
  value = EndianSwap32(value, endian);
  uint word = address >> 2u;
  destination.data[word + 0u] = value.x;
  destination.data[word + 1u] = value.y;
  destination.data[word + 2u] = value.z;
  destination.data[word + 3u] = value.w;
}

void main() {
  uvec2 scale = (uvec2(pc.coordinate_info) >> uvec2(16u, 19u)) & 7u;
  uint width_div_8_scaled = ((pc.coordinate_info >> 5u) & 2047u) * scale.x;
  uvec2 invocation = gl_GlobalInvocationID.xy;
  if (invocation.x >= width_div_8_scaled) {
    return;
  }
  uvec2 pixel = invocation * uvec2(8u, 1u);
  uint fill_enabled = (pc.edram_info >> 29u) & 1u;
  uvec2 fill_source = fill_enabled != 0u ? (scale >> 1u) : uvec2(0u);
  uvec2 edram_offset = (((uvec2(pc.coordinate_info) >> uvec2(0u, 4u)) & uvec2(15u, 1u)) << 3u) * scale;
  uint sample_select = (pc.dest_coordinate_info >> 28u) & 7u;
  uint sample_index = sample_select <= 3u ? sample_select : (sample_select == 5u ? 2u : 0u);
  uint endian = pc.dest_info & 7u;
  bool swap_red_blue = (pc.dest_info & (1u << 24u)) != 0u;
  uint dest_offset_x = ((pc.dest_coordinate_info >> 20u) & 15u) * 8u * scale.x;
  uint dest_offset_y = ((pc.dest_coordinate_info >> 24u) & 15u) * 8u * scale.y;

#if DIRECT_SOURCE_MULTISAMPLED
  // Keep multisampled source addressing per pixel; GPU parity for the new
  // reuse path has only been established for non-multisampled sources.
  uint pixels[8];
  for (uint i = 0u; i < 8u; ++i) {
    uvec2 source_pixel = pixel + uvec2(i, 0u);
    source_pixel.y = max(source_pixel.y, fill_source.y);
    source_pixel += edram_offset;
    uint image_sample;
    uvec2 source_coordinate = SourceCoordinate(source_pixel, sample_index, scale, image_sample);
    uvec2 source_size = uvec2(textureSize(source_image));
    pixels[i] = LoadSourcePixel(source_coordinate, image_sample, source_size);
  }
#else
  // The supported 1x/2x-MSAA path has no horizontal sample expansion.
  // pixel.x and edram_offset.x are multiples of 8, as is 80 * scale.x,
  // so these eight pixels cannot cross an EDRAM tile boundary. Reuse its
  // address calculation, but retain each pixel's bounds check and sample fetch.
  uvec2 source_pixel = pixel;
  source_pixel.y = max(source_pixel.y, fill_source.y);
  source_pixel += edram_offset;
  uint image_sample;
  uvec2 source_coordinate = SourceCoordinate(source_pixel, sample_index, scale, image_sample);
  uvec2 source_size = uvec2(textureSize(source_image
#if DIRECT_SOURCE_MULTISAMPLED
#else
                                      , 0
#endif
                                      ));
  uint pixels[8];
  for (uint i = 0u; i < 8u; ++i) {
    pixels[i] = LoadSourcePixel(source_coordinate + uvec2(i, 0u), image_sample, source_size);
  }
#endif
  if (pixel.x == 0u && fill_source.x != 0u) {
    if (fill_source.x >= 3u) pixels[2] = pixels[3];
    if (fill_source.x >= 2u) pixels[1] = pixels[2];
    pixels[0] = pixels[1];
  }
  if (swap_red_blue) {
    for (uint i = 0u; i < 8u; ++i) pixels[i] = SwapRedBlue(pixels[i]);
  }

  uvec2 dest_position = pixel + uvec2(dest_offset_x, dest_offset_y);
#if DIRECT_SAMPLED_IMAGE_OUTPUT
  ivec2 output_size = imageSize(sampled_destination);
  for (uint i = 0u; i < 8u; ++i) {
    ivec2 p = ivec2(dest_position + uvec2(i, 0u));
    if (p.x < output_size.x && p.y < output_size.y) {
      uint v = pixels[i];
      imageStore(sampled_destination, p,
                 vec4(float(v & 255u), float((v >> 8u) & 255u),
                      float((v >> 16u) & 255u), float(v >> 24u)) / 255.0);
    }
  }
#endif
#if !DIRECT_SKIP_GUEST_BYTE_STORE
  uint dest_address = ScaledDestinationAddress(dest_position, scale);
  StoreFour(dest_address, uvec4(pixels[0], pixels[1], pixels[2], pixels[3]), endian);
  // A four-pixel sub-unit boundary is not necessarily contiguous in the
  // scaled tiled layout (at 2x, x=0 and x=4 begin at bytes 0 and 32).
  uint dest_address_right = ScaledDestinationAddress(dest_position + uvec2(4u, 0u), scale);
  StoreFour(dest_address_right, uvec4(pixels[4], pixels[5], pixels[6], pixels[7]), endian);
#endif
}
)GLSL";

}  // namespace rex::graphics::vulkan
