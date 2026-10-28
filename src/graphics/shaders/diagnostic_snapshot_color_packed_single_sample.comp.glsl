#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
layout(std430, set = 0, binding = 0) buffer SnapshotBuffer { uint words[]; } xe_snapshot;
layout(std430, set = 0, binding = 1) readonly buffer ConstantsBuffer {
  ivec2 origin;
  uvec2 extent;
  uint destination_word_offset;
  uint host_sample_0;
  uint host_sample_1;
  uint canary_word_index;
} xe_constants;
layout(set = 1, binding = 0) uniform texture2D xe_color;

void main() {
  uvec2 pixel = gl_GlobalInvocationID.xy;
  if (all(equal(pixel, uvec2(0u)))) xe_snapshot.words[xe_constants.canary_word_index] = 0xC0DEC0DEu;
  if (any(greaterThanEqual(pixel, xe_constants.extent))) return;
  ivec2 source_position = xe_constants.origin + ivec2(pixel);
  uint pixel_index = pixel.y * xe_constants.extent.x + pixel.x;
  vec4 color = texelFetch(xe_color, source_position, 0);
  uint packed_color = packUnorm4x8(clamp(color, 0.0, 1.0));
  for (uint logical_sample = 0u; logical_sample < 2u; ++logical_sample) {
    uint destination = xe_constants.destination_word_offset + (pixel_index * 2u + logical_sample) * 2u;
    xe_snapshot.words[destination] = packed_color;
  }
}
