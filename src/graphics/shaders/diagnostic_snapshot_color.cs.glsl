#version 450
#extension GL_EXT_samplerless_texture_functions : require

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(std430, set = 0, binding = 0) buffer SnapshotBuffer {
  uint words[];
} xe_snapshot;
layout(std430, set = 0, binding = 1) readonly buffer ConstantsBuffer {
  ivec2 origin;
  uvec2 extent;
  uint destination_word_offset;
  uint host_sample_0;
  uint host_sample_1;
  uint canary_word_index;
} xe_constants;

#ifdef XE_SNAPSHOT_SINGLE_SAMPLE
layout(set = 1, binding = 0) uniform utexture2D xe_color;
#else
layout(set = 1, binding = 0) uniform utexture2DMS xe_color;
#endif

void main() {
  uvec2 pixel = gl_GlobalInvocationID.xy;
  if (all(equal(pixel, uvec2(0u)))) xe_snapshot.words[xe_constants.canary_word_index] = 0xC0DEC0DEu;
  if (any(greaterThanEqual(pixel, xe_constants.extent))) {
    return;
  }
  ivec2 source_position = xe_constants.origin + ivec2(pixel);
  uint pixel_index = pixel.y * xe_constants.extent.x + pixel.x;
  uint samples[2] = uint[2](xe_constants.host_sample_0, xe_constants.host_sample_1);
  for (uint logical_sample = 0u; logical_sample < 2u; ++logical_sample) {
#ifdef XE_SNAPSHOT_SINGLE_SAMPLE
    uvec4 color = texelFetch(xe_color, source_position, 0);
#else
    uvec4 color = texelFetch(xe_color, source_position, int(samples[logical_sample]));
#endif
    uint destination = xe_constants.destination_word_offset +
                       (pixel_index * 2u + logical_sample) * 8u;
    xe_snapshot.words[destination + 0u] = color.r;
    xe_snapshot.words[destination + 1u] = color.g;
    xe_snapshot.words[destination + 2u] = color.b;
    xe_snapshot.words[destination + 3u] = color.a;
    xe_snapshot.words[destination + 6u] = samples[logical_sample];
    xe_snapshot.words[destination + 7u] = logical_sample;
  }
}
