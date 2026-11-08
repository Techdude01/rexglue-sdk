#pragma once

#include <cmath>
#include <cstdint>

// Default-off direct storage-image texture load path (diagnostic / optimisation).
//
// When eligible, the texture load compute shader writes the final texels to the
// sampleable VkImage with imageStore and the scratch-buffer materialisation plus
// the vkCmdCopyBufferToImage are skipped. Everything else - guest resolve writes,
// invalidation, load order, residency - is unchanged.
//
// This header holds the pure decision logic and the normalized-store encoding so
// both can be unit-tested without a Vulkan device.

namespace rex::graphics::vulkan {

// Guest TextureFormat::k_8_8_8_8 (the first supported slice).
inline constexpr uint32_t kDirectStorageImageGuestFormat = 6;
// VK_FORMAT_R8G8B8A8_UNORM.
inline constexpr uint32_t kDirectStorageImageHostFormat = 37;
// TextureCache::kLoadShaderIndex32bpb.
inline constexpr uint32_t kDirectStorageImageLoadShader = 2;
inline constexpr uint32_t kDirectStorageImageDepthGuestFormat = 22; // k_24_8
inline constexpr uint32_t kDirectStorageImageDepthHostFormat = 100; // R32_SFLOAT
inline constexpr uint32_t kDirectStorageImageDepthLoadShader = 35; // DepthUnorm

enum class DirectStorageImageFallback : uint32_t {
  // Eligible - the direct path may be used.
  kNone = 0,
  kPathDisabled,
  kGuestFormat,
  kHostFormat,
  kLoadShader,
  kNotScaledResolve,
  kNotTiled,
  kSignedSeparate,
  kNot2D,
  kMipLevels,
  kLayers,
  kPackedMips,
  kDeviceUnsupported,
  kCount,
};

struct DirectStorageImageCriteria {
  bool enabled = false;
  bool scaled_resolve = false;
  bool tiled = false;
  bool signed_separate = false;
  bool packed_mips = false;
  bool is_2d = false;
  uint32_t guest_format = 0;
  uint32_t host_format = 0;
  uint32_t load_shader = 0;
  // Highest mip level present in the texture (0 = base only).
  uint32_t mip_max_level = 0;
  uint32_t depth_or_array_size = 1;
  // vkGetPhysicalDeviceImageFormatProperties with VK_IMAGE_USAGE_STORAGE_BIT for
  // the exact image type/format/tiling/usage of this texture.
  bool device_supported = false;
};

inline DirectStorageImageFallback EvaluateDirectStorageImage(
    const DirectStorageImageCriteria& criteria) {
  if (!criteria.enabled) {
    return DirectStorageImageFallback::kPathDisabled;
  }
  const bool depth = criteria.guest_format == kDirectStorageImageDepthGuestFormat;
  if (criteria.guest_format != kDirectStorageImageGuestFormat && !depth) {
    return DirectStorageImageFallback::kGuestFormat;
  }
  if (criteria.host_format != (depth ? kDirectStorageImageDepthHostFormat : kDirectStorageImageHostFormat)) {
    return DirectStorageImageFallback::kHostFormat;
  }
  if (criteria.load_shader != (depth ? kDirectStorageImageDepthLoadShader : kDirectStorageImageLoadShader)) {
    return DirectStorageImageFallback::kLoadShader;
  }
  if (!criteria.scaled_resolve) {
    return DirectStorageImageFallback::kNotScaledResolve;
  }
  if (!criteria.tiled) {
    return DirectStorageImageFallback::kNotTiled;
  }
  if (criteria.signed_separate) {
    return DirectStorageImageFallback::kSignedSeparate;
  }
  if (!criteria.is_2d) {
    return DirectStorageImageFallback::kNot2D;
  }
  if (criteria.mip_max_level != 0) {
    return DirectStorageImageFallback::kMipLevels;
  }
  if (criteria.depth_or_array_size != 1) {
    return DirectStorageImageFallback::kLayers;
  }
  if (criteria.packed_mips) {
    return DirectStorageImageFallback::kPackedMips;
  }
  if (!criteria.device_supported) {
    return DirectStorageImageFallback::kDeviceUnsupported;
  }
  return DirectStorageImageFallback::kNone;
}

inline const char* DirectStorageImageFallbackName(DirectStorageImageFallback reason) {
  switch (reason) {
    case DirectStorageImageFallback::kNone:
      return "none";
    case DirectStorageImageFallback::kPathDisabled:
      return "path_disabled";
    case DirectStorageImageFallback::kGuestFormat:
      return "guest_format";
    case DirectStorageImageFallback::kHostFormat:
      return "host_format";
    case DirectStorageImageFallback::kLoadShader:
      return "load_shader";
    case DirectStorageImageFallback::kNotScaledResolve:
      return "not_scaled_resolve";
    case DirectStorageImageFallback::kNotTiled:
      return "not_tiled";
    case DirectStorageImageFallback::kSignedSeparate:
      return "signed_separate";
    case DirectStorageImageFallback::kNot2D:
      return "not_2d";
    case DirectStorageImageFallback::kMipLevels:
      return "mip_levels";
    case DirectStorageImageFallback::kLayers:
      return "layers";
    case DirectStorageImageFallback::kPackedMips:
      return "packed_mips";
    case DirectStorageImageFallback::kDeviceUnsupported:
      return "device_unsupported";
    default:
      return "unknown";
  }
}

// The direct shader would store a normalized float per channel (see the GLSL
// `xe_direct_storage_image_unpack` helper: byte / 255.0). A normalized UNORM8
// attachment converts a stored float back with round-to-nearest, so this must
// return the original byte for every byte value for the direct path to be
// texel-identical to the buffer path.
inline uint8_t DirectStorageImageUnorm8(float value) {
  const float clamped = value < 0.0f ? 0.0f : (value > 1.0f ? 1.0f : value);
  const long rounded = std::lround(clamped * 255.0f);
  return static_cast<uint8_t>(rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded));
}

inline uint8_t DirectStorageImageEncodeByte(uint8_t byte) {
  return DirectStorageImageUnorm8(static_cast<float>(byte) * (1.0f / 255.0f));
}

// The direct path writes the image as a compute storage image and leaves it to
// be read by the guest shaders. These are the two sides of that transition; the
// pipeline stages of the sampling side depend on the guest pipeline stages, so
// only the access mask and layout are fixed here.
// Mirrored with static_asserts against the real Vulkan enums in texture_cache.cpp.
struct DirectStorageImageUsageMasks {
  uint32_t stage_mask;
  uint32_t access_mask;
  int32_t layout;
};

// VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
// VK_IMAGE_LAYOUT_GENERAL.
inline constexpr DirectStorageImageUsageMasks kDirectStorageImageWriteUsage = {
    0x00000800u, 0x00000040u, 1};
// Guest shader stages (dynamic), VK_ACCESS_SHADER_READ_BIT,
// VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL.
inline constexpr DirectStorageImageUsageMasks kDirectStorageImageSampledReadUsage = {
    0u, 0x00000020u, 5};

}  // namespace rex::graphics::vulkan
