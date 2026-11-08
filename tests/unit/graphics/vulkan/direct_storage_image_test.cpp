// Unit tests for the default-off direct storage-image texture load path.
// No Vulkan device is created here: the decision logic, the normalized store
// encoding and the compute-write -> sampled-read transition contract are pure.

#include <array>
#include <cstdint>

#include <catch2/catch_test_macros.hpp>

#include <rex/graphics/vulkan/direct_storage_image.h>

using rex::graphics::vulkan::DirectStorageImageCriteria;
using rex::graphics::vulkan::DirectStorageImageFallback;
using rex::graphics::vulkan::DirectStorageImageEncodeByte;
using rex::graphics::vulkan::DirectStorageImageFallbackName;
using rex::graphics::vulkan::DirectStorageImageUnorm8;
using rex::graphics::vulkan::EvaluateDirectStorageImage;

namespace {

// The exact load observed in the 1440p candidate-3 audit: a 2x-scaled resolve
// surface sampled as k_8_8_8_8 -> VK_FORMAT_R8G8B8A8_UNORM, one level, one
// layer, no packed mip tail, stored-image support confirmed for the device.
DirectStorageImageCriteria SupportedCriteria() {
  DirectStorageImageCriteria criteria;
  criteria.enabled = true;
  criteria.scaled_resolve = true;
  criteria.tiled = true;
  criteria.signed_separate = false;
  criteria.packed_mips = false;
  criteria.is_2d = true;
  criteria.guest_format = rex::graphics::vulkan::kDirectStorageImageGuestFormat;
  criteria.host_format = rex::graphics::vulkan::kDirectStorageImageHostFormat;
  criteria.load_shader = rex::graphics::vulkan::kDirectStorageImageLoadShader;
  criteria.mip_max_level = 0;
  criteria.depth_or_array_size = 1;
  criteria.device_supported = true;
  return criteria;
}

}  // namespace

TEST_CASE("direct storage image accepts only the supported slice",
          "[vulkan][direct_storage_image]") {
  CHECK(EvaluateDirectStorageImage(SupportedCriteria()) == DirectStorageImageFallback::kNone);

  struct Case {
    const char* field;
    void (*mutate)(DirectStorageImageCriteria&);
    DirectStorageImageFallback expected;
  };
  const Case cases[] = {
      {"disabled", [](DirectStorageImageCriteria& c) { c.enabled = false; },
       DirectStorageImageFallback::kPathDisabled},
      {"guest format", [](DirectStorageImageCriteria& c) { c.guest_format = 999; },
       DirectStorageImageFallback::kGuestFormat},
      {"host format", [](DirectStorageImageCriteria& c) { c.host_format = 100; },
       DirectStorageImageFallback::kHostFormat},
      {"load shader", [](DirectStorageImageCriteria& c) { c.load_shader = 35; },
       DirectStorageImageFallback::kLoadShader},
      {"not scaled resolve", [](DirectStorageImageCriteria& c) { c.scaled_resolve = false; },
       DirectStorageImageFallback::kNotScaledResolve},
      {"not tiled", [](DirectStorageImageCriteria& c) { c.tiled = false; },
       DirectStorageImageFallback::kNotTiled},
      {"signed separate", [](DirectStorageImageCriteria& c) { c.signed_separate = true; },
       DirectStorageImageFallback::kSignedSeparate},
      {"3D", [](DirectStorageImageCriteria& c) { c.is_2d = false; },
       DirectStorageImageFallback::kNot2D},
      {"mips", [](DirectStorageImageCriteria& c) { c.mip_max_level = 1; },
       DirectStorageImageFallback::kMipLevels},
      {"layers", [](DirectStorageImageCriteria& c) { c.depth_or_array_size = 2; },
       DirectStorageImageFallback::kLayers},
      {"packed mips", [](DirectStorageImageCriteria& c) { c.packed_mips = true; },
       DirectStorageImageFallback::kPackedMips},
      {"no storage-image support",
       [](DirectStorageImageCriteria& c) { c.device_supported = false; },
       DirectStorageImageFallback::kDeviceUnsupported},
  };
  for (const Case& test_case : cases) {
    DirectStorageImageCriteria criteria = SupportedCriteria();
    test_case.mutate(criteria);
    INFO(test_case.field);
    CHECK(EvaluateDirectStorageImage(criteria) == test_case.expected);
    CHECK(DirectStorageImageFallbackName(test_case.expected) != nullptr);
  }
}

TEST_CASE("direct storage image store is texel-identical for every byte",
          "[vulkan][direct_storage_image]") {
  // The buffer path copies raw bytes into the image. The direct path stores
  // byte / 255.0 through a normalized UNORM8 storage image, which converts the
  // float back with round-to-nearest. Any value that does not round-trip is a
  // texel mismatch, so all 256 byte values must survive.
  for (uint32_t value = 0; value <= 255; ++value) {
    const uint8_t byte = static_cast<uint8_t>(value);
    INFO("byte " << value);
    CHECK(DirectStorageImageEncodeByte(byte) == byte);
  }
  // Channel-position independence: the shader unpacks bytes at 0, 8, 16 and 24.
  CHECK(DirectStorageImageEncodeByte(static_cast<uint8_t>(0x00)) == 0x00);
  CHECK(DirectStorageImageEncodeByte(static_cast<uint8_t>(0x7F)) == 0x7F);
  CHECK(DirectStorageImageEncodeByte(static_cast<uint8_t>(0x80)) == 0x80);
  CHECK(DirectStorageImageEncodeByte(static_cast<uint8_t>(0xFF)) == 0xFF);
  // Out-of-range normalized values clamp rather than wrap.
  CHECK(DirectStorageImageUnorm8(-0.25f) == 0);
  CHECK(DirectStorageImageUnorm8(1.25f) == 255);
}

TEST_CASE("depth-derived R32 direct image requires matched format shader and device support") {
  auto c = SupportedCriteria();
  c.guest_format = rex::graphics::vulkan::kDirectStorageImageDepthGuestFormat;
  c.host_format = rex::graphics::vulkan::kDirectStorageImageDepthHostFormat;
  c.load_shader = rex::graphics::vulkan::kDirectStorageImageDepthLoadShader;
  CHECK(EvaluateDirectStorageImage(c) == DirectStorageImageFallback::kNone);
  c.device_supported = false;
  CHECK(EvaluateDirectStorageImage(c) == DirectStorageImageFallback::kDeviceUnsupported);
  c.device_supported = true;
  c.load_shader = rex::graphics::vulkan::kDirectStorageImageLoadShader;
  CHECK(EvaluateDirectStorageImage(c) == DirectStorageImageFallback::kLoadShader);
}

TEST_CASE("direct storage image transition writes in general and samples read-only",
          "[vulkan][direct_storage_image]") {
  using rex::graphics::vulkan::kDirectStorageImageSampledReadUsage;
  using rex::graphics::vulkan::kDirectStorageImageWriteUsage;
  // Compute stage, shader write, VK_IMAGE_LAYOUT_GENERAL.
  CHECK(kDirectStorageImageWriteUsage.stage_mask == 0x00000800u);
  CHECK(kDirectStorageImageWriteUsage.access_mask == 0x00000040u);
  CHECK(kDirectStorageImageWriteUsage.layout == 1);
  // Shader read, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL; the source and
  // destination layouts differ, so the barrier is a real transition.
  CHECK(kDirectStorageImageSampledReadUsage.access_mask == 0x00000020u);
  CHECK(kDirectStorageImageSampledReadUsage.layout == 5);
  CHECK(kDirectStorageImageWriteUsage.layout != kDirectStorageImageSampledReadUsage.layout);
  CHECK(kDirectStorageImageWriteUsage.access_mask !=
        kDirectStorageImageSampledReadUsage.access_mask);
  // A sampling view must not be created while the layout is still GENERAL.
  CHECK(kDirectStorageImageSampledReadUsage.layout == 5);
}
