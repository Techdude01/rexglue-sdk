#include <catch2/catch_test_macros.hpp>
#include <rex/graphics/vulkan/gpu_work_labels.h>
#include <limits>
#include <string_view>
using namespace rex::graphics::vulkan;

TEST_CASE("GPU identities are opt-in and limited to a bounded non-overflowing window") {
  REQUIRE_FALSE(IsGpuWorkLabelFrame(false,4000,4000,64));
  REQUIRE_FALSE(IsGpuWorkLabelFrame(true,3999,4000,64));
  REQUIRE(IsGpuWorkLabelFrame(true,4000,4000,64));
  REQUIRE(IsGpuWorkLabelFrame(true,4063,4000,64));
  REQUIRE_FALSE(IsGpuWorkLabelFrame(true,4064,4000,64));
  REQUIRE_FALSE(IsGpuWorkLabelFrame(true,4000,4000,0));
  REQUIRE(IsGpuWorkLabelFrame(true,4000,4000,kMaxGpuWorkLabelFrames));
  REQUIRE(IsGpuWorkLabelFrame(true,4000+kMaxGpuWorkLabelFrames-1,4000,kMaxGpuWorkLabelFrames));
  REQUIRE_FALSE(IsGpuWorkLabelFrame(true,4000+kMaxGpuWorkLabelFrames,4000,kMaxGpuWorkLabelFrames));
  REQUIRE_FALSE(IsGpuWorkLabelFrame(true,4000,4000,kMaxGpuWorkLabelFrames+1));
  REQUIRE_FALSE(IsGpuWorkLabelFrame(true,1,0,64));
  constexpr auto max = std::numeric_limits<uint64_t>::max();
  REQUIRE(IsGpuWorkLabelFrame(true,max,max-3,4));
  REQUIRE_FALSE(IsGpuWorkLabelFrame(true,1,max-3,4));
}
TEST_CASE("GPU identities preserve full pipeline handles, stream and dispatch dimensions") {
  auto a = FormatGpuWorkLabel(4000,7,false,12,"resolve",0xABCDEF0123456789ull,8,16,2);
  REQUIRE(std::string_view(a.data()) == "CRGPU f=4000 sub=7 stream=main op=12 kind=resolve pipe=ABCDEF0123456789 x=8 y=16 z=2");
  auto b = FormatGpuWorkLabel(4000,7,true,12,"resolve",0xABCDEF0123456789ull,8,16,2);
  REQUIRE(std::string_view(a.data()) != std::string_view(b.data()));
  constexpr auto max = std::numeric_limits<uint64_t>::max();
  auto c = FormatGpuWorkLabel(max,max,true,max,"texture_guest",max,UINT32_MAX,UINT32_MAX,UINT32_MAX);
  REQUIRE(std::string_view(c.data()).ends_with("z=4294967295"));
  REQUIRE(std::string_view(c.data()).size() < c.size()-1);
}

TEST_CASE("Resolve work metadata shares the deferred dispatch operation identity") {
  ResolveWorkMetadata resolve;
  resolve.valid = true;
  resolve.is_depth = true;
  resolve.path = 2;
  resolve.shader = 6;
  resolve.msaa = 1;
  resolve.sample_select = 4;
  resolve.source_format = 3;
  resolve.dest_format = 22;
  resolve.dest_base = 0x0F018000;
  resolve.dest_start = 0x0F018000;
  resolve.dest_length = 0x00100000;
  std::snprintf(resolve.fallback_reason, sizeof(resolve.fallback_reason), "%s",
                "shader_scale_or_fused_disabled");
  auto label = FormatGpuWorkLabel(4000, 17, false, 23, "ignored", 0x1234, 40, 180, 1,
                                  &resolve);
  const std::string_view formatted(label.data());
  REQUIRE(formatted.starts_with("CRGPU f=4000 sub=17 stream=main op=23 kind=resolve_copy"));
  REQUIRE(formatted.find("path=fallback class=depth shader=6 msaa=1 select=4") !=
          std::string_view::npos);
  REQUIRE(formatted.find("srcfmt=3 dstfmt=22 base=0F018000 start=0F018000 len=00100000") !=
          std::string_view::npos);
  REQUIRE(formatted.find("reason=shader_scale_or_fused_disabled") != std::string_view::npos);
  REQUIRE(formatted.size() < label.size() - 1);
}
