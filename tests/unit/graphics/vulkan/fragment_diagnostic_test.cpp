#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <rex/graphics/vulkan/fragment_diagnostic.h>
#include <rex/graphics/vulkan/render_target_snapshot_diagnostic.h>
#include <rex/graphics/vulkan/vertex_input_diagnostic.h>

using namespace rex::graphics::vulkan::diagnostics;

TEST_CASE("Counter validation distinguishes absent entry instrumentation from corrupt counts",
          "[vulkan][diagnostics]") {
  CHECK(FragmentDiagnosticCountersValid(100, 10, 90, 9));
  CHECK_FALSE(FragmentDiagnosticCountersValid(0, 0, 90, 9));
  CHECK(FragmentDiagnosticCountersValid(0, 0, 90, 9, false));
  CHECK_FALSE(FragmentDiagnosticCountersValid(0, 0, 90, 91, false));
  CHECK_FALSE(FragmentDiagnosticCountersValid(1, 0, 90, 9, false));
  CHECK_FALSE(FragmentDiagnosticCountersValid(0, 1, 90, 9, false));
}

namespace snapshot_shaders {
#include "../../../../src/graphics/shaders/vulkan_spirv/diagnostic_snapshot_color_cs.h"
#include "../../../../src/graphics/shaders/vulkan_spirv/diagnostic_snapshot_color_float_cs.h"
#include "../../../../src/graphics/shaders/vulkan_spirv/diagnostic_snapshot_depth_cs.h"
#include "../../../../src/graphics/shaders/vulkan_spirv/diagnostic_snapshot_color_single_sample_cs.h"
#include "../../../../src/graphics/shaders/vulkan_spirv/diagnostic_snapshot_color_float_single_sample_cs.h"
#include "../../../../src/graphics/shaders/vulkan_spirv/diagnostic_snapshot_depth_single_sample_cs.h"
#include "../../../../src/graphics/shaders/vulkan_spirv/diagnostic_snapshot_color_packed_cs.h"
#include "../../../../src/graphics/shaders/vulkan_spirv/diagnostic_snapshot_depth_packed_cs.h"
}  // namespace snapshot_shaders

// Check the actual compiled modules that the diagnostic binds. The runtime
// image-type mismatch is not detectable by validating either module alone.
template <size_t WordCount>
static void CheckSnapshotImageType(const uint32_t (&module)[WordCount], bool multisampled) {
  REQUIRE(WordCount >= 5);
  REQUIRE(module[0] == 0x07230203u);
  uint32_t image_count = 0;
  uint32_t fetch_count = 0;
  for (size_t offset = 5; offset < WordCount;) {
    const uint32_t words = module[offset] >> 16;
    const uint32_t opcode = module[offset] & 0xFFFFu;
    REQUIRE(words != 0);
    REQUIRE(offset + words <= WordCount);
    if (opcode == 25) {  // OpTypeImage: MS is operand 6.
      REQUIRE(words >= 9);
      CHECK(module[offset + 6] == uint32_t(multisampled));
      ++image_count;
    } else if (opcode == 95) {  // OpImageFetch: Image Operands is operand 5.
      REQUIRE(words >= 7);
      // Single-sample fetches specify Lod; multisample fetches specify Sample.
      CHECK(module[offset + 5] == (multisampled ? 0x40u : 0x2u));
      ++fetch_count;
    }
    offset += words;
  }
  CHECK(image_count != 0);
  CHECK(fetch_count != 0);
}

TEST_CASE("Attachment snapshot SPIR-V image types match the source sample count",
          "[vulkan][diagnostics]") {
  using namespace snapshot_shaders;
  CheckSnapshotImageType(diagnostic_snapshot_color_cs, true);
  CheckSnapshotImageType(diagnostic_snapshot_color_float_cs, true);
  CheckSnapshotImageType(diagnostic_snapshot_depth_cs, true);
  CheckSnapshotImageType(diagnostic_snapshot_color_single_sample_cs, false);
  CheckSnapshotImageType(diagnostic_snapshot_color_float_single_sample_cs, false);
  CheckSnapshotImageType(diagnostic_snapshot_depth_single_sample_cs, false);
  CheckSnapshotImageType(diagnostic_snapshot_color_packed_cs, true);
  CheckSnapshotImageType(diagnostic_snapshot_depth_packed_cs, true);
}

TEST_CASE("Render target snapshot requests are default-off, draw-keyed, and bounded",
          "[vulkan][diagnostics]") {
  RenderTargetSnapshotRequest defaults;
  CHECK_FALSE(defaults.enabled);
  CHECK_FALSE(defaults.valid());
  CHECK_FALSE(defaults.matches(8680, 271));

  const auto request = ParseRenderTargetSnapshotRequest(
      "mode=render-target-snapshot\nenabled=1\nframe=6000\ndraw=532\nx=598\ny=15\n"
      "width=17\nheight=19\ncolor=1\ndepth=1\nafter_each_draw=1\n");
  CHECK(request.valid());
  CHECK(ParseRenderTargetSnapshotMode(
      "frame=6000\nmode=render-target-snapshot\n"));
  CHECK_FALSE(ParseRenderTargetSnapshotMode("frame=6000\nmode=census\n"));
  CHECK(request.matches(6000, 532));
  CHECK_FALSE(request.matches(6001, 532));
  CHECK_FALSE(request.matches(6000, 533));
  const auto first_matching_draw = ParseRenderTargetSnapshotRequest(
      "mode=render-target-snapshot\nenabled=1\nframe=6000\ndraw=0\n"
      "x=598\ny=15\nwidth=17\nheight=19\ncolor=1\ndepth=1\n");
  CHECK(first_matching_draw.matches(6000, 1));
  CHECK(first_matching_draw.matches(6000, 466));
  const auto marker_request = ParseRenderTargetSnapshotRequest(
      "mode=render-target-snapshot\nenabled=1\nframe=6000\ndraw=0\n"
      "target=quad-partner\nx=584\ny=0\nwidth=96\nheight=96\n"
      "color=1\ndepth=1\nafter_each_draw=0\n");
  REQUIRE(marker_request.valid());
  CHECK(marker_request.target == RenderTargetSnapshotTarget::kQuadPartner);
  CHECK_FALSE(marker_request.after_each_draw);
  CHECK_FALSE(first_matching_draw.matches(6001, 466));
  CHECK(request.x == 598);
  CHECK(request.y == 15);
  CHECK(request.after_each_draw);
  CHECK(request.payload_bytes() == 17u * 19u * 2u * sizeof(RenderTargetSnapshotSample));
  CHECK((GetGuest2xHostSampleIndices(true) == std::array<uint32_t, 2>{0, 1}));
  CHECK((GetGuest2xHostSampleIndices(false) == std::array<uint32_t, 2>{0, 3}));
  CHECK(sizeof(RenderTargetSnapshotBoundary) == 136);

  const auto ben_compact = ParseRenderTargetSnapshotRequest(
      "mode=ben-occlusion-snapshot\nenabled=1\nframe=6000\ndraw=0\n"
      "target=ben-cuff\nx=0\ny=0\nwidth=1280\nheight=336\n"
      "color=1\ndepth=1\nafter_each_draw=1\n");
  REQUIRE(ben_compact.valid());
  CHECK(ben_compact.compact_ben);
  CHECK(ben_compact.payload_bytes() == 1280u * 336u * 2u * 8u);
  CHECK(ben_compact.payload_bytes() == 6881280u);
  CHECK(uint64_t(kBenOcclusionSnapshotMaxBoundaries) * ben_compact.payload_bytes() + 4u <=
        kBenOcclusionSnapshotMaxBytes);
  CHECK(kBenOcclusionSnapshotMaxBoundaries == 39);
  CHECK(ParseRenderTargetSnapshotMode("mode=ben-occlusion-snapshot\n"));
  CHECK(GetBenOcclusionCandidateKind(UINT64_C(0x046523BA1EB2717E),
                                     UINT64_C(0x2D4C010D3D75DEA5), 1836, 1280, 336) ==
        kBenOcclusionCandidateCuff);
  CHECK(GetBenOcclusionCandidateKind(UINT64_C(0x8A449D0796A656D7),
                                     UINT64_C(0x6E6F06115807B602), 144, 1280, 336) ==
        kBenOcclusionCandidateDial);
  CHECK(GetBenOcclusionCandidateKind(UINT64_C(0x7A0841365E75A823),
                                     UINT64_C(0x7ABDFEFBCADB4BF9), 60, 1280, 336) ==
        kBenOcclusionCandidateGlow);
  CHECK(GetBenOcclusionCandidateKind(UINT64_C(0x7A0841365E75A823),
                                     UINT64_C(0x7ABDFEFBCADB4BF9), 60, 1280, 720) == 0);
  CHECK_FALSE(ParseRenderTargetSnapshotRequest(
                  "mode=ben-occlusion-snapshot\nenabled=1\nframe=6000\ndraw=0\n"
                  "target=ben-cuff\nx=0\ny=0\nwidth=512\nheight=336\n"
                  "color=1\ndepth=1\n")
                  .valid());

  BenOcclusionSnapshotFileHeader ben_header{};
  ben_header.magic = kBenOcclusionSnapshotMagic;
  ben_header.version = kBenOcclusionSnapshotVersion;
  ben_header.frame = 6000;
  ben_header.width = 1280;
  ben_header.height = 336;
  ben_header.boundary_count = 1;
  ben_header.logical_samples_per_pixel = 2;
  ben_header.sample_record_size = 8;
  ben_header.color_encoding = 1;
  ben_header.depth_encoding = 1;
  ben_header.boundary_metadata_size = sizeof(BenOcclusionSnapshotBoundaryMetadata);
  const size_t ben_payload_size = size_t(ben_header.width) * ben_header.height * 2 * 8;
  std::vector<uint8_t> ben_file(sizeof(ben_header) + sizeof(RenderTargetSnapshotBoundary) +
                                sizeof(BenOcclusionSnapshotBoundaryMetadata) + ben_payload_size);
  std::memcpy(ben_file.data(), &ben_header, sizeof(ben_header));
  RenderTargetSnapshotBoundary ben_boundary{};
  ben_boundary.draw_ordinal = 848;
  ben_boundary.point = uint32_t(RenderTargetSnapshotPoint::kBeforeHeadDraw);
  ben_boundary.capture_flags = kBenOcclusionCandidateCuff << 16;
  std::memcpy(ben_file.data() + sizeof(ben_header), &ben_boundary, sizeof(ben_boundary));
  BenOcclusionSnapshotBoundaryMetadata ben_metadata{};
  ben_metadata.viewport_width = 1280;
  ben_metadata.viewport_height = 336;
  ben_metadata.index_count = 1836;
  ben_metadata.candidate_kind = kBenOcclusionCandidateCuff;
  std::memcpy(ben_file.data() + sizeof(ben_header) + sizeof(ben_boundary), &ben_metadata,
              sizeof(ben_metadata));
  BenOcclusionSnapshotView ben_view;
  REQUIRE(ParseBenOcclusionSnapshot(ben_file, ben_view));
  CHECK(ben_view.sample_records.size() == ben_payload_size);
  RenderTargetSnapshotBoundary parsed_ben_boundary{};
  REQUIRE(ReadBenOcclusionSnapshotBoundary(ben_view, 0, parsed_ben_boundary));
  CHECK(parsed_ben_boundary.draw_ordinal == 848);
  BenOcclusionSnapshotBoundaryMetadata parsed_ben_metadata{};
  REQUIRE(ReadBenOcclusionSnapshotBoundaryMetadata(ben_view, 0, parsed_ben_metadata));
  CHECK(parsed_ben_metadata.index_count == 1836);
  CHECK_FALSE(ParseBenOcclusionSnapshot(std::span<const uint8_t>(ben_file).first(64), ben_view));

  CHECK_FALSE(ParseRenderTargetSnapshotRequest(
                   "mode=render-target-snapshot\nenabled=1\nframe=6000\ndraw=532\n"
                   "x=0\ny=0\nwidth=513\nheight=1\n"
                   "color=1\ndepth=1\n")
                   .valid());
  CHECK_FALSE(ParseRenderTargetSnapshotRequest(
                   "mode=render-target-snapshot\nenabled=1\nframe=6000\ndraw=532\n"
                   "x=0\ny=0\nwidth=0\nheight=1\n"
                   "color=1\ndepth=1\n")
                   .valid());
  CHECK_FALSE(ParseRenderTargetSnapshotRequest(
                   "mode=render-target-snapshot\nenabled=1\nframe=6000\ndraw=532\n"
                   "x=0\ny=0\nwidth=512\nheight=512\n"
                   "color=1\ndepth=1\n")
                   .valid());
  CHECK_FALSE(ParseRenderTargetSnapshotRequest(
                   "mode=render-target-snapshot\nenabled=1\nframe=6000\ndraw=532\n"
                   "x=0\ny=0\nwidth=10\nheight=10\n"
                   "color=0\ndepth=0\n")
                   .valid());
  CHECK_FALSE(ParseRenderTargetSnapshotRequest(
                   "mode=render-target-snapshot\nenabled=1\nframe=6000\n"
                   "draw=4294967296\nx=0\ny=0\nwidth=1\nheight=1\n"
                   "color=1\ndepth=1\n")
                   .valid());
  CHECK_FALSE(ParseRenderTargetSnapshotRequest(
                   "mode=render-target-snapshot\nenabled=1\nframe=6000\ndraw=532\n"
                   "x=4294967295\ny=0\nwidth=2\n"
                   "height=1\ncolor=1\ndepth=1\n")
                   .valid());
  CHECK_FALSE(ParseRenderTargetSnapshotRequest(
                   "mode=render-target-snapshot\nenabled=1\nframe=18446744073709551616\n"
                   "draw=532\nx=0\ny=0\nwidth=1\n"
                   "height=1\ncolor=1\ndepth=1\n")
                   .valid());
  CHECK_FALSE(ParseRenderTargetSnapshotRequest(
                   "mode=render-target-snapshot\nenabled=0\nframe=6000\ndraw=532\n"
                   "x=0\ny=0\nwidth=1\nheight=1\n"
                   "color=1\ndepth=1\n")
                   .valid());
}

TEST_CASE("Vertex input dump is gated to the frame-6000 lower-strip head draw",
          "[vulkan][diagnostics]") {
  CHECK(GetPixelDiagnosticStorageVersion(true, true, true, false, true, true) == 10);
  CHECK(GetPixelDiagnosticStorageVersion(true, true, false, false, true, true) == 0);
  CHECK(GetPixelDiagnosticStorageVersion(true, false, false, false, true, true) == 10);
  CHECK(GetPixelDiagnosticStorageVersion(true, true, true, false, false, false) == 0);
  CHECK(GetPixelDiagnosticStorageVersion(false, true, true, false, true, true) == 0);
  CHECK(GetPixelDiagnosticStorageVersion(true, true, true, true, false, false) == 9);
  CHECK(ParseVertexInputDiagnosticMode("frame=8680\nmode=vertex-input-dump\n"));
  CHECK_FALSE(ParseVertexInputDiagnosticMode("frame=8680\nmode=census\n"));
  CHECK(IsVertexInputDiagnosticHeadDraw(
      kVertexInputDiagnosticSnapshotFrame, kVertexInputDiagnosticTargetDrawOrdinal,
      kVertexInputDiagnosticTargetScissorHeight,
      kVertexInputDiagnosticVertexHash, kVertexInputDiagnosticPixelHash));
  CHECK(IsVertexInputDiagnosticHeadDraw(
      kVertexInputDiagnosticSnapshotFrame, 400, kVertexInputDiagnosticTargetScissorHeight,
      kVertexInputDiagnosticVertexHash, kVertexInputDiagnosticPixelHash));
  CHECK(IsVertexInputDiagnosticHeadDraw(
      kVertexInputDiagnosticSnapshotFrame, 620, kVertexInputDiagnosticTargetScissorHeight,
      kVertexInputDiagnosticVertexHash, kVertexInputDiagnosticPixelHash));
  CHECK_FALSE(IsVertexInputDiagnosticHeadDraw(
      kVertexInputDiagnosticSnapshotFrame - 1, 400,
      kVertexInputDiagnosticTargetScissorHeight, kVertexInputDiagnosticVertexHash,
      kVertexInputDiagnosticPixelHash));
  CHECK_FALSE(IsVertexInputDiagnosticHeadDraw(
      kVertexInputDiagnosticSnapshotFrame, 400, 384,
      kVertexInputDiagnosticVertexHash, kVertexInputDiagnosticPixelHash));
  CHECK_FALSE(IsVertexInputDiagnosticHeadDraw(
      kVertexInputDiagnosticSnapshotFrame, 400, kVertexInputDiagnosticTargetScissorHeight,
      kVertexInputDiagnosticVertexHash, 0));
  CHECK(kVertexInputDiagnosticMaxBytes == 16 * 1024 * 1024);
  CHECK(IsVertexInputDiagnosticRangeValid(0x1000, 1));
  CHECK(IsVertexInputDiagnosticRangeValid(0x1000,
                                          kVertexInputDiagnosticMaxBytes));
  CHECK_FALSE(IsVertexInputDiagnosticRangeValid(0x1000, 0));
  CHECK_FALSE(IsVertexInputDiagnosticRangeValid(0x1000,
                                               uint64_t(kVertexInputDiagnosticMaxBytes) + 1));
  CHECK_FALSE(IsVertexInputDiagnosticRangeValid(0xFFFFFFF0, 32));
  CHECK(IsVertexOutputRecordInBounds(0, 32768));
  CHECK(IsVertexOutputRecordInBounds(32767, 32768));
  CHECK_FALSE(IsVertexOutputRecordInBounds(32768, 32768));
  CHECK_FALSE(IsVertexOutputRecordInBounds(0, 0));
  CHECK(kVertexOutputDiagnosticStageMarker == 2);
}

TEST_CASE("Fragment diagnostic capacity accommodates the observed local nose workload",
          "[vulkan][diagnostics]") {
  // The bounded frame-8680 local ROI capture reached 8,368 eligible tf0/tf2/tf14
  // triples before overflowing the former 4,096-record buffer.
  CHECK(kFragmentDiagnosticMaxRecords >= 3 * 8368);
}

TEST_CASE("Fragment diagnostic descriptor output is gated, exact-key deduplicated and bounded",
          "[vulkan][diagnostics]") {
  FragmentDiagnosticDescriptorLog log;
  FragmentDiagnosticDescriptorLog::Key key{14, 0xDE20, 18, 1, false, true};
  CHECK_FALSE(log.ShouldLog(false, key));
  CHECK(log.ShouldLog(true, key));
  for (uint32_t i = 0; i < 10000; ++i)
    CHECK_FALSE(log.ShouldLog(true, key));
  auto signed_key = key;
  signed_key.is_signed = true;
  CHECK(log.ShouldLog(true, signed_key));
  auto format_key = key;
  ++format_key.format;
  CHECK(log.ShouldLog(true, format_key));
  auto dimension_key = key;
  ++dimension_key.dimension;
  CHECK(log.ShouldLog(true, dimension_key));
  for (uint32_t i = 4; i < FragmentDiagnosticDescriptorLog::kMaxEntries; ++i) {
    key.base_page = i;
    CHECK(log.ShouldLog(true, key));
  }
  ++key.fetch;
  for (uint32_t i = 0; i < 10000; ++i) {
    key.base_page = i;
    CHECK_FALSE(log.ShouldLog(true, key));
  }
}

TEST_CASE("Fragment diagnostic gate content selects a bounded frame or the historical default",
          "[vulkan][diagnostics]") {
  CHECK(ParseFragmentDiagnosticFrame("enabled") == 8680);
  CHECK(ParseFragmentDiagnosticFrame("") == 8680);
  CHECK(ParseFragmentDiagnosticFrame("enabled\nframe=600\n") == 600);
  CHECK(ParseFragmentDiagnosticFrame("  frame=1 \r\n") == 1);
  for (auto content : {"frame=", "frame=-1", "frame=0", "frame=abc", "frame=600junk",
                       "frame=18446744073709551615", "frame=18446744073709551616"}) {
    CHECK(ParseFragmentDiagnosticFrame(content) == 8680);
  }
}

TEST_CASE("Fragment diagnostic census mode and nose ROI are explicit and bounded",
          "[vulkan][diagnostics]") {
  CHECK(ParseFragmentDiagnosticCensusMode("frame=8680\nmode=census\n"));
  CHECK(ParseFragmentDiagnosticCensusMode(" mode=census \r\n"));
  CHECK_FALSE(ParseFragmentDiagnosticCensusMode("frame=8680\nmode=triples\n"));
  CHECK_FALSE(ParseFragmentDiagnosticCensusMode("frame=8680\n"));
  CHECK(ParseFragmentDiagnosticOverwriteCensusMode("frame=8680\nmode=overwrite-census\n"));
  CHECK_FALSE(ParseFragmentDiagnosticOverwriteCensusMode("mode=census\n"));
  CHECK(ParseFragmentDiagnosticLocalOverwriteCensusMode(
      "frame=8680\nmode=local-overwrite-census\n"));
  CHECK_FALSE(ParseFragmentDiagnosticLocalOverwriteCensusMode("mode=overwrite-census\n"));
  CHECK(ParseFragmentDiagnosticLocalCandidateTriplesMode(
      "frame=8680\nmode=local-candidate-triples\n"));
  CHECK(ParseFragmentDiagnosticLocalCandidateOutputMode(
      "frame=8680\nmode=local-candidate-output\n"));
  CHECK_FALSE(ParseFragmentDiagnosticLocalCandidateOutputMode("mode=census\n"));
  CHECK_FALSE(ParseFragmentDiagnosticLocalCandidateTriplesMode("mode=census\n"));
  CHECK(
      IsFragmentDiagnosticNoseCoordinate(kFragmentDiagnosticNoseMinX, kFragmentDiagnosticNoseMinY));
  CHECK(
      IsFragmentDiagnosticNoseCoordinate(kFragmentDiagnosticNoseMaxX, kFragmentDiagnosticNoseMaxY));
  CHECK_FALSE(IsFragmentDiagnosticNoseCoordinate(kFragmentDiagnosticNoseMinX - 1,
                                                 kFragmentDiagnosticNoseMinY));
  CHECK_FALSE(IsFragmentDiagnosticNoseCoordinate(kFragmentDiagnosticNoseMaxX + 1,
                                                 kFragmentDiagnosticNoseMaxY));
  CHECK_FALSE(IsFragmentDiagnosticNoseCoordinate(kFragmentDiagnosticNoseMinX,
                                                 kFragmentDiagnosticNoseMinY - 1));
  CHECK_FALSE(IsFragmentDiagnosticNoseCoordinate(kFragmentDiagnosticNoseMaxX,
                                                 kFragmentDiagnosticNoseMaxY + 1));
  CHECK(IsFragmentDiagnosticLocalCandidateCoordinate(kFragmentDiagnosticLocalCandidateMinX,
                                                      kFragmentDiagnosticLocalCandidateMinY));
  CHECK(IsFragmentDiagnosticLocalCandidateCoordinate(kFragmentDiagnosticLocalCandidateMaxX,
                                                      kFragmentDiagnosticLocalCandidateMaxY));
  CHECK(kFragmentDiagnosticLowerStripGuestRows == 384);
  CHECK(kFragmentDiagnosticCaptureScaleY == 1);
  CHECK(kFragmentDiagnosticLowerStripOutputOffsetY ==
        kFragmentDiagnosticLowerStripGuestRows * kFragmentDiagnosticCaptureScaleY);
  CHECK(kFragmentDiagnosticLowerStripOutputOffsetY == 384);
  CHECK(kFragmentDiagnosticLocalCandidateMinX == kFragmentDiagnosticNoseMinX);
  // The existing capture includes the right derivative partner.
  CHECK(kFragmentDiagnosticLocalCandidateMaxX == kFragmentDiagnosticNoseMaxX + 1);
  CHECK(kFragmentDiagnosticLocalCandidateMinY + kFragmentDiagnosticLowerStripOutputOffsetY ==
        kFragmentDiagnosticNoseMinY);
  CHECK(kFragmentDiagnosticLocalCandidateMaxY + kFragmentDiagnosticLowerStripOutputOffsetY ==
        kFragmentDiagnosticNoseMaxY);
  CHECK_FALSE(IsFragmentDiagnosticLocalCandidateCoordinate(kFragmentDiagnosticLocalCandidateMinX,
                                                           kFragmentDiagnosticLocalCandidateMinY - 1));
  CHECK(kFragmentDiagnosticMaxCensusDraws == 4096);
  CHECK(kFragmentDiagnosticMaxDrawStateLogEntries == 4096);
  FragmentDiagnosticDrawLog comprehensive_log(8680, true);
  for (uint32_t draw = 0; draw < kFragmentDiagnosticMaxDrawStateLogEntries - 1; ++draw) {
    CHECK(comprehensive_log.Observe(8680, 0));
  }
  CHECK_FALSE(comprehensive_log.Observe(8680, 0));
  CHECK(comprehensive_log.Observe(8680, kFragmentDiagnosticPixelHash));
  CHECK_FALSE(comprehensive_log.Observe(8680, 0));
}

TEST_CASE("Fragment diagnostic tf0 mip filter gate is explicit and head-fetch scoped",
          "[vulkan][diagnostics]") {
  CHECK(ParseFragmentDiagnosticTf0MipFilterMode("frame=6000\ntf0_mip=point\n") ==
        FragmentDiagnosticTf0MipFilterMode::kPoint);
  CHECK(ParseFragmentDiagnosticTf0MipFilterMode(" tf0_mip=linear \r\n") ==
        FragmentDiagnosticTf0MipFilterMode::kLinear);
  CHECK(ParseFragmentDiagnosticTf0MipFilterMode("frame=6000\n") ==
        FragmentDiagnosticTf0MipFilterMode::kDefault);
  CHECK(ParseFragmentDiagnosticTf0MipFilterMode("tf0_mip=nearest\n") ==
        FragmentDiagnosticTf0MipFilterMode::kDefault);
  CHECK(IsFragmentDiagnosticTf0MipFilterTarget(FragmentDiagnosticTf0MipFilterMode::kLinear, 6000,
                                               kFragmentDiagnosticPixelHash, 0, 6000));
  CHECK_FALSE(IsFragmentDiagnosticTf0MipFilterTarget(FragmentDiagnosticTf0MipFilterMode::kDefault,
                                                     6000, kFragmentDiagnosticPixelHash, 0, 6000));
  CHECK_FALSE(IsFragmentDiagnosticTf0MipFilterTarget(FragmentDiagnosticTf0MipFilterMode::kLinear,
                                                     6001, kFragmentDiagnosticPixelHash, 0, 6000));
  CHECK_FALSE(IsFragmentDiagnosticTf0MipFilterTarget(FragmentDiagnosticTf0MipFilterMode::kLinear,
                                                     6000, kFragmentDiagnosticPixelHash, 14, 6000));
  CHECK(std::string_view(FragmentDiagnosticTf0MipFilterModeName(
            FragmentDiagnosticTf0MipFilterMode::kPoint)) == "tf0-mip-point");
  CHECK(std::string_view(FragmentDiagnosticTf0MipFilterModeName(
            FragmentDiagnosticTf0MipFilterMode::kLinear)) == "tf0-mip-linear");
}

TEST_CASE("comprehensive draw log covers the frame and reports its cap", "[vulkan][diagnostics]") {
  using DrawLog = FragmentDiagnosticDrawLog;
  DrawLog complete_log(8680, true);
  uint32_t logged = 0;
  for (uint32_t draw = 0; draw < 200; ++draw) {
    if (complete_log.Observe(8680, draw == 100 ? kFragmentDiagnosticPixelHash : 0))
      ++logged;
  }
  CHECK(logged == 200);
  DrawLog::FrameSummary summary{};
  CHECK_FALSE(complete_log.Observe(8681, 0, &summary));
  CHECK(summary.available);
  CHECK(summary.frame == 8680);
  CHECK(summary.observed_draws == 200);
  CHECK(summary.logged_entries == 200);
  CHECK_FALSE(summary.truncated);

  DrawLog capped_log(8680, true);
  for (uint32_t draw = 0; draw < kFragmentDiagnosticMaxDrawStateLogEntries - 1; ++draw)
    CHECK(capped_log.Observe(8680, 0));
  CHECK(capped_log.Observe(8680, kFragmentDiagnosticPixelHash));
  CHECK_FALSE(capped_log.Observe(8680, 0));
  CHECK_FALSE(capped_log.Observe(8681, 0, &summary));
  CHECK(summary.available);
  CHECK(summary.observed_draws == kFragmentDiagnosticMaxDrawStateLogEntries + 1);
  CHECK(summary.logged_entries == kFragmentDiagnosticMaxDrawStateLogEntries);
  CHECK(summary.truncated);
}

TEST_CASE("Fragment diagnostic sampler override is opt-in and targets only PS fetch 14",
          "[vulkan][diagnostics]") {
  using Tf14Mode = FragmentDiagnosticTf14FilterMode;
  CHECK(ParseFragmentDiagnosticTf14FilterMode("frame=8680\nmode=census\nsampler=tf14-point\n") ==
        Tf14Mode::kPoint);
  CHECK(ParseFragmentDiagnosticTf14FilterMode(" sampler=tf14-linear \r\n") == Tf14Mode::kLinear);
  CHECK(ParseFragmentDiagnosticTf14FilterMode("sampler=tf14-pointish\n") == Tf14Mode::kDefault);
  CHECK(ParseFragmentDiagnosticTf14FilterMode("mode=census\n") == Tf14Mode::kDefault);
  CHECK(FragmentDiagnosticTf14FilterModeName(Tf14Mode::kLinear) == "tf14-linear");

  CHECK(IsFragmentDiagnosticTf14FilterTarget(
      Tf14Mode::kLinear, 8680, kFragmentDiagnosticPixelHash, 14, 8680));
  CHECK_FALSE(IsFragmentDiagnosticTf14FilterTarget(
      Tf14Mode::kDefault, 8680, kFragmentDiagnosticPixelHash, 14, 8680));
  CHECK_FALSE(IsFragmentDiagnosticTf14FilterTarget(
      Tf14Mode::kLinear, 8681, kFragmentDiagnosticPixelHash, 14, 8680));
  CHECK_FALSE(IsFragmentDiagnosticTf14FilterTarget(
      Tf14Mode::kLinear, 8680, 0, 14, 8680));
  CHECK_FALSE(IsFragmentDiagnosticTf14FilterTarget(
      Tf14Mode::kLinear, 8680, kFragmentDiagnosticPixelHash, 13, 8680));
}

TEST_CASE("Fragment diagnostic selection and near misses use the same configured frame and PS",
          "[vulkan][diagnostics]") {
  for (uint64_t target : {uint64_t(1), uint64_t(600), uint64_t(8680)}) {
    CHECK(IsFragmentDiagnosticTarget(target, kFragmentDiagnosticPixelHash, target));
    CHECK_FALSE(IsFragmentDiagnosticTarget(target - 1, kFragmentDiagnosticPixelHash, target));
    CHECK_FALSE(IsFragmentDiagnosticTarget(target + 1, kFragmentDiagnosticPixelHash, target));
    CHECK_FALSE(IsFragmentDiagnosticTarget(target, 0, target));
    FragmentDiagnosticDrawLog log(target);
    for (uint64_t frame = target - 1; frame <= target + 2; ++frame) {
      for (uint32_t i = 1; i <= 20; ++i) {
        CHECK(log.Observe(frame, 0) == (i <= 8));
        CHECK(log.ordinal() == i);
      }
      // An expected PS after many unrelated draws is still reported. Observation
      // does not consume the caller's first-matching-draw capture latch.
      CHECK(log.Observe(frame, kFragmentDiagnosticPixelHash));
      CHECK_FALSE(log.Observe(frame, kFragmentDiagnosticPixelHash));
      CHECK(IsFragmentDiagnosticTarget(frame, kFragmentDiagnosticPixelHash, target) ==
            (frame == target));
    }
    CHECK_FALSE(log.Observe(target + 3, kFragmentDiagnosticPixelHash));
  }
}

TEST_CASE("Fragment diagnostic status survives application log removal and reports IO failure",
          "[vulkan][diagnostics]") {
  const auto dir = std::filesystem::temp_directory_path() /
                   ("rex-fragment-diagnostic-" +
                    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  const auto path = dir / "status.log";
  REQUIRE(AppendFragmentDiagnosticStatus(path, "Fragment diagnostic gate: abi=5 capacity=4096"));
  {
    std::ofstream app_log(dir / "application.log");
    app_log << "rotated away";
  }
  std::filesystem::remove(dir / "application.log");
  REQUIRE(AppendFragmentDiagnosticStatus(path,
                                         "Fragment diagnostic capture: frame=8680 status=failed"));
  std::ifstream status(path);
  const std::string contents((std::istreambuf_iterator<char>(status)), {});
  CHECK(contents ==
        "Fragment diagnostic gate: abi=5 capacity=4096\n"
        "Fragment diagnostic capture: frame=8680 status=failed\n");
  CHECK_FALSE(AppendFragmentDiagnosticStatus(path / "not-a-directory.log", "unwritable"));
  status.close();
  std::filesystem::remove_all(dir);
}
