#include <catch2/catch_test_macros.hpp>
#include <array>
#include <cstdint>

namespace {
struct Coordinate {
  uint32_t x, y, sample;
  bool operator==(const Coordinate&) const = default;
};
// Reference equations from SourceCoordinate in direct_host_resolve_shader.h.
Coordinate SourceCoordinate(uint32_t x, uint32_t y, uint32_t sample,
                            uint32_t sx, uint32_t sy, bool msaa2,
                            bool native2, uint32_t resolve_base,
                            uint32_t resolve_pitch, uint32_t source_base,
                            uint32_t source_pitch) {
  uint32_t rx = x;
  uint32_t ry = (y << uint32_t(msaa2)) + (sample & 1);
  uint32_t tw = 80 * sx, th = 16 * sy;
  uint32_t tile = (resolve_base + (ry / th) * resolve_pitch + rx / tw) & 2047;
  uint32_t relative = (tile - source_base) & 2047;
  rx = (relative % source_pitch) * tw + rx % tw;
  ry = (relative / source_pitch) * th + ry % th;
  uint32_t image_sample = 0;
  if (msaa2) {
    image_sample = native2 ? ((ry & 1) ? 0 : 1) : ((ry & 1) ? 3 : 0);
    ry >>= 1;
  }
  return {rx, ry, image_sample};
}
}

TEST_CASE("Fused resolve eight-pixel chunks preserve source coordinates and MSAA samples") {
  for (auto scale : {std::array<uint32_t,2>{1,1}, {2,2}, {3,2}, {7,7}}) {
    for (bool msaa2 : {false,true}) for (bool native2 : {false,true}) {
      for (uint32_t source_pitch : {1u,7u,80u,1023u}) {
        for (uint32_t base : {0u,1u,2040u,2047u}) {
          for (uint32_t y : {0u,1u,7u,15u,31u,255u,1023u}) {
            for (uint32_t sample = 0; sample <= uint32_t(msaa2); ++sample) {
              for (uint32_t x = 0; x < 3*80*scale[0]; x += 8) {
                auto first = SourceCoordinate(x,y,sample,scale[0],scale[1],msaa2,
                                               native2,base,7,2047-base,source_pitch);
                for (uint32_t i=0;i<8;++i) {
                  auto reference=SourceCoordinate(x+i,y,sample,scale[0],scale[1],msaa2,
                                                  native2,base,7,2047-base,source_pitch);
                  REQUIRE((reference == Coordinate{first.x+i,first.y,first.sample}));
                }
              }
            }
          }
        }
      }
    }
  }
}
TEST_CASE("Unaligned source chunks cannot use the fused-row optimization") {
  auto a=SourceCoordinate(79,0,0,1,1,false,false,0,7,0,3);
  auto b=SourceCoordinate(80,0,0,1,1,false,false,0,7,0,3);
  // With a differing source pitch, tile-boundary transitions are not a general
  // contiguous row rule. The optimization requires the caller's 8-alignment.
  auto c=SourceCoordinate(239,0,0,1,1,false,false,0,7,0,3);
  auto d=SourceCoordinate(240,0,0,1,1,false,false,0,7,0,3);
  REQUIRE(a.x+1==b.x);
  REQUIRE(c.x+1!=d.x);
}
