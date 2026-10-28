#include <catch2/catch_test_macros.hpp>
#include <rex/graphics/vulkan/scope_profile_timing.h>

using rex::graphics::vulkan::ScopeProfileMilliseconds;

TEST_CASE("Unsupported or incomplete scope timing is unavailable, not zero", "[scope-profile]") {
  CHECK_FALSE(ScopeProfileMilliseconds(0, 1.0, true, 0, 1, 100, 1));
  CHECK_FALSE(ScopeProfileMilliseconds(65, 1.0, true, 0, 1, 100, 1));
  CHECK_FALSE(ScopeProfileMilliseconds(64, 1.0, false, 0, 1, 100, 1));
  CHECK_FALSE(ScopeProfileMilliseconds(64, 1.0, true, 0, 0, 100, 1));
  CHECK_FALSE(ScopeProfileMilliseconds(64, 1.0, true, 0, 1, 100, 0));
}

TEST_CASE("Scope durations respect valid bits and accept a zero counter", "[scope-profile]") {
  CHECK(ScopeProfileMilliseconds(64, 1.0, true, 0, 1, 1000000, 1).value() == 1.0);
  CHECK(ScopeProfileMilliseconds(8, 1000000.0, true, 250, 1, 4, 1).value() == 10.0);
  CHECK(ScopeProfileMilliseconds(64, 1000000.0, true, UINT64_MAX - 4, 1, 5, 1).value() == 10.0);
  CHECK(ScopeProfileMilliseconds(64, 1.0, true, 10, 1, 10, 1).value() == 0.0);
}
