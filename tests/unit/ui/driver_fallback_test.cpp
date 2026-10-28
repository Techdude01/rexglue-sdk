#include <catch2/catch_test_macros.hpp>
#include <rex/ui/vulkan/driver_fallback.h>
#include <memory>
#include <vector>

using rex::ui::vulkan::CreateWithMacOSDriverFallback;

TEST_CASE("Explicit MoltenVK is a single attempt, never a Krisp fallback", "[driver-fallback]") {
  std::vector<bool> attempts;
  int logs = 0;
  auto result = CreateWithMacOSDriverFallback(false, true,
      [&](bool moltenvk) { attempts.push_back(moltenvk); return std::make_unique<int>(1); },
      [&](const char*) { ++logs; });
  REQUIRE(result);
  CHECK(attempts == std::vector<bool>{true});
  CHECK(logs == 0);
}

TEST_CASE("Missing Krisp manifest falls back once and logs once", "[driver-fallback]") {
  std::vector<bool> attempts;
  int logs = 0;
  auto result = CreateWithMacOSDriverFallback(true, false,
      [&](bool moltenvk) { attempts.push_back(moltenvk); return std::make_unique<int>(1); },
      [&](const char*) { ++logs; });
  REQUIRE(result);
  CHECK(attempts == std::vector<bool>{true});
  CHECK(logs == 1);
}

TEST_CASE("Failed Krisp initialization retries MoltenVK exactly once", "[driver-fallback]") {
  for (bool fallback_succeeds : {false, true}) {
    std::vector<bool> attempts;
    int logs = 0;
    auto result = CreateWithMacOSDriverFallback(true, true,
        [&](bool moltenvk) {
          attempts.push_back(moltenvk);
          return moltenvk && fallback_succeeds ? std::make_unique<int>(1) : nullptr;
        }, [&](const char*) { ++logs; });
    CHECK(bool(result) == fallback_succeeds);
    CHECK(attempts == std::vector<bool>{false, true});
    CHECK(logs == 1);
  }
}

TEST_CASE("Working Krisp never opens MoltenVK", "[driver-fallback]") {
  std::vector<bool> attempts;
  int logs = 0;
  auto result = CreateWithMacOSDriverFallback(true, true,
      [&](bool moltenvk) { attempts.push_back(moltenvk); return std::make_unique<int>(1); },
      [&](const char*) { ++logs; });
  REQUIRE(result);
  CHECK(attempts == std::vector<bool>{false});
  CHECK(logs == 0);
}
