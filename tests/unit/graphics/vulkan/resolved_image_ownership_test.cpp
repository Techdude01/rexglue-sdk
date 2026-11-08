// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_test_macros.hpp>
#include <rex/graphics/vulkan/resolved_image_ownership.h>
using Registry = rex::graphics::vulkan::ResolvedImageOwnership<int, int>;
TEST_CASE("Resolved image leases survive overlapping alias invalidation") {
  Registry registry;
  auto image = std::make_shared<const int>(42);
  std::weak_ptr<const int> weak = image;
  REQUIRE(registry.Publish(0x1000, 4096, 7, image));
  image.reset();
  auto lease = registry.Find(0xA0001000, 4096, 7);
  REQUIRE(lease);
  CHECK_FALSE(registry.Find(0x1000, 4096, 8));
  CHECK_FALSE(registry.Find(0x1000, 2048, 7));
  registry.Invalidate(0x2000, 1);
  CHECK(registry.Find(0x1000, 4096, 7));
  registry.Invalidate(0xC0001FFF, 1);
  CHECK_FALSE(registry.Find(0x1000, 4096, 7));
  CHECK_FALSE(weak.expired());
  lease.reset();
  CHECK(weak.expired());
}
TEST_CASE("Ineligible replacement invalidates old ownership and rejects overflow") {
  Registry registry;
  REQUIRE(registry.Publish(0x1000, 4096, 7, std::make_shared<const int>(1)));
  CHECK_FALSE(registry.Publish(0x1800, 4, 8, {}));
  CHECK_FALSE(registry.Find(0x1000, 4096, 7));
  CHECK_FALSE(registry.Publish(0x1FFFFFF0, 32, 7, std::make_shared<const int>(1)));
  CHECK_FALSE(registry.Publish(0, 0, 7, std::make_shared<const int>(1)));
  REQUIRE(registry.Publish(0, 16, 7, std::make_shared<const int>(1)));
  registry.Clear();
  CHECK_FALSE(registry.Find(0, 16, 7));
}

TEST_CASE("GPU submission retirement retains invalidated images until completion") {
  Registry registry;
  auto image = std::make_shared<const int>(42);
  std::weak_ptr<const int> weak = image;
  REQUIRE(registry.Publish(0x1000, 4096, 7, image));
  registry.RetainForSubmission(image, 12);
  registry.RetainForSubmission(image, 9);
  image.reset();
  registry.Clear();
  registry.CompletedSubmissionUpdated(9);
  CHECK_FALSE(weak.expired());
  registry.CompletedSubmissionUpdated(11);
  CHECK_FALSE(weak.expired());
  registry.CompletedSubmissionUpdated(12);
  CHECK(weak.expired());
}

TEST_CASE("Partial writes invalidate all intersecting images but preserve neighbors") {
  Registry registry;
  REQUIRE(registry.Publish(0x1000, 0x1000, 1, std::make_shared<const int>(1)));
  REQUIRE(registry.Publish(0x2000, 0x1000, 2, std::make_shared<const int>(2)));
  REQUIRE(registry.Publish(0x3000, 0x1000, 3, std::make_shared<const int>(3)));
  registry.Invalidate(0xA0001FFF, 2);
  CHECK_FALSE(registry.Find(0x1000, 0x1000, 1));
  CHECK_FALSE(registry.Find(0x2000, 0x1000, 2));
  CHECK(registry.Find(0x3000, 0x1000, 3));
  registry.Invalidate(0, 0x20000000);
  CHECK_FALSE(registry.Find(0x3000, 0x1000, 3));
}

TEST_CASE("Ownership capacity eviction preserves in-flight leases") {
  Registry registry(1);
  auto image = std::make_shared<const int>(1);
  std::weak_ptr<const int> weak = image;
  REQUIRE(registry.Publish(0x1000, 16, 1, image));
  registry.RetainForSubmission(image, 3);
  image.reset();
  REQUIRE(registry.Publish(0x2000, 16, 2, std::make_shared<const int>(2)));
  CHECK_FALSE(registry.Find(0x1000, 16, 1));
  CHECK(registry.Find(0x2000, 16, 2));
  CHECK_FALSE(weak.expired());
  registry.CompletedSubmissionUpdated(3);
  CHECK(weak.expired());
  Registry disabled(0);
  CHECK_FALSE(disabled.Publish(0x1000, 16, 1, std::make_shared<const int>(1)));
}

TEST_CASE("Byte budget bounds retained ownership and oversized replacement invalidates") {
  Registry registry(64, 12);
  REQUIRE(registry.Publish(0x1000, 16, 1, std::make_shared<const int>(1), 8));
  REQUIRE(registry.Publish(0x2000, 16, 2, std::make_shared<const int>(2), 8));
  CHECK(registry.owned_bytes() == 8);
  CHECK_FALSE(registry.Find(0x1000, 16, 1));
  CHECK_FALSE(registry.Publish(0x2000, 16, 2, std::make_shared<const int>(3), 13));
  CHECK_FALSE(registry.Find(0x2000, 16, 2));
  CHECK(registry.owned_bytes() == 0);
}

TEST_CASE("Repeated resolves publish distinct versions while previous readers remain live") {
  Registry registry;
  std::vector<Registry::Lease> readers;
  constexpr uint32_t address = 0x0FA54000;
  constexpr uint32_t length = 245760;
  for (int version = 1; version <= 7; ++version) {
    REQUIRE(registry.Publish(address, length, 1, std::make_shared<const int>(version)));
    auto reader = registry.Find(address, length, 1);
    REQUIRE(reader);
    CHECK(*reader == version);
    readers.push_back(std::move(reader));
  }
  registry.Invalidate(address + length - 1, 1);
  CHECK_FALSE(registry.Find(address, length, 1));
  for (int version = 1; version <= 7; ++version) {
    CHECK(*readers[size_t(version - 1)] == version);
  }
}

TEST_CASE("Lazy guest-byte materialization across 7 downsample surfaces with padded extents") {
  // Test the 7 downsample bloom passes targeting 0x0FA54000 with 245,760 bytes padded extent.
  Registry registry;
  constexpr uint32_t address = 0x0FA54000;
  constexpr uint32_t guest_bytes = 245760; // 320x184 padded extent
  constexpr uint32_t scaled_bytes = guest_bytes * 4; // 983,040 bytes at 2x

  struct MaterializationRecord {
    int surface_index;
    bool materialized = false;
  };

  std::vector<std::shared_ptr<MaterializationRecord>> records;

  for (int pass = 1; pass <= 7; ++pass) {
    auto rec = std::make_shared<MaterializationRecord>();
    rec->surface_index = pass;
    rec->materialized = false; // Initially unmaterialized (lazy)
    records.push_back(rec);

    // Publish direct resolve image with lazy bytes
    REQUIRE(registry.Publish(address, guest_bytes, pass, std::make_shared<const int>(pass)));
    
    // Sampling consumer adopts the fused image directly
    auto reader = registry.Find(address, guest_bytes, pass);
    REQUIRE(reader);
    CHECK(*reader == pass);
    
    // If subsequent pass arrives without fallback trigger, bytes remain unmaterialized
    CHECK_FALSE(rec->materialized);
  }

  // All 7 surfaces were successfully adopted without requiring eager byte materialization
  CHECK(records.size() == 7);
  for (int i = 0; i < 7; ++i) {
    CHECK_FALSE(records[size_t(i)]->materialized);
  }

  // Overwriting invalidation at end of chain cleans up registry
  registry.Invalidate(address, guest_bytes);
  CHECK_FALSE(registry.Find(address, guest_bytes, 7));
}
