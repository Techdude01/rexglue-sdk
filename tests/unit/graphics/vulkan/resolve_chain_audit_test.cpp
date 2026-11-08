#include <catch2/catch_test_macros.hpp>
#include <rex/graphics/vulkan/resolve_chain_audit.h>
using rex::graphics::vulkan::ResolveChainAudit;

TEST_CASE("Resolve chain links exact overlapping extents and preserves partial writers") {
  ResolveChainAudit audit;
  audit.ObserveFrame(4000);
  auto first = audit.Write(4096, 4096);
  auto second = audit.Write(5120, 1024);
  auto sample = audit.Load(1, false, 4096, 4096);
  REQUIRE(sample.valid);
  REQUIRE_FALSE(sample.previous);
  REQUIRE(sample.unknown_bytes == 0);
  REQUIRE(sample.producer_bytes[first] == 3072);
  REQUIRE(sample.producer_bytes[second] == 1024);
  REQUIRE(audit.Load(1, false, 4096, 4096).same_observed_writers);
  audit.Write(16384, 4096);
  REQUIRE(audit.Load(1, false, 4096, 4096).same_observed_writers);
  audit.Write(4097, 1);
  REQUIRE_FALSE(audit.Load(1, false, 4096, 4096).same_observed_writers);
}

TEST_CASE("Unknown history never means identical content and texture parts stay separate") {
  ResolveChainAudit audit;
  auto first = audit.Load(1, false, 0, 1024);
  REQUIRE(first.unknown_bytes == 1024);
  auto repeat = audit.Load(1, false, 0, 1024);
  REQUIRE(repeat.same_observed_writers);  // Only observed writers, not bytes.
  REQUIRE(repeat.unknown_bytes == 1024);
  REQUIRE_FALSE(audit.Load(1, true, 0, 1024).previous);
  REQUIRE_FALSE(audit.Load(2, false, 0, 1024).previous);
  audit.ForgetTexture(1);
  REQUIRE_FALSE(audit.Load(1, false, 0, 1024).previous);
}

TEST_CASE("Source changes, capture gaps and address overflow stay conservative") {
  ResolveChainAudit audit;
  audit.ObserveFrame(4000);
  auto first = audit.Write(0, 8192);
  audit.Load(1, false, 0, 8192);
  audit.ObserveFrame(4001);
  REQUIRE(audit.Load(1, false, 0, 8192).same_observed_writers);
  REQUIRE_FALSE(audit.Load(1, false, 1, 8191).same_observed_writers);
  REQUIRE(audit.Write(0x1fffffff, 2) == 0);
  REQUIRE_FALSE(audit.Load(1, false, 0xffffffff, 2).valid);
  REQUIRE_FALSE(audit.Load(1, false, 0, 0).valid);
  audit.ObserveFrame(4003);
  auto unknown = audit.Load(1, false, 0, 8192);
  REQUIRE_FALSE(unknown.previous);
  REQUIRE(unknown.unknown_bytes == 8192);
  REQUIRE(audit.Write(0, 8192) > first);  // IDs cannot alias across an audit gap.
}

TEST_CASE("Multiple ranges conserve source coverage and erase overwritten writers") {
  ResolveChainAudit audit;
  auto a = audit.Write(100, 50);
  auto b = audit.Write(200, 50);
  auto mixed = audit.Load(1, false, 0, 300);
  REQUIRE(mixed.unknown_bytes == 200);
  REQUIRE(mixed.producer_bytes[a] == 50);
  REQUIRE(mixed.producer_bytes[b] == 50);
  auto replacement = audit.Write(0, 300);
  auto covered = audit.Load(1, false, 0, 300);
  REQUIRE(covered.producer_bytes.size() == 1);
  REQUIRE(covered.producer_bytes[replacement] == 300);
  REQUIRE(covered.unknown_bytes == 0);
}
