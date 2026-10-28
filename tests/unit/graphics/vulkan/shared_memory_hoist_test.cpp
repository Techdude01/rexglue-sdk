// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <utility>
#include <vector>

// Standalone verification of the hazard detection and fallback mechanics for
// Option A shared-memory upload hoisting.
namespace {

class SharedMemoryHazardSimulator {
 public:
  void ResetSubmission() {
    submission_read_ranges_.clear();
    hoisted_uploads_pending_ = false;
  }

  void ResetFrame() {
    ResetSubmission();
    fallbacks_ = 0;
    successes_ = 0;
  }

  void AddSubmissionReadRange(uint32_t start, uint32_t length) {
    if (!length) return;
    submission_read_ranges_.push_back({start, length});
  }

  bool HasSubmissionReadRange(uint32_t start, uint32_t length) const {
    if (!length) return false;
    uint32_t end = start + length;
    for (const auto& r : submission_read_ranges_) {
      if (start < r.first + r.second && r.first < end) {
        return true;
      }
    }
    return false;
  }

  // Simulates VulkanSharedMemory::UploadRanges logic
  bool SimulateUpload(const std::vector<std::pair<uint32_t, uint32_t>>& upload_ranges,
                      bool hoist_enabled) {
    if (upload_ranges.empty()) return true;

    bool can_hoist = hoist_enabled;
    if (can_hoist) {
      for (const auto& r : upload_ranges) {
        if (HasSubmissionReadRange(r.first, r.second)) {
          can_hoist = false;
          break;
        }
      }
    }

    if (can_hoist) {
      hoisted_uploads_pending_ = true;
      ++successes_;
    } else {
      if (hoist_enabled) {
        ++fallbacks_;
      }
    }
    return can_hoist;
  }

  uint32_t fallbacks() const { return fallbacks_; }
  uint32_t successes() const { return successes_; }
  bool hoisted_uploads_pending() const { return hoisted_uploads_pending_; }
  size_t read_range_count() const { return submission_read_ranges_.size(); }

 private:
  std::vector<std::pair<uint32_t, uint32_t>> submission_read_ranges_;
  bool hoisted_uploads_pending_ = false;
  uint32_t fallbacks_ = 0;
  uint32_t successes_ = 0;
};

}  // namespace

TEST_CASE("Shared-memory upload hoisting: fresh range hoists without fallback", "[shmem_hoist]") {
  SharedMemoryHazardSimulator sim;
  sim.ResetFrame();

  // First upload before any reads: should hoist cleanly
  bool hoisted = sim.SimulateUpload({{0x1000, 0x1000}}, true);
  REQUIRE(hoisted == true);
  REQUIRE(sim.successes() == 1);
  REQUIRE(sim.fallbacks() == 0);
  REQUIRE(sim.hoisted_uploads_pending() == true);
}

TEST_CASE("Shared-memory upload hoisting: reading a range then uploading forces fallback", "[shmem_hoist]") {
  SharedMemoryHazardSimulator sim;
  sim.ResetFrame();

  // 1. Initial upload to range [0x1000, 0x2000) hoists successfully
  REQUIRE(sim.SimulateUpload({{0x1000, 0x1000}}, true) == true);
  REQUIRE(sim.successes() == 1);
  REQUIRE(sim.fallbacks() == 0);

  // 2. Guest draw reads range [0x1000, 0x2000)
  sim.AddSubmissionReadRange(0x1000, 0x1000);
  REQUIRE(sim.read_range_count() == 1);

  // 3. Subsequent upload to the EXACT SAME range within the same submission:
  // Must detect write-after-read hazard and force fallback to in-place upload!
  bool hoisted_conflict = sim.SimulateUpload({{0x1000, 0x1000}}, true);
  REQUIRE(hoisted_conflict == false);
  REQUIRE(sim.fallbacks() == 1);
  REQUIRE(sim.successes() == 1);

  // 4. Subsequent upload to an overlapping sub-range [0x1400, 0x1800) also forces fallback
  REQUIRE(sim.SimulateUpload({{0x1400, 0x400}}, true) == false);
  REQUIRE(sim.fallbacks() == 2);

  // 5. Subsequent upload to a disjoint range [0x5000, 0x6000) succeeds without fallback
  REQUIRE(sim.SimulateUpload({{0x5000, 0x1000}}, true) == true);
  REQUIRE(sim.successes() == 2);
  REQUIRE(sim.fallbacks() == 2);
}

TEST_CASE("Shared-memory upload hoisting: boundary and overlap hazard edge cases", "[shmem_hoist]") {
  SharedMemoryHazardSimulator sim;
  sim.ResetFrame();

  // Read range: [1000, 2000)
  sim.AddSubmissionReadRange(1000, 1000);

  // Abutting before: [500, 1000) -> no overlap
  REQUIRE(sim.HasSubmissionReadRange(500, 500) == false);
  REQUIRE(sim.SimulateUpload({{500, 500}}, true) == true);

  // Abutting after: [2000, 2500) -> no overlap
  REQUIRE(sim.HasSubmissionReadRange(2000, 500) == false);
  REQUIRE(sim.SimulateUpload({{2000, 500}}, true) == true);

  // Overlap by 1 byte at the start: [999, 1001) -> overlap!
  REQUIRE(sim.HasSubmissionReadRange(999, 2) == true);
  REQUIRE(sim.SimulateUpload({{999, 2}}, true) == false);

  // Overlap by 1 byte at the end: [1999, 2001) -> overlap!
  REQUIRE(sim.HasSubmissionReadRange(1999, 2) == true);
  REQUIRE(sim.SimulateUpload({{1999, 2}}, true) == false);

  // Enclosing: [500, 2500) -> overlap!
  REQUIRE(sim.HasSubmissionReadRange(500, 2000) == true);
  REQUIRE(sim.SimulateUpload({{500, 2000}}, true) == false);

  // Zero-length: no overlap
  REQUIRE(sim.HasSubmissionReadRange(1500, 0) == false);
}

TEST_CASE("Shared-memory upload hoisting: submission boundary clears conflict state", "[shmem_hoist]") {
  SharedMemoryHazardSimulator sim;
  sim.ResetFrame();

  // Draw in submission 1 reads [0x2000, 0x3000)
  sim.AddSubmissionReadRange(0x2000, 0x1000);
  REQUIRE(sim.SimulateUpload({{0x2000, 0x1000}}, true) == false);
  REQUIRE(sim.fallbacks() == 1);

  // Submission 1 ends, submission 2 begins
  sim.ResetSubmission();

  // In submission 2, uploading [0x2000, 0x3000) before any reads is safe again
  REQUIRE(sim.SimulateUpload({{0x2000, 0x1000}}, true) == true);
  REQUIRE(sim.successes() == 1);
  REQUIRE(sim.fallbacks() == 1);  // Fallbacks counter retained across submissions in frame
}
