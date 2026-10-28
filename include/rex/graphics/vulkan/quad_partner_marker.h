// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <fstream>
#include <string>

namespace rex::graphics::vulkan::diagnostics {

inline constexpr char kQuadPartnerMarkerPath[] =
    "/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/quad-partner-marker.txt";
inline constexpr uint32_t kQuadPartnerMarkerMode = 19;
inline constexpr uint32_t kQuadPartnerComponentMarkerMode = 20;
inline constexpr uint64_t kQuadPartnerMarkerFrame = 6000;

inline constexpr bool IsQuadPartnerMarkerShader(uint64_t hash) {
  // Head and jacket evidence uses this same PS. The other historical
  // diagnostic candidate hashes also shade unrelated scene geometry.
  return hash == UINT64_C(0x79A3448304FA0D22);
}

// Latch host controls only. This never selects a different marker shader binary.
inline bool QuadPartnerMarkerEnabled() {
  static const bool enabled = [] {
    std::ifstream input(kQuadPartnerMarkerPath);
    std::string line;
    return bool(std::getline(input, line)) && line == "frame=6000";
  }();
  return enabled;
}

inline uint32_t QuadPartnerDiagnosticRevision() {
  static const uint32_t revision = [] {
    std::ifstream input(kQuadPartnerMarkerPath);
    std::string line;
    while (std::getline(input, line)) {
      if (line == "diagnostic_variant=no-entry-block") return 12u;
      if (line == "diagnostic_variant=entry-gate-only") return 13u;
      if (line == "diagnostic_variant=no-sample-inputs") return 14u;
    }
    return 11u;
  }();
  return revision;
}

inline uint32_t QuadPartnerOutputMode() {
  static const uint32_t mode = [] {
    std::ifstream input(kQuadPartnerMarkerPath);
    std::string line;
    while (std::getline(input, line)) {
      if (line == "marker_match=component") return kQuadPartnerComponentMarkerMode;
    }
    return kQuadPartnerMarkerMode;
  }();
  return mode;
}

inline constexpr bool QuadPartnerOmitsEntryCounters(uint32_t revision) {
  return revision >= 12 && revision <= 14;
}

// Diagnostic only: draw guest 2x surfaces with one host sample. This is not a
// guest-MSAA correction. The file is absent for ordinary launches.
inline bool ForceHostSingleSample() {
  static const bool enabled = [] {
    std::ifstream input(
        "/Users/Kushal/Projects/cosmicR/out/face-stage-20261020/host-single-sample.txt");
    std::string line;
    return bool(std::getline(input, line)) && line == "host_samples=1";
  }();
  return enabled;
}

}  // namespace rex::graphics::vulkan::diagnostics
