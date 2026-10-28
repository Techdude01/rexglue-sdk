#include <catch2/catch_test_macros.hpp>
#include <rex/graphics/frame_cpu_timeline.h>

using rex::graphics::FrameCpuTimeline;

TEST_CASE("Nested CPU scopes partition the same frame interval", "[frame_cpu_timeline]") {
  FrameCpuTimeline timeline;
  auto start = FrameCpuTimeline::Clock::now();
  timeline.Start(start);
  {
    FrameCpuTimeline::Scope draw(timeline, FrameCpuTimeline::Bucket::kDraw);
    { FrameCpuTimeline::Scope submit(timeline, FrameCpuTimeline::Bucket::kSubmit); }
  }
  auto end = FrameCpuTimeline::Clock::now();
  timeline.Stop(end);
  std::chrono::nanoseconds total{0};
  for (auto value : timeline.durations()) total += value;
  REQUIRE(total == std::chrono::duration_cast<std::chrono::nanoseconds>(end - start));
  REQUIRE(timeline.current() == FrameCpuTimeline::Bucket::kOther);
}

TEST_CASE("Wait counters retain the most frequent addresses and reset at frame start", "[frame_cpu_timeline]") {
  FrameCpuTimeline timeline;
  timeline.Start(FrameCpuTimeline::Clock::now());
  timeline.ObserveWait(true, 42, 8);
  timeline.ObserveWait(true, 42, 5);
  timeline.ObserveWait(false, 12, 2);
  REQUIRE(timeline.wait_counts()[0] == 2);
  REQUIRE(timeline.wait_counts()[1] == 1);
  REQUIRE(timeline.wait_polls()[0].polls == 13);
  timeline.Stop(FrameCpuTimeline::Clock::now());
  timeline.Start(FrameCpuTimeline::Clock::now());
  REQUIRE(timeline.wait_counts()[0] == 0);
  REQUIRE(timeline.wait_polls()[0].polls == 0);
}
