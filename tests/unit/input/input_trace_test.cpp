#include <catch2/catch_test_macros.hpp>
#include <rex/input/input_trace.h>

#include <chrono>
#include <fstream>

using rex::input::InputTrace;
using rex::input::InputTraceState;

namespace {
struct TraceFile {
  std::filesystem::path path = std::filesystem::temp_directory_path() /
      ("rex-input-trace-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  ~TraceFile() { std::filesystem::remove(path); }
  void Write(const char* text) { std::ofstream(path) << text; }
};
}

TEST_CASE("Input trace records a complete route and preserves overdue short presses", "[input][trace]") {
  TraceFile file;
  InputTrace recorder;
  REQUIRE(recorder.OpenRecord(file.path));
  InputTraceState neutral, press;
  press.buttons = 0x1000;
  press.rx = -12345;
  press.rt = 255;
  REQUIRE(recorder.Record(0, neutral));
  REQUIRE(recorder.Record(100, press));
  REQUIRE(recorder.Record(101, neutral));
  REQUIRE(recorder.Record(500, neutral, true));
  REQUIRE(recorder.complete());
  InputTrace replay;
  REQUIRE(replay.OpenReplay(file.path));
  CHECK(replay.Replay(0).neutral());
  // A slow next poll must still see the tap, even though its release is overdue.
  const auto held = replay.Replay(10000);
  CHECK(held.buttons == 0x1000);
  CHECK(held.rx == -12345);
  CHECK(held.rt == 255);
  CHECK_FALSE(replay.complete());
  CHECK(replay.Replay(10001).neutral());
  CHECK(replay.complete());
  CHECK(replay.Replay(20000).neutral());
}

TEST_CASE("Input trace finishing waits for release and never overwrites a route", "[input][trace]") {
  TraceFile file;
  InputTrace recorder;
  REQUIRE(recorder.OpenRecord(file.path));
  InputTrace other;
  CHECK_FALSE(other.OpenRecord(file.path));
  InputTraceState pressed;
  pressed.buttons = 16;
  REQUIRE(recorder.Record(0, pressed, true));
  CHECK_FALSE(recorder.complete());
  InputTrace incomplete;
  CHECK_FALSE(incomplete.OpenReplay(file.path));
  REQUIRE(recorder.Record(10, {}, true));
  CHECK(recorder.complete());
}

TEST_CASE("Input trace rejects unsupported guest keystroke streams instead of losing input", "[input][trace]") {
  TraceFile file;
  InputTrace recorder;
  REQUIRE(recorder.OpenRecord(file.path));
  REQUIRE(recorder.Record(0, {}));
  recorder.RejectKeystroke();
  CHECK_FALSE(recorder.recording());
  CHECK_FALSE(recorder.complete());
  InputTrace replay;
  CHECK_FALSE(replay.OpenReplay(file.path));
}

TEST_CASE("Input trace poll replay keeps input polling independent of wall time", "[input][trace]") {
  TraceFile file;
  file.Write("COSMIC_INPUT_TRACE 1\nS 0 1 0 1 0 0 0 0 0 0 0\n"
             "S 100 3 0 2 16 0 0 0 0 0 0\nS 101 4 0 3 0 0 0 0 0 0 0\nEND 200 5\n");
  InputTrace replay;
  REQUIRE(replay.OpenReplay(file.path, true));
  CHECK(replay.Replay(99999).buttons == 0);
  CHECK(replay.Replay(99999).buttons == 0);
  CHECK(replay.Replay(99999).buttons == 16);
  CHECK(replay.Replay(99999).buttons == 0);
  CHECK_FALSE(replay.complete());
  replay.Replay(99999);
  CHECK(replay.complete());
}

TEST_CASE("Input trace finishes with resting stick drift and releases replay at its end", "[input][trace]") {
  TraceFile file;
  InputTrace recorder;
  REQUIRE(recorder.OpenRecord(file.path));
  InputTraceState resting;
  resting.lx = -207;
  resting.ly = -135;
  resting.rx = -813;
  resting.ry = 213;
  REQUIRE(recorder.Record(0, resting));
  REQUIRE(recorder.Record(100, resting, true));
  REQUIRE(recorder.complete());
  InputTrace replay;
  REQUIRE(replay.OpenReplay(file.path));
  CHECK(replay.Replay(0).rx == -813);  // Retain authentic input before route end.
  CHECK(replay.Replay(100).neutral());
  CHECK(replay.complete());
}

TEST_CASE("Input trace rejects malformed incomplete and out of range routes", "[input][trace]") {
  for (const char* content : {
           "COSMIC_INPUT_TRACE 2\n",
           "COSMIC_INPUT_TRACE 1\nS 0 1 0 0 0 0 0 0 0 0 0\n",
           "COSMIC_INPUT_TRACE 1\nS 0 1 0 0 16 0 0 0 0 0 0\nEND 1 2\n",
           "COSMIC_INPUT_TRACE 1\nS 0 1 0 0 0 0 0 0 0 32768 0\nEND 1 2\n",
           "COSMIC_INPUT_TRACE 1\nS 2 1 0 0 0 0 0 0 0 0 0\nEND 1 2\n",
           "COSMIC_INPUT_TRACE 1\nS 0 1 0 0 0 0 0 0 0 0 0\nEND 1 2\njunk\n"}) {
    TraceFile file;
    file.Write(content);
    InputTrace replay;
    CHECK_FALSE(replay.OpenReplay(file.path));
  }
}
