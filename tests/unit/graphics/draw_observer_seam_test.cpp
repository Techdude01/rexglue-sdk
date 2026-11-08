// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_test_macros.hpp>
#include <rex/system/interfaces/graphics.h>
#include <atomic>
#include <thread>
#include <vector>

namespace {

// Mock IGraphicsSystem to verify default interface behavior
class MockGraphicsSystemDefault : public rex::system::IGraphicsSystem {
 public:
  rex::X_STATUS SetupPresentation(rex::ui::WindowedAppContext*) override { return 0; }
  rex::X_STATUS SetupGuestGpu(rex::runtime::FunctionDispatcher*, rex::system::KernelState*) override {
    return 0;
  }
  bool has_presentation() const override { return false; }
  void Shutdown() override {}
};

// Simulation of CommandProcessor observer dispatch mechanics
class ObserverDispatchModel {
 public:
  void SetDrawObserver(rex::system::GpuDrawObserver observer, void* user) {
    draw_observer_.store(nullptr, std::memory_order_release);
    draw_observer_user_ = user;
    draw_observer_.store(observer, std::memory_order_release);
  }

  void EmitDraw(const rex::system::GpuDrawInfo& info) {
    if (rex::system::GpuDrawObserver observer = draw_observer_.load(std::memory_order_acquire)) {
      observer(draw_observer_user_, info);
    }
  }

  bool HasObserver() const {
    return draw_observer_.load(std::memory_order_relaxed) != nullptr;
  }

 private:
  std::atomic<rex::system::GpuDrawObserver> draw_observer_{nullptr};
  void* draw_observer_user_ = nullptr;
};

struct TestContext {
  uint32_t call_count = 0;
  uint32_t last_packet_address = 0;
  uint32_t last_opcode = 0;
  bool last_issued = false;
  uint64_t last_vs_hash = 0;
  uint64_t last_ps_hash = 0;
};

void TestDrawObserverCallback(void* user, const rex::system::GpuDrawInfo& info) {
  auto* ctx = static_cast<TestContext*>(user);
  if (ctx) {
    ctx->call_count++;
    ctx->last_packet_address = info.packet_address;
    ctx->last_opcode = info.opcode;
    ctx->last_issued = info.issued;
    ctx->last_vs_hash = info.vertex_shader_hash;
    ctx->last_ps_hash = info.pixel_shader_hash;
  }
}

}  // namespace

TEST_CASE("Draw observer default interface contract", "[draw_observer]") {
  MockGraphicsSystemDefault sys;
  TestContext ctx;
  // Default base implementation returns false when no command processor exists
  CHECK_FALSE(sys.SetDrawObserver(&TestDrawObserverCallback, &ctx));
  CHECK_FALSE(sys.SetHostRendering(false));
  CHECK_FALSE(sys.SetOcclusionQueryResolver(nullptr, nullptr));
  CHECK_FALSE(sys.SetSignalGate(nullptr, nullptr));
  CHECK_FALSE(sys.SetSwapObserver(nullptr, nullptr));
  rex::system::GammaRamp ramp{};
  CHECK_FALSE(sys.GetGammaRamp(ramp));
  uint64_t issued = 0, executed = 0;
  CHECK_FALSE(sys.GetSwapCounts(issued, executed));
}

TEST_CASE("Draw observer lifecycle and callback delivery", "[draw_observer]") {
  ObserverDispatchModel model;
  CHECK_FALSE(model.HasObserver());

  // 1. Initial emit with no observer
  uint32_t dummy_registers[16] = {0x12345678};
  rex::system::GpuDrawInfo info{};
  info.packet_address = 0x80010000;
  info.opcode = 0x2C;  // PM4_DRAW_INDX
  info.registers = dummy_registers;
  info.register_count = 16;
  info.bin_mask = 0xFF;
  info.bin_select = 0x01;
  info.issued = true;
  info.vertex_shader_hash = 0xAAAA5555;
  info.pixel_shader_hash = 0x5555AAAA;

  TestContext ctx;
  model.EmitDraw(info);
  CHECK(ctx.call_count == 0);

  // 2. Register observer
  model.SetDrawObserver(&TestDrawObserverCallback, &ctx);
  CHECK(model.HasObserver());

  model.EmitDraw(info);
  CHECK(ctx.call_count == 1);
  CHECK(ctx.last_packet_address == 0x80010000);
  CHECK(ctx.last_opcode == 0x2C);
  CHECK(ctx.last_issued == true);
  CHECK(ctx.last_vs_hash == 0xAAAA5555);
  CHECK(ctx.last_ps_hash == 0x5555AAAA);

  // 3. Second draw with different data
  info.packet_address = 0x80010040;
  info.opcode = 0x36;  // PM4_DRAW_INDX_2
  info.issued = false;
  model.EmitDraw(info);
  CHECK(ctx.call_count == 2);
  CHECK(ctx.last_packet_address == 0x80010040);
  CHECK(ctx.last_opcode == 0x36);
  CHECK(ctx.last_issued == false);

  // 4. Unregister observer
  model.SetDrawObserver(nullptr, nullptr);
  CHECK_FALSE(model.HasObserver());

  model.EmitDraw(info);
  CHECK(ctx.call_count == 2);  // Count must not increase
}

TEST_CASE("Draw observer concurrent register/unregister safety", "[draw_observer]") {
  ObserverDispatchModel model;
  std::atomic<bool> running{true};
  std::atomic<uint64_t> observed_events{0};

  auto observer_fn = [](void* user, const rex::system::GpuDrawInfo&) {
    auto* counter = static_cast<std::atomic<uint64_t>*>(user);
    if (counter) {
      counter->fetch_add(1, std::memory_order_relaxed);
    }
  };

  // GPU dispatcher thread
  std::thread gpu_thread([&]() {
    rex::system::GpuDrawInfo info{};
    info.packet_address = 0x82000000;
    info.opcode = 0x2C;
    info.issued = true;
    while (running.load(std::memory_order_relaxed)) {
      model.EmitDraw(info);
    }
  });

  // Host register/unregister thread
  for (int iter = 0; iter < 1000; ++iter) {
    model.SetDrawObserver(observer_fn, &observed_events);
    CHECK(model.HasObserver());
    std::this_thread::yield();
    model.SetDrawObserver(nullptr, nullptr);
    CHECK_FALSE(model.HasObserver());
  }

  running.store(false, std::memory_order_relaxed);
  gpu_thread.join();

  // Ensure that events were successfully received when registered without crashing
  CHECK(observed_events.load(std::memory_order_relaxed) > 0);
}
