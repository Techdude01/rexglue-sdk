// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_test_macros.hpp>
#include <rex/system/interfaces/graphics.h>
#include <atomic>
#include <thread>

namespace {

class MockGraphicsSystemDefault : public rex::system::IGraphicsSystem {
 public:
  rex::X_STATUS SetupPresentation(rex::ui::WindowedAppContext*) override { return 0; }
  rex::X_STATUS SetupGuestGpu(rex::runtime::FunctionDispatcher*, rex::system::KernelState*) override {
    return 0;
  }
  bool has_presentation() const override { return false; }
  void Shutdown() override {}
};

// Mirrors CommandProcessor::SetSignalGate / PassSignalGate release/acquire + null skip.
class SignalGateDispatchModel {
 public:
  void SetSignalGate(rex::system::GpuSignalGate gate, void* user) {
    signal_gate_user_ = user;
    signal_gate_.store(gate, std::memory_order_release);
  }

  void PassSignalGate() {
    if (auto gate = signal_gate_.load(std::memory_order_acquire)) {
      gate(signal_gate_user_);
    }
  }

  bool HasGate() const { return signal_gate_.load(std::memory_order_relaxed) != nullptr; }

 private:
  std::atomic<rex::system::GpuSignalGate> signal_gate_{nullptr};
  void* signal_gate_user_ = nullptr;
};

}  // namespace

TEST_CASE("Signal gate default interface contract", "[signal_gate]") {
  MockGraphicsSystemDefault sys;
  CHECK_FALSE(sys.SetSignalGate(nullptr, nullptr));
}

TEST_CASE("Signal gate lifecycle and callback delivery", "[signal_gate]") {
  SignalGateDispatchModel model;
  CHECK_FALSE(model.HasGate());

  std::atomic<uint32_t> hits{0};
  model.PassSignalGate();
  CHECK(hits.load() == 0);

  model.SetSignalGate(
      [](void* user) {
        static_cast<std::atomic<uint32_t>*>(user)->fetch_add(1, std::memory_order_relaxed);
      },
      &hits);
  CHECK(model.HasGate());
  model.PassSignalGate();
  CHECK(hits.load() == 1);
  model.PassSignalGate();
  CHECK(hits.load() == 2);

  model.SetSignalGate(nullptr, nullptr);
  CHECK_FALSE(model.HasGate());
  model.PassSignalGate();
  CHECK(hits.load() == 2);
}

TEST_CASE("Signal gate concurrent register/unregister safety", "[signal_gate]") {
  SignalGateDispatchModel model;
  std::atomic<bool> running{true};
  std::atomic<uint64_t> hits{0};

  std::thread gpu_thread([&]() {
    while (running.load(std::memory_order_relaxed)) {
      model.PassSignalGate();
    }
  });

  for (int iter = 0; iter < 1000; ++iter) {
    model.SetSignalGate(
        [](void* user) {
          static_cast<std::atomic<uint64_t>*>(user)->fetch_add(1, std::memory_order_relaxed);
        },
        &hits);
    CHECK(model.HasGate());
    std::this_thread::yield();
    model.SetSignalGate(nullptr, nullptr);
    CHECK_FALSE(model.HasGate());
  }

  running.store(false, std::memory_order_relaxed);
  gpu_thread.join();
  CHECK(hits.load(std::memory_order_relaxed) > 0);
}
