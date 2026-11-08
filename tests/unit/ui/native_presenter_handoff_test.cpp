// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_test_macros.hpp>
#include <rex/ui/presenter.h>
#include <rex/ui/surface.h>
#include <rex/ui/window.h>
#include <rex/ui/windowed_app_context.h>

namespace {

class DummySurface final : public rex::ui::Surface {
 public:
  DummySurface() = default;
  TypeIndex GetType() const override { return kTypeIndex_CAMetalLayer; }
  bool GetSizeImpl(uint32_t& w, uint32_t& h) const override {
    w = 1280;
    h = 720;
    return true;
  }
};

class MockPresenter final : public rex::ui::Presenter {
 public:
  MockPresenter() : Presenter(nullptr) {}

  rex::ui::Surface::TypeFlags GetSupportedSurfaceTypes() const override {
    return rex::ui::Surface::kTypeFlag_CAMetalLayer;
  }
  bool CaptureGuestOutput(rex::ui::RawImage&) override { return false; }

  int connect_calls = 0;
  int disconnect_calls = 0;
  bool is_connected = false;

 protected:
  SurfacePaintConnectResult ConnectOrReconnectPaintingToSurfaceFromUIThread(
      rex::ui::Surface&, uint32_t, uint32_t, bool, bool& is_vsync_implicit) override {
    connect_calls++;
    is_connected = true;
    is_vsync_implicit = false;
    return SurfacePaintConnectResult::kSuccess;
  }

  void DisconnectPaintingFromSurfaceFromUIThreadImpl() override {
    disconnect_calls++;
    is_connected = false;
  }

  bool RefreshGuestOutputImpl(uint32_t, uint32_t, uint32_t,
                              std::function<bool(GuestOutputRefreshContext&)>, bool&) override {
    return false;
  }
  PaintResult PaintAndPresentImpl(bool) override { return PaintResult::kPresented; }
};

class MockWindow final : public rex::ui::Window {
 public:
  explicit MockWindow(rex::ui::WindowedAppContext& ctx)
      : Window(ctx, "test_window", 1280, 720) {}

  void* GetNativeWindowHandle() const override {
    return mock_native_handle_;
  }

  rex::ui::Presenter* get_presenter() const { return presenter(); }
  bool has_surface() const { return HasSurface(); }

  void* mock_native_handle_ = reinterpret_cast<void*>(0x12345678);

 protected:
  bool OpenImpl() override { return true; }
  void RequestCloseImpl() override {}
  std::unique_ptr<rex::ui::Surface> CreateSurfaceImpl(
      rex::ui::Surface::TypeFlags allowed_types) override {
    if (allowed_types & rex::ui::Surface::kTypeFlag_CAMetalLayer) {
      return std::make_unique<DummySurface>();
    }
    return nullptr;
  }
  void RequestPaintImpl() override {}
};

class MockAppContext final : public rex::ui::WindowedAppContext {
 protected:
  void NotifyUILoopOfPendingFunctions() override {}
  void PlatformQuitFromUIThread() override {}
};

}  // namespace

TEST_CASE("Transactional Presenter Handoff and Restoration", "[presenter_handoff]") {
  MockAppContext context;
  MockWindow window(context);
  REQUIRE(window.Open());

  MockPresenter xenos_presenter;

  // 1. Initial attach of Xenos presenter
  window.SetPresenter(&xenos_presenter);
  CHECK(window.get_presenter() == &xenos_presenter);
  CHECK(window.has_surface());
  CHECK(xenos_presenter.connect_calls == 1);
  CHECK(xenos_presenter.disconnect_calls == 0);
  CHECK(xenos_presenter.is_connected == true);

  // 2. Simulated Native takeover start: detach Xenos presenter
  window.SetPresenter(nullptr);
  CHECK(window.get_presenter() == nullptr);
  CHECK_FALSE(window.has_surface());
  CHECK(xenos_presenter.disconnect_calls == 1);
  CHECK(xenos_presenter.is_connected == false);

  // 3. Simulated Native initialization failure (e.g. device creation / swapchain failed)
  bool native_init_succeeded = false;
  CHECK_FALSE(native_init_succeeded);

  // 4. Transactional restoration of Xenos presenter
  window.SetPresenter(&xenos_presenter);
  CHECK(window.get_presenter() == &xenos_presenter);
  CHECK(window.has_surface());
  CHECK(xenos_presenter.connect_calls == 2);
  CHECK(xenos_presenter.is_connected == true);

  // 5. Verify cleanup on final window teardown
  window.SetPresenter(nullptr);
  CHECK(xenos_presenter.disconnect_calls == 2);
  CHECK(xenos_presenter.is_connected == false);
}

TEST_CASE("Native window handle query contract", "[presenter_handoff]") {
  MockAppContext context;
  MockWindow window(context);
  CHECK(window.GetNativeWindowHandle() == reinterpret_cast<void*>(0x12345678));
  window.mock_native_handle_ = nullptr;
  CHECK(window.GetNativeWindowHandle() == nullptr);
}
