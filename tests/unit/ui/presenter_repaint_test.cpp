// Copyright 2026 ReXGlue contributors.
// SPDX-License-Identifier: BSD-3-Clause
#include <catch2/catch_test_macros.hpp>

#include <thread>

#include <rex/ui/imgui_dialog.h>
#include <rex/ui/imgui_drawer.h>
#include <rex/ui/flags.h>
#include <rex/ui/overlay/achievement_toast.h>
#include <rex/ui/window.h>
#include <rex/ui/windowed_app_context.h>

namespace {
class TestContext final : public rex::ui::WindowedAppContext {
 protected:
  void NotifyUILoopOfPendingFunctions() override {}
  void PlatformQuitFromUIThread() override {}
};

class VisibleDialog final : public rex::ui::ImGuiDialog {
 public:
  explicit VisibleDialog(rex::ui::ImGuiDrawer* drawer) : ImGuiDialog(drawer) {}
};

class TestPresenter final : public rex::ui::Presenter {
 public:
  TestPresenter() : Presenter(nullptr) {}
  using Presenter::NoteGuestOutputPresented;
  using Presenter::NoteSwapchainNeedsPresent;
  using Presenter::ShouldSkipUnchangedGuestOutputPaint;
  rex::ui::Surface::TypeFlags GetSupportedSurfaceTypes() const override { return 0; }
  bool CaptureGuestOutput(rex::ui::RawImage&) override { return false; }

 protected:
  SurfacePaintConnectResult ConnectOrReconnectPaintingToSurfaceFromUIThread(
      rex::ui::Surface&, uint32_t, uint32_t, bool, bool&) override {
    return SurfacePaintConnectResult::kFailure;
  }
  void DisconnectPaintingFromSurfaceFromUIThreadImpl() override {}
  bool RefreshGuestOutputImpl(uint32_t, uint32_t, uint32_t,
      std::function<bool(GuestOutputRefreshContext&)>, bool&) override {
    return false;
  }
  PaintResult PaintAndPresentImpl(bool) override { return PaintResult::kNotPresented; }
};

struct EnableSkip {
  bool previous = REXCVAR_GET(present_skip_unchanged_guest_output);
  EnableSkip() { REXCVAR_SET(present_skip_unchanged_guest_output, true); }
  ~EnableSkip() { REXCVAR_SET(present_skip_unchanged_guest_output, previous); }
};
}  // namespace

TEST_CASE("An idle achievement listener does not require UI painting", "[presenter_repaint]") {
  TestContext context;
  // Construct the real window/drawer/dialog objects without opening a native
  // window or creating a GPU device.
  auto window = rex::ui::Window::Create(context, "repaint test", 1280, 720);
  rex::ui::ImGuiDrawer drawer(window.get(), 64);
  rex::ui::AchievementToastDialog toast(&drawer, nullptr, nullptr);

  REQUIRE(drawer.HasDialogs());
  CHECK_FALSE(drawer.NeedsPaint());
  {
    VisibleDialog dialog(&drawer);
    CHECK(drawer.NeedsPaint());
  }
  CHECK_FALSE(drawer.NeedsPaint());

  // The production notification entry point is called from guest threads.
  std::thread producer([&] { toast.Push({}); });
  producer.join();
  context.ExecutePendingFunctionsFromUIThread();
  CHECK(drawer.NeedsPaint());
}

TEST_CASE("Presenter skips dormant UI but preserves invalidation and queued toasts", "[presenter_repaint]") {
  EnableSkip enable;
  TestContext context;
  auto window = rex::ui::Window::Create(context, "repaint test", 1280, 720);
  TestPresenter presenter;
  rex::ui::ImGuiDrawer drawer(window.get(), 64);
  drawer.SetPresenter(&presenter);
  rex::ui::AchievementToastDialog toast(&drawer, nullptr, nullptr);

  CHECK_FALSE(presenter.ShouldSkipUnchangedGuestOutputPaint(true));
  presenter.NoteGuestOutputPresented(0);
  CHECK(presenter.ShouldSkipUnchangedGuestOutputPaint(true));
  {
    VisibleDialog dialog(&drawer);
    presenter.NoteGuestOutputPresented(0);
    CHECK_FALSE(presenter.ShouldSkipUnchangedGuestOutputPaint(true));
  }
  // Closing a visible dialog must clear its old pixels once, even though the
  // persistent (idle) toast listener keeps the same drawer registered.
  CHECK_FALSE(presenter.ShouldSkipUnchangedGuestOutputPaint(true));
  presenter.NoteGuestOutputPresented(0);
  CHECK(presenter.ShouldSkipUnchangedGuestOutputPaint(true));
  presenter.NoteSwapchainNeedsPresent();
  CHECK_FALSE(presenter.ShouldSkipUnchangedGuestOutputPaint(true));
  presenter.NoteGuestOutputPresented(0);
  CHECK(presenter.ShouldSkipUnchangedGuestOutputPaint(true));

  std::thread producer([&] { toast.Push({}); });
  producer.join();
  context.ExecutePendingFunctionsFromUIThread();
  CHECK_FALSE(presenter.ShouldSkipUnchangedGuestOutputPaint(true));
  presenter.NoteGuestOutputPresented(0);
  CHECK_FALSE(presenter.ShouldSkipUnchangedGuestOutputPaint(true));
}

TEST_CASE("A queued toast wake cannot outlive its drawer", "[presenter_repaint]") {
  EnableSkip enable;
  TestContext context;
  auto window = rex::ui::Window::Create(context, "repaint test", 1280, 720);
  {
    rex::ui::ImGuiDrawer drawer(window.get(), 64);
    rex::ui::AchievementToastDialog toast(&drawer, nullptr, nullptr);
    std::thread producer([&] { toast.Push({}); });
    producer.join();
  }
  // Executes the real deferred wake after its target was destroyed.
  context.ExecutePendingFunctionsFromUIThread();
  SUCCEED();
}
