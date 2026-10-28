/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2013 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <algorithm>
#include <cmath>

#include <rex/dbg.h>
#include <rex/input/device_assignment.h>
#include <rex/input/flags.h>
#include <rex/input/input_driver.h>
#include <rex/input/input_system.h>
#include <rex/input/mnk/mnk_input_driver.h>
#include <rex/input/nop/nop_input_driver.h>
#include <rex/input/sdl/sdl_input_driver.h>
#include <rex/input/startup/startup_input_driver.h>
#include <rex/input/state_merge.h>
#include <rex/input/xinput/xinput_input_driver.h>
#include <rex/logging.h>

REXCVAR_DEFINE_STRING(input_backend, "sdl", "Input", "Input backend: sdl, xinput")
    .allowed({"sdl", "xinput"});

REXCVAR_DEFINE_BOOL(guide_button, false, "Input", "Enable guide button pass-through");
REXCVAR_DEFINE_BOOL(test_startup_input, false, "Input/Test",
                    "Enable bounded synthetic startup input for automated runs");
REXCVAR_DEFINE_STRING(startup_input_sequence, "a@180:2", "Input/Test",
                      "Synthetic button or stick pulses as input@poll:duration, comma-separated");
REXCVAR_DEFINE_UINT32(startup_input_cutoff_poll, 240, "Input/Test",
                      "Stop synthetic startup input after this guest XInput poll count");
REXCVAR_DEFINE_STRING(input_record_path, "", "Input/Test",
                      "Record user-0 controller state transitions; finish via PATH.stop after releasing controls");
REXCVAR_DEFINE_STRING(input_replay_path, "", "Input/Test", "Replay a completed controller input trace");
REXCVAR_DEFINE_STRING(input_replay_clock, "time", "Input/Test", "Input trace pacing: time or poll")
    .allowed({"time", "poll"});
namespace rex::input {

namespace {

// Synthetic devices are parked past every physical ordinal so they cannot push
// a real pad off guest user 0. SlotAssignment routes them by their synthetic
// flag and never reads this value.
constexpr uint32_t kSyntheticOrdinal = UINT32_MAX;

}  // namespace

InputSystem::InputSystem(rex::ui::Window* window) : window_(window) {}

InputSystem::~InputSystem() = default;

X_STATUS InputSystem::Setup() {
  const auto& record = REXCVAR_GET(input_record_path);
  const auto& replay = REXCVAR_GET(input_replay_path);
  if ((!record.empty() && !replay.empty()) ||
      ((!record.empty() || !replay.empty()) && REXCVAR_GET(test_startup_input))) {
    REXLOG_ERROR("Input trace: record/replay and synthetic startup input are mutually exclusive");
    return X_STATUS_UNSUCCESSFUL;
  }
  if (!record.empty() || !replay.empty()) {
    trace_ = std::make_unique<InputTrace>();
    const bool opened = !record.empty()
                            ? trace_->OpenRecord(record)
                            : trace_->OpenReplay(replay, REXCVAR_GET(input_replay_clock) == "poll");
    if (!opened) {
      REXLOG_ERROR("Input trace: {}", trace_->error());
      return X_STATUS_UNSUCCESSFUL;
    }
    trace_stop_path_ = record + ".stop";
    REXLOG_INFO("Input trace: mode={} path='{}' clock={} user=0",
                record.empty() ? "replay" : "record", record.empty() ? replay : record,
                REXCVAR_GET(input_replay_clock));
  }
  return X_STATUS_SUCCESS;
}

uint64_t InputSystem::TraceMicros() {
  const auto now = std::chrono::steady_clock::now();
  if (trace_start_ == std::chrono::steady_clock::time_point{}) trace_start_ = now;
  return uint64_t(std::chrono::duration_cast<std::chrono::microseconds>(now - trace_start_).count());
}

void InputSystem::RecordInputState(uint32_t user_index, X_RESULT result, const X_INPUT_STATE& state) {
  if (!trace_ || user_index != 0 || trace_->replaying()) return;
  std::lock_guard lock(trace_mutex_);
  if (!trace_->recording()) return;
  if ((trace_polls_++ % 64) == 0 && !trace_finish_requested_) {
    std::error_code ec;
    trace_finish_requested_ = std::filesystem::exists(trace_stop_path_, ec) && !ec;
  }
  const auto& pad = state.gamepad;
  const InputTraceState value{uint32_t(result), uint32_t(state.packet_number), uint16_t(pad.buttons),
                              pad.left_trigger, pad.right_trigger, int16_t(pad.thumb_lx),
                              int16_t(pad.thumb_ly), int16_t(pad.thumb_rx), int16_t(pad.thumb_ry)};
  if (!trace_->Record(TraceMicros(), value, trace_finish_requested_)) {
    REXLOG_ERROR("Input trace recording failed: {}", trace_->error());
  } else if (trace_->complete()) {
    REXLOG_INFO("Input trace recording complete: controls released, end marker persisted");
  }
}

void InputSystem::Shutdown() {
  // device_owners_ holds raw driver pointers.
  devices_.clear();
  device_owners_.clear();
  drivers_.clear();
}

void InputSystem::AddDriver(std::unique_ptr<InputDriver> driver) {
  drivers_.push_back(std::move(driver));
}

void InputSystem::AttachWindow(rex::ui::Window* window) {
  window_ = window;
  for (auto& driver : drivers_) {
    driver->OnWindowAvailable(window);
  }
}

void InputSystem::SetActiveCallback(std::function<bool()> callback) {
  for (auto& driver : drivers_) {
    driver->set_is_active_callback(callback);
  }
}

void InputSystem::SetDeviceAssignment(std::unique_ptr<DeviceAssignment> assignment) {
  assignment_ = std::move(assignment);
  if (assignment_) {
    assignment_->OnDevicesChanged(devices_);
  }
}

void InputSystem::RefreshDevices() {
  std::vector<DeviceInfo> seen;
  std::vector<InputDriver*> owners;
  std::vector<DeviceInfo> enumerated;
  for (auto& driver : drivers_) {
    enumerated.clear();
    driver->EnumerateDevices(enumerated);
    for (auto& info : enumerated) {
      seen.push_back(info);
      owners.push_back(driver.get());
    }
  }

  // Carry forward ordinals already handed out, so a device keeps its guest user
  // when another pad is unplugged.
  bool changed = seen.size() != devices_.size();
  std::vector<bool> fresh(seen.size(), false);
  for (size_t i = 0; i < seen.size(); i++) {
    auto existing = std::find_if(devices_.begin(), devices_.end(),
                                 [&](const DeviceInfo& d) { return d.id == seen[i].id; });
    if (existing != devices_.end()) {
      seen[i].ordinal = existing->ordinal;
      continue;
    }
    fresh[i] = true;
    changed = true;
  }

  // Runs after the carry-forward pass so a new device cannot take an ordinal a
  // live one is still holding. Lowest free rather than a growing counter,
  // because a reconnected pad arrives as a new device and would otherwise walk
  // off the end of the guest users.
  for (size_t i = 0; i < seen.size(); i++) {
    if (!fresh[i]) {
      continue;
    }
    // Only physical devices consume an ordinal, so the first pad to connect is
    // guest user 0 however many synthetic devices enumerated ahead of it.
    if (seen[i].synthetic) {
      seen[i].ordinal = kSyntheticOrdinal;
      fresh[i] = false;
      continue;
    }
    auto taken = [&](uint32_t candidate) {
      for (size_t j = 0; j < seen.size(); j++) {
        if (!fresh[j] && !seen[j].synthetic && seen[j].ordinal == candidate) {
          return true;
        }
      }
      return false;
    };
    uint32_t ordinal = 0;
    while (taken(ordinal)) {
      ordinal++;
    }
    seen[i].ordinal = ordinal;
    fresh[i] = false;
  }

  for (const auto& old : devices_) {
    if (std::none_of(seen.begin(), seen.end(),
                     [&](const DeviceInfo& d) { return d.id == old.id; })) {
      active_devices_.Forget(old.id);
      changed = true;
    }
  }

  std::vector<size_t> order(seen.size());
  for (size_t i = 0; i < order.size(); i++) {
    order[i] = i;
  }
  // Stable: synthetic devices share one ordinal, and their relative order
  // decides which answers GetCapabilities when no pad is attached.
  std::stable_sort(order.begin(), order.end(),
                   [&](size_t a, size_t b) { return seen[a].ordinal < seen[b].ordinal; });

  devices_.clear();
  device_owners_.clear();
  for (size_t i : order) {
    devices_.push_back(seen[i]);
    device_owners_.push_back(owners[i]);
  }

  if (changed && assignment_) {
    assignment_->OnDevicesChanged(devices_);
  }
}

InputDriver* InputSystem::DriverForDevice(DeviceId id) {
  for (size_t i = 0; i < devices_.size(); i++) {
    if (devices_[i].id == id) {
      return device_owners_[i];
    }
  }
  return nullptr;
}

const DeviceInfo* InputSystem::DeviceInfoFor(DeviceId id) const {
  for (const auto& device : devices_) {
    if (device.id == id) {
      return &device;
    }
  }
  return nullptr;
}

X_RESULT InputSystem::GetCapabilities(uint32_t user_index, uint32_t flags,
                                      X_INPUT_CAPABILITIES* out_caps) {
  SCOPE_profile_cpu_f("hid");
  if (!out_caps || !assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  if (trace_ && trace_->replaying() && !trace_->complete() && user_index == 0) {
    *out_caps = {};
    out_caps->type = 1;
    out_caps->sub_type = 1;
    out_caps->gamepad.buttons = 0xFFFF;
    out_caps->gamepad.left_trigger = out_caps->gamepad.right_trigger = 255;
    out_caps->gamepad.thumb_lx = out_caps->gamepad.thumb_ly = 32767;
    out_caps->gamepad.thumb_rx = out_caps->gamepad.thumb_ry = 32767;
    return X_ERROR_SUCCESS;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);
  if (ids.empty()) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  // Prefer the pad in hand, so button glyphs follow it rather than whichever
  // device enumerated first.
  DeviceId chosen = active_devices_.Active(user_index);
  if (std::find(ids.begin(), ids.end(), chosen) == ids.end()) {
    chosen = ids.front();
  }

  auto* driver = DriverForDevice(chosen);
  if (!driver) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  return driver->GetDeviceCapabilities(chosen, flags, out_caps);
}

X_RESULT InputSystem::GetState(uint32_t user_index, X_INPUT_STATE* out_state) {
  SCOPE_profile_cpu_f("hid");
  if (trace_ && trace_->replaying() && !trace_->complete() && user_index == 0) {
    std::lock_guard lock(trace_mutex_);
    const auto value = trace_->Replay(TraceMicros());
    if (out_state) {
      *out_state = {};
      out_state->packet_number = value.packet;
      auto& pad = out_state->gamepad;
      pad.buttons = value.buttons;
      pad.left_trigger = value.lt;
      pad.right_trigger = value.rt;
      pad.thumb_lx = value.lx;
      pad.thumb_ly = value.ly;
      pad.thumb_rx = value.rx;
      pad.thumb_ry = value.ry;
    }
    if (trace_->complete() && !trace_reported_) {
      trace_reported_ = true;
      REXLOG_INFO("Input trace replay complete: route ended with controls released");
    }
    return X_RESULT(value.result);
  }
  if (!assignment_) {
    RecordInputState(user_index, X_ERROR_DEVICE_NOT_CONNECTED, {});
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);

  X_INPUT_STATE merged = {};
  bool any = false;
  bool test_controller_connected = false;
  for (DeviceId id : ids) {
    auto* driver = DriverForDevice(id);
    if (!driver) {
      continue;
    }
    X_INPUT_STATE state = {};
    if (driver->GetDeviceState(id, &state) != X_ERROR_SUCCESS) {
      continue;
    }
    const DeviceInfo* info = DeviceInfoFor(id);
    test_controller_connected |= info && info->name == "Test startup controller";
    active_devices_.Observe(user_index, id, state.gamepad);
    if (!any) {
      merged = state;
      any = true;
    } else {
      MergeInto(merged, state);
    }
  }

  if (!any) {
    RecordInputState(user_index, X_ERROR_DEVICE_NOT_CONNECTED, {});
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }
  RecordInputState(user_index, X_ERROR_SUCCESS, merged);
  if (out_state) {
    *out_state = merged;
    if (test_controller_connected) {
      const uint16_t buttons = static_cast<uint16_t>(merged.gamepad.buttons);
      if (buttons != last_test_guest_buttons_) {
        REXLOG_INFO("Startup input guest XInput response user={} buttons=0x{:04X}", user_index,
                    buttons);
        last_test_guest_buttons_ = buttons;
      }
    }
  }
  return X_ERROR_SUCCESS;
}

X_RESULT InputSystem::SetState(uint32_t user_index, X_INPUT_VIBRATION* vibration) {
  SCOPE_profile_cpu_f("hid");
  if (!assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);

  // Every pad on this user belongs to the same player, so all of them buzz.
  // Only pads decide the result: synthetic devices accept any vibration and
  // would otherwise report success for a pad that never rumbled.
  bool any_pad = false;
  bool any_synthetic = false;
  bool pad_rumbled = false;
  X_RESULT pad_error = X_ERROR_DEVICE_NOT_CONNECTED;
  for (DeviceId id : ids) {
    auto* driver = DriverForDevice(id);
    const DeviceInfo* info = DeviceInfoFor(id);
    if (!driver || !info) {
      continue;
    }
    X_RESULT result = driver->SetDeviceVibration(id, vibration);
    if (info->synthetic) {
      any_synthetic = true;
      continue;
    }
    any_pad = true;
    if (result == X_ERROR_SUCCESS) {
      pad_rumbled = true;
    } else {
      pad_error = result;
    }
  }
  if (any_pad) {
    return pad_rumbled ? X_ERROR_SUCCESS : pad_error;
  }
  return any_synthetic ? X_ERROR_SUCCESS : X_ERROR_DEVICE_NOT_CONNECTED;
}

X_RESULT InputSystem::GetKeystroke(uint32_t user_index, uint32_t flags,
                                   X_INPUT_KEYSTROKE* out_keystroke) {
  SCOPE_profile_cpu_f("hid");
  if (trace_ && trace_->replaying() && !trace_->complete() && user_index == 0) return X_ERROR_EMPTY;
  if (!assignment_) {
    return X_ERROR_DEVICE_NOT_CONNECTED;
  }

  RefreshDevices();
  std::vector<DeviceId> ids;
  assignment_->DevicesForUser(user_index, ids);

  bool any_connected = false;
  for (DeviceId id : ids) {
    auto* driver = DriverForDevice(id);
    if (!driver) {
      continue;
    }
    X_RESULT result = driver->GetDeviceKeystroke(id, flags, out_keystroke);
    if (result == X_ERROR_SUCCESS) {
      if (trace_ && user_index == 0 && !trace_->replaying()) {
        std::lock_guard lock(trace_mutex_);
        if (trace_->recording()) {
          trace_->RejectKeystroke();
          REXLOG_ERROR("Input trace recording failed: {}", trace_->error());
        }
      }
      out_keystroke->user_index = static_cast<uint8_t>(user_index);
      return result;
    }
    if (result != X_ERROR_DEVICE_NOT_CONNECTED) {
      any_connected = true;
    }
  }
  return any_connected ? X_ERROR_EMPTY : X_ERROR_DEVICE_NOT_CONNECTED;
}

std::unique_ptr<InputSystem> CreateDefaultInputSystem(bool tool_mode) {
  auto input = std::make_unique<InputSystem>(nullptr);

  if (!tool_mode) {
    if (REXCVAR_GET(test_startup_input)) {
      auto startup_driver = std::make_unique<startup::StartupInputDriver>(
          REXCVAR_GET(startup_input_sequence), REXCVAR_GET(startup_input_cutoff_poll));
      if (startup_driver->Setup() == X_STATUS_SUCCESS) {
        input->AddDriver(std::move(startup_driver));
      }
    }
#if REX_PLATFORM_WIN32
    if (REXCVAR_GET(input_backend) == "xinput") {
      auto xinput_driver = std::make_unique<xinput::XinputInputDriver>(nullptr, 0);
      if (xinput_driver->Setup() == X_STATUS_SUCCESS) {
        input->AddDriver(std::move(xinput_driver));
      }
    }
#endif

    if (REXCVAR_GET(input_backend) == "sdl") {
      auto sdl_driver = std::make_unique<sdl::SDLInputDriver>(nullptr, 0);
      if (sdl_driver->Setup() == X_STATUS_SUCCESS) {
        input->AddDriver(std::move(sdl_driver));
      }
    }

    // MnK driver (keyboard/mouse -> controller emulation)
    auto mnk_driver = std::make_unique<mnk::MnkInputDriver>(nullptr, 0);
    if (mnk_driver->Setup() == X_STATUS_SUCCESS) {
      input->AddDriver(std::move(mnk_driver));
    }
  }

  // NOP driver (primary in tool mode, fallback otherwise)
  uint8_t nop_index = tool_mode ? 0 : 1;
  input->AddDriver(std::make_unique<nop::NopInputDriver>(nullptr, nop_index));
  input->SetDeviceAssignment(std::make_unique<SlotAssignment>());
  return input;
}

}  // namespace rex::input
