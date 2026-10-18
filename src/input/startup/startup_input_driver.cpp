// SPDX-License-Identifier: BSD-3-Clause
#include <rex/input/startup/startup_input_driver.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string_view>
#include <utility>

#include <rex/input/flags.h>
#include <rex/logging.h>
#include <rex/ui/virtual_key.h>

namespace rex::input::startup {

namespace {
constexpr DeviceId kStartupDevice = static_cast<DeviceId>(0x5354415254555001ull);

bool ParseNumber(std::string_view text, uint32_t& value) {
  if (text.empty())
    return false;
  uint64_t parsed = 0;
  for (char c : text) {
    if (c < '0' || c > '9')
      return false;
    parsed = parsed * 10 + static_cast<uint32_t>(c - '0');
    if (parsed > UINT32_MAX)
      return false;
  }
  value = static_cast<uint32_t>(parsed);
  return true;
}

bool ParseSequence(std::string_view text, std::vector<StartupInputDriver::Pulse>& pulses) {
  while (!text.empty()) {
    const size_t comma = text.find(',');
    const std::string_view item = text.substr(0, comma);
    const size_t at = item.find('@');
    const size_t colon = item.find(':');
    if (at == std::string_view::npos || colon == std::string_view::npos || colon <= at) {
      return false;
    }
    std::string button(item.substr(0, at));
    std::transform(button.begin(), button.end(), button.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    uint16_t mask = 0;
    int16_t thumb_lx = 0, thumb_ly = 0;
    int16_t thumb_rx = 0, thumb_ry = 0;
    if (button == "a")
      mask = X_INPUT_GAMEPAD_A;
    else if (button == "start")
      mask = X_INPUT_GAMEPAD_START;
    else if (button == "up")
      thumb_ly = 24576;
    else if (button == "down")
      thumb_ly = -24576;
    else if (button == "left")
      thumb_lx = -24576;
    else if (button == "right")
      thumb_lx = 24576;
    else if (button == "camera_up")
      thumb_ry = 24576;
    else if (button == "camera_down")
      thumb_ry = -24576;
    else if (button == "camera_left")
      thumb_rx = -24576;
    else if (button == "camera_right")
      thumb_rx = 24576;
    else
      return false;
    uint32_t first = 0, count = 0;
    if (!ParseNumber(item.substr(at + 1, colon - at - 1), first) ||
        !ParseNumber(item.substr(colon + 1), count) || count == 0 ||
        (mask ? count > 4 : count > 60)) {
      return false;
    }
    pulses.push_back({first, count, mask, thumb_lx, thumb_ly, thumb_rx, thumb_ry});
    if (pulses.size() > 32)
      return false;
    if (comma == std::string_view::npos)
      break;
    text.remove_prefix(comma + 1);
  }
  return !pulses.empty();
}
}  // namespace

StartupInputDriver::StartupInputDriver(std::string sequence, uint32_t cutoff_poll)
    : InputDriver(nullptr, 0), cutoff_poll_(cutoff_poll) {
  if (!ParseSequence(sequence, pulses_)) {
    pulses_.clear();
    REXLOG_ERROR(
        "Startup input disabled: invalid sequence (buttons a/start: 1-4 polls; stick directions up/down/left/right, camera_up/down/left/right: 1-60 polls): '{}'",
        sequence);
  }
  std::stable_sort(pulses_.begin(), pulses_.end(),
                   [](const Pulse& a, const Pulse& b) { return a.first_poll < b.first_poll; });
}

X_STATUS StartupInputDriver::Setup() {
  if (pulses_.empty()) {
    return X_STATUS_UNSUCCESSFUL;
  }
  REXLOG_INFO("Startup input enabled: synthetic XInput controller, cutoff poll={}, pulses={}",
              cutoff_poll_, pulses_.size());
  return X_STATUS_SUCCESS;
}

void StartupInputDriver::EnumerateDevices(std::vector<DeviceInfo>& out) {
  out.push_back({kStartupDevice, 0, "Test startup controller", "", true});
}

X_RESULT StartupInputDriver::GetDeviceState(DeviceId id, X_INPUT_STATE* out_state) {
  if (id != kStartupDevice)
    return X_ERROR_DEVICE_NOT_CONNECTED;
  ++poll_count_;
  uint16_t buttons = 0;
  int16_t thumb_lx = 0, thumb_ly = 0;
  int16_t thumb_rx = 0, thumb_ry = 0;
  if (poll_count_ <= cutoff_poll_) {
    for (const Pulse& pulse : pulses_) {
      if (poll_count_ >= pulse.first_poll && poll_count_ - pulse.first_poll < pulse.poll_count) {
        buttons |= pulse.button;
        thumb_lx = pulse.thumb_lx;
        thumb_ly = pulse.thumb_ly;
        thumb_rx = pulse.thumb_rx;
        thumb_ry = pulse.thumb_ry;
      }
    }
  }
  if (buttons != last_buttons_ || thumb_lx != last_thumb_lx_ || thumb_ly != last_thumb_ly_ ||
      thumb_rx != last_thumb_rx_ || thumb_ry != last_thumb_ry_) {
    ++packet_number_;
    REXLOG_INFO(
        "Startup input XInput poll {}: buttons=0x{:04X} thumb_lx={} thumb_ly={} thumb_rx={} thumb_ry={}{}",
        poll_count_, buttons, thumb_lx, thumb_ly, thumb_rx, thumb_ry,
        buttons || thumb_lx || thumb_ly || thumb_rx || thumb_ry ? " active" : " released");
    last_buttons_ = buttons;
    last_thumb_lx_ = thumb_lx;
    last_thumb_ly_ = thumb_ly;
    last_thumb_rx_ = thumb_rx;
    last_thumb_ry_ = thumb_ry;
  }
  const uint16_t changed = buttons ^ state_buttons_;
  for (const auto [mask, key] :
       {std::pair<uint16_t, uint16_t>{X_INPUT_GAMEPAD_A,
                                      static_cast<uint16_t>(rex::ui::VirtualKey::kXInputPadA)},
        std::pair<uint16_t, uint16_t>{
            X_INPUT_GAMEPAD_START, static_cast<uint16_t>(rex::ui::VirtualKey::kXInputPadStart)}}) {
    if (!(changed & mask))
      continue;
    if (keystrokes_.size() < 32) {
      keystrokes_.push_back({key, static_cast<uint16_t>(buttons & mask ? X_INPUT_KEYSTROKE_KEYDOWN
                                                                       : X_INPUT_KEYSTROKE_KEYUP)});
    }
  }
  state_buttons_ = buttons;
  if (out_state) {
    std::memset(out_state, 0, sizeof(*out_state));
    out_state->packet_number = packet_number_;
    out_state->gamepad.buttons = buttons;
    out_state->gamepad.thumb_lx = thumb_lx;
    out_state->gamepad.thumb_ly = thumb_ly;
    out_state->gamepad.thumb_rx = thumb_rx;
    out_state->gamepad.thumb_ry = thumb_ry;
  }
  return X_ERROR_SUCCESS;
}

X_RESULT StartupInputDriver::GetDeviceCapabilities(DeviceId id, uint32_t,
                                                   X_INPUT_CAPABILITIES* out_caps) {
  if (id != kStartupDevice)
    return X_ERROR_DEVICE_NOT_CONNECTED;
  if (out_caps) {
    std::memset(out_caps, 0, sizeof(*out_caps));
    out_caps->type = XINPUT_DEVTYPE_GAMEPAD;
    out_caps->sub_type = 0x01;
    out_caps->gamepad.buttons = X_INPUT_GAMEPAD_A | X_INPUT_GAMEPAD_START;
    out_caps->gamepad.thumb_lx = static_cast<int16_t>(0x7FFF);
    out_caps->gamepad.thumb_ly = static_cast<int16_t>(0x7FFF);
  }
  return X_ERROR_SUCCESS;
}

X_RESULT StartupInputDriver::SetDeviceVibration(DeviceId id, X_INPUT_VIBRATION*) {
  return id == kStartupDevice ? X_ERROR_SUCCESS : X_ERROR_DEVICE_NOT_CONNECTED;
}

X_RESULT StartupInputDriver::GetDeviceKeystroke(DeviceId id, uint32_t,
                                                X_INPUT_KEYSTROKE* out_keystroke) {
  if (id != kStartupDevice)
    return X_ERROR_DEVICE_NOT_CONNECTED;
  if (!out_keystroke)
    return X_ERROR_BAD_ARGUMENTS;
  if (keystrokes_.empty())
    return X_ERROR_EMPTY;
  const Keystroke event = keystrokes_.front();
  keystrokes_.pop_front();
  out_keystroke->virtual_key = event.virtual_key;
  out_keystroke->unicode = 0;
  out_keystroke->flags = event.flags;
  out_keystroke->user_index = 0;
  out_keystroke->hid_code = 0;
  REXLOG_INFO("Startup input guest keystroke virtual_key=0x{:04X} flags=0x{:04X}",
              event.virtual_key, event.flags);
  return X_ERROR_SUCCESS;
}

}  // namespace rex::input::startup
