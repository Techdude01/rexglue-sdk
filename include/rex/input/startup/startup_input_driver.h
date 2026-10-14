// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

#include <rex/input/input_driver.h>

namespace rex::input::startup {

/// Test-only synthetic controller installed only by the explicit cvar.
class StartupInputDriver final : public InputDriver {
 public:
  struct Pulse {
    uint32_t first_poll;
    uint32_t poll_count;
    uint16_t button;
  };

  StartupInputDriver(std::string sequence, uint32_t cutoff_poll);
  X_STATUS Setup() override;
  void EnumerateDevices(std::vector<DeviceInfo>& out) override;
  X_RESULT GetDeviceState(DeviceId id, X_INPUT_STATE* out_state) override;
  X_RESULT GetDeviceCapabilities(DeviceId id, uint32_t flags,
                                 X_INPUT_CAPABILITIES* out_caps) override;
  X_RESULT SetDeviceVibration(DeviceId id, X_INPUT_VIBRATION* vibration) override;
  X_RESULT GetDeviceKeystroke(DeviceId id, uint32_t flags,
                              X_INPUT_KEYSTROKE* out_keystroke) override;

 private:
  struct Keystroke {
    uint16_t virtual_key;
    uint16_t flags;
  };

  std::vector<Pulse> pulses_;
  std::deque<Keystroke> keystrokes_;
  uint32_t cutoff_poll_;
  uint32_t poll_count_ = 0;
  uint32_t packet_number_ = 0;
  uint16_t last_buttons_ = 0;
  uint16_t state_buttons_ = 0;
};

}  // namespace rex::input::startup
