// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <rex/graphics/register_file.h>
#include <rex/logging.h>

namespace rex::graphics::detail {

// Register metadata is diagnostic only; storage and register side effects stay
// in CommandProcessor::WriteRegister. Re-check the logger's live level so a
// runtime debug-level change still enables unknown-register diagnostics.
inline void LogRegisterWriteDiagnostic(RegisterFile& registers, uint32_t index,
                                       uint32_t value, bool lazy_metadata) {
  if (lazy_metadata) {
    auto* logger = rex::GetLoggerRaw(rex::log::gpu());
    if (!logger || !logger->should_log(spdlog::level::debug)) return;
  }
  if (!registers.GetRegisterInfo(index)) {
    REXGPU_DEBUG("GPU: Write to unknown register ({:04X} = {:08X})", index, value);
  }
}

}  // namespace rex::graphics::detail
