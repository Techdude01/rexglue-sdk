#pragma once

#include <utility>

namespace rex::ui::vulkan {

// Retry once with a fresh runtime only when KosmicKrisp was requested. The
// factory must destroy failed instance/device state before returning failure.
// Explicit MoltenVK selection is never retried against a different driver.
template <typename Factory, typename LogFallback>
auto CreateWithMacOSDriverFallback(bool request_kosmickrisp, bool manifest_available,
                                  Factory&& create, LogFallback&& log_fallback) {
  if (!request_kosmickrisp) return create(true);
  if (!manifest_available) {
    log_fallback("manifest unavailable");
    return create(true);
  }
  auto result = create(false);
  if (result) return result;
  log_fallback("instance/device initialization failed");
  return create(true);
}

}  // namespace rex::ui::vulkan
