/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2021 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Rien Gupta, 2026 - Adapted for ReXGlue runtime
 */

#include <rex/ui/surface_mac.h>

#include <CoreGraphics/CGGeometry.h>
#include <objc/message.h>
#include <objc/runtime.h>
#include <SDL3/SDL_metal.h>
#include <SDL3/SDL_video.h>

namespace rex {
namespace ui {

CAMetalLayerSurface::~CAMetalLayerSurface() {
  if (metal_view_) {
    SDL_Metal_DestroyView(static_cast<SDL_MetalView>(metal_view_));
    metal_view_ = nullptr;
  }
}

bool CAMetalLayerSurface::GetSizeImpl(uint32_t& width_out, uint32_t& height_out) const {
  // Size the swapchain for the layer's drawableSize when it is valid: that is
  // what MoltenVK presents to, and SDL's window pixel size can race ahead of
  // the layer while resizing or migrating displays (a swapchain built for the
  // racing size then presents suboptimally forever). Fall back to the SDL size
  // while the layer has no valid drawable yet (startup, minimization).
  if (layer_) {
    SEL sel = sel_registerName("drawableSize");
    using MsgSendSize = CGSize (*)(const void*, SEL);
    MsgSendSize send = reinterpret_cast<MsgSendSize>(objc_msgSend);
    CGSize drawable = send(layer_, sel);
    if (drawable.width >= 2 && drawable.height >= 2) {
      width_out = uint32_t(drawable.width);
      height_out = uint32_t(drawable.height);
      return true;
    }
  }
  int pixel_width = 0;
  int pixel_height = 0;
  if (!SDL_GetWindowSizeInPixels(window_, &pixel_width, &pixel_height)) {
    return false;
  }
  width_out = uint32_t(pixel_width);
  height_out = uint32_t(pixel_height);
  return true;
}

}  // namespace ui
}  // namespace rex
