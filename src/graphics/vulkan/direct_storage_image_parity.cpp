// Debug-only same-source GPU equivalence check. Default budget is zero.
#include <algorithm>
#include <cstring>
#include <rex/cvar.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/texture_cache.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/util.h>

namespace rex::graphics::vulkan {
size_t VulkanTextureCache::BeginParityReadback(Texture& texture, bool load_base, bool load_mips) {
  const uint32_t limit = uint32_t(REXCVAR_GET(gpu_direct_storage_image_parity_loads));
  if (parity_reentry_ || parity_requested_ >= limit ||
      !command_processor_.IsResolveChainAuditFrame()) return SIZE_MAX;
  auto& vk_texture = static_cast<VulkanTexture&>(texture);
  ParityReadback job;
  job.width = texture.key().GetWidth() * draw_resolution_scale_x();
  job.height = texture.key().GetHeight() * draw_resolution_scale_y();
  job.bytes = uint64_t(job.width) * job.height * 4;
  job.frame = command_processor_.GetCurrentFrame();
  job.format = texture.key().format == xenos::TextureFormat::k_24_8
      ? VK_FORMAT_R32_SFLOAT : VK_FORMAT_R8G8B8A8_UNORM;
  VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  info.size = job.bytes * 2;
  info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo allocation{};
  allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  allocation.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VmaAllocationInfo allocated{};
  if (vmaCreateBuffer(vma_allocator_, &info, &allocation, &job.buffer, &job.allocation, &allocated) != VK_SUCCESS) {
    REXGPU_ERROR("[DIRECT_IMAGE_PARITY_ERROR] reason=allocation_failed frame={}", job.frame);
    return SIZE_MAX;
  }
  job.mapping = allocated.pMappedData;
  // A missing/no-op readback must fail, not compare two zero-initialized halves.
  if (!job.mapping) {
    REXGPU_ERROR("[DIRECT_IMAGE_PARITY_ERROR] reason=unmapped_allocation frame={}", job.frame);
    vmaDestroyBuffer(vma_allocator_, job.buffer, job.allocation);
    return SIZE_MAX;
  }
  std::memset(job.mapping, 0xA5, size_t(job.bytes));
  std::memset(static_cast<uint8_t*>(job.mapping) + job.bytes, 0x5A, size_t(job.bytes));
  if (vmaFlushAllocation(vma_allocator_, job.allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS) {
    REXGPU_ERROR("[DIRECT_IMAGE_PARITY_ERROR] reason=canary_flush_failed frame={}", job.frame);
    vmaDestroyBuffer(vma_allocator_, job.buffer, job.allocation);
    return SIZE_MAX;
  }
  job.submission = command_processor_.GetCurrentSubmission();
  const size_t index = parity_readbacks_.size();
  parity_readbacks_.push_back(job);
  ++parity_requested_;
  // The original backend writes this very image through its unchanged scratch
  // buffer -> image path. No guest work or source write can interleave here.
  parity_reentry_ = true;
  const bool ok = LoadTextureDataFromResidentMemoryImpl(texture, load_base, load_mips);
  parity_reentry_ = false;
  if (!ok) {
    REXGPU_ERROR("[DIRECT_IMAGE_PARITY_ERROR] reason=legacy_load_failed frame={}", job.frame);
    return SIZE_MAX; // Allocation stays alive until its submitted work completes.
  }
  CaptureParityImage(vk_texture, parity_readbacks_[index], false);
  return index;
}

void VulkanTextureCache::CaptureParityImage(VulkanTexture& texture, ParityReadback& job, bool direct) {
  VkPipelineStageFlags src_stage, dst_stage;
  VkAccessFlags src_access, dst_access;
  VkImageLayout old_layout, new_layout;
  const auto old = texture.SetUsage(VulkanTexture::Usage::kParityTransferSource);
  GetTextureUsageMasks(old, src_stage, src_access, old_layout);
  GetTextureUsageMasks(VulkanTexture::Usage::kParityTransferSource, dst_stage, dst_access, new_layout);
  command_processor_.PushImageMemoryBarrier(texture.image(),
      ui::vulkan::util::InitializeSubresourceRange(), src_stage, dst_stage,
      src_access, dst_access, old_layout, new_layout);
  command_processor_.SubmitBarriers(true);
  VkBufferImageCopy region{};
  region.bufferOffset = direct ? job.bytes : 0;
  region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.layerCount = 1;
  region.imageExtent = {job.width, job.height, 1};
  command_processor_.deferred_command_buffer().CmdParityCopyImageToBuffer(texture.image(), job.buffer, region);
  if (direct) {
    command_processor_.PushBufferMemoryBarrier(job.buffer, 0, job.bytes * 2,
        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
        VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
    command_processor_.SubmitBarriers(true);
    job.submission = command_processor_.GetCurrentSubmission();
    job.complete = true;
  }
}

void VulkanTextureCache::ProcessParityReadbacks() {
  auto it = parity_readbacks_.begin();
  while (it != parity_readbacks_.end()) {
    auto& job = *it;
    if (job.submission > command_processor_.GetCompletedSubmission()) { ++it; continue; }
    if (!job.complete || !job.mapping ||
        vmaInvalidateAllocation(vma_allocator_, job.allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS) {
      REXGPU_ERROR("[DIRECT_IMAGE_PARITY_ERROR] reason=incomplete_or_unreadable frame={} format={}", job.frame, job.format);
    } else {
      const auto* a = static_cast<const uint8_t*>(job.mapping);
      const auto* b = a + job.bytes;
      uint64_t mismatches = 0, first = UINT64_MAX;
      if (std::memcmp(a, b, size_t(job.bytes))) {
        for (uint64_t i = 0; i < job.bytes; ++i) if (a[i] != b[i]) {
          if (!mismatches) first = i;
          ++mismatches;
        }
      }
      REXGPU_INFO("[DIRECT_IMAGE_PARITY] frame={} format={} width={} height={} compared_bytes={} mismatched_bytes={} first_mismatch={} completed_submission={}",
          job.frame, job.format, job.width, job.height, job.bytes, mismatches, first,
          command_processor_.GetCompletedSubmission());
    }
    vmaDestroyBuffer(vma_allocator_, job.buffer, job.allocation);
    it = parity_readbacks_.erase(it);
  }
}
} // namespace rex::graphics::vulkan
