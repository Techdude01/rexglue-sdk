// Debug-only fused-vs-legacy resolve comparison. The guest-visible destination
// is always produced by the legacy path while this capture is enabled.
#include <cstring>
#include <limits>

#include <rex/cvar.h>
#include <rex/graphics/flags.h>
#include <rex/graphics/vulkan/command_processor.h>
#include <rex/graphics/vulkan/deferred_command_buffer.h>
#include <rex/graphics/vulkan/texture_cache.h>
#include <rex/logging.h>
#include <rex/ui/vulkan/util.h>

namespace rex::graphics::vulkan {

size_t VulkanTextureCache::BeginDirectResolveParity(uint64_t bytes, uint32_t msaa_class,
                                                    VkBuffer& candidate_buffer_out) {
  candidate_buffer_out = VK_NULL_HANDLE;
  const uint32_t limit = uint32_t(REXCVAR_GET(gpu_direct_host_resolve_parity_resolves));
  if (!limit || direct_resolve_parity_requested_ >= limit || msaa_class > 1 || !bytes ||
      bytes > std::numeric_limits<VkDeviceSize>::max() / 2 ||
      !command_processor_.IsResolveChainAuditFrame()) {
    return SIZE_MAX;
  }
  const uint32_t msaa_bit = 1u << msaa_class;
  if (REXCVAR_GET(gpu_fused_resolve_image_parity) &&
      (bytes != 245760u * 4u || msaa_class != 0)) {
    return SIZE_MAX;
  }
  if (!REXCVAR_GET(gpu_fused_resolve_image_parity) &&
      (direct_resolve_parity_msaa_mask_ & msaa_bit)) {
    return SIZE_MAX;
  }

  DirectResolveParityReadback job;
  job.bytes = bytes;
  job.frame = command_processor_.GetCurrentFrame();
  job.msaa_class = msaa_class;

  VkBufferCreateInfo candidate_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  candidate_info.size = bytes;
  candidate_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                         VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
  candidate_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo candidate_allocation{};
  candidate_allocation.usage = VMA_MEMORY_USAGE_AUTO;
  if (vmaCreateBuffer(vma_allocator_, &candidate_info, &candidate_allocation, &job.candidate_buffer,
                      &job.candidate_allocation, nullptr) != VK_SUCCESS) {
    REXGPU_ERROR(
        "[DIRECT_HOST_RESOLVE_PARITY_ERROR] reason=candidate_allocation_failed frame={} bytes={}",
        job.frame, bytes);
    return SIZE_MAX;
  }

  VkBufferCreateInfo readback_info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
  readback_info.size = bytes * 2;
  readback_info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
  readback_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
  VmaAllocationCreateInfo readback_allocation{};
  readback_allocation.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
  readback_allocation.flags =
      VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
  VmaAllocationInfo allocated{};
  if (vmaCreateBuffer(vma_allocator_, &readback_info, &readback_allocation, &job.readback_buffer,
                      &job.readback_allocation, &allocated) != VK_SUCCESS ||
      !allocated.pMappedData) {
    REXGPU_ERROR(
        "[DIRECT_HOST_RESOLVE_PARITY_ERROR] reason=readback_allocation_failed frame={} bytes={}",
        job.frame, bytes);
    if (job.candidate_buffer != VK_NULL_HANDLE) {
      vmaDestroyBuffer(vma_allocator_, job.candidate_buffer, job.candidate_allocation);
    }
    if (job.readback_buffer != VK_NULL_HANDLE) {
      vmaDestroyBuffer(vma_allocator_, job.readback_buffer, job.readback_allocation);
    }
    return SIZE_MAX;
  }
  job.mapping = allocated.pMappedData;
  std::memset(job.mapping, 0xA5, size_t(bytes));
  std::memset(static_cast<uint8_t*>(job.mapping) + bytes, 0x5A, size_t(bytes));
  if (vmaFlushAllocation(vma_allocator_, job.readback_allocation, 0, VK_WHOLE_SIZE) != VK_SUCCESS) {
    REXGPU_ERROR("[DIRECT_HOST_RESOLVE_PARITY_ERROR] reason=canary_flush_failed frame={}",
                 job.frame);
    vmaDestroyBuffer(vma_allocator_, job.candidate_buffer, job.candidate_allocation);
    vmaDestroyBuffer(vma_allocator_, job.readback_buffer, job.readback_allocation);
    return SIZE_MAX;
  }

  const size_t index = direct_resolve_parity_readbacks_.size();
  direct_resolve_parity_readbacks_.push_back(job);
  ++direct_resolve_parity_requested_;
  direct_resolve_parity_msaa_mask_ |= msaa_bit;
  candidate_buffer_out = job.candidate_buffer;
  return index;
}

void VulkanTextureCache::CompleteDirectResolveParity(size_t job_index,
                                                     VkDeviceSize reference_offset) {
  if (job_index >= direct_resolve_parity_readbacks_.size()) {
    return;
  }
  DirectResolveParityReadback& job = direct_resolve_parity_readbacks_[job_index];
  const VkDeviceSize bytes = VkDeviceSize(job.bytes);
  const VkBuffer reference_buffer = scaled_resolve_buffer_;
  if (!reference_buffer || reference_offset > scaled_resolve_buffer_size_ ||
      bytes > scaled_resolve_buffer_size_ - reference_offset) {
    REXGPU_ERROR(
        "[DIRECT_HOST_RESOLVE_PARITY_ERROR] reason=reference_range_invalid frame={} offset={} "
        "bytes={}",
        job.frame, reference_offset, bytes);
    return;
  }

  command_processor_.PushBufferMemoryBarrier(
      job.candidate_buffer, 0, bytes, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
  command_processor_.PushBufferMemoryBarrier(
      reference_buffer, reference_offset, bytes, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
      VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, false);
  command_processor_.SubmitBarriers(true);

  VkBufferCopy regions[2]{};
  regions[0].srcOffset = 0;
  regions[0].dstOffset = 0;
  regions[0].size = bytes;
  regions[1].srcOffset = reference_offset;
  regions[1].dstOffset = bytes;
  regions[1].size = bytes;
  command_processor_.deferred_command_buffer().CmdVkCopyBuffer(job.candidate_buffer,
                                                               job.readback_buffer, 1, &regions[0]);
  command_processor_.deferred_command_buffer().CmdVkCopyBuffer(reference_buffer,
                                                               job.readback_buffer, 1, &regions[1]);
  command_processor_.PushBufferMemoryBarrier(
      job.readback_buffer, 0, bytes * 2, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT, VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED, false);
  command_processor_.PushBufferMemoryBarrier(
      reference_buffer, reference_offset, bytes, VK_PIPELINE_STAGE_TRANSFER_BIT,
      VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_TRANSFER_READ_BIT,
      VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT, VK_QUEUE_FAMILY_IGNORED,
      VK_QUEUE_FAMILY_IGNORED, false);
  command_processor_.SubmitBarriers(true);
  job.submission = command_processor_.GetCurrentSubmission();
  job.complete = true;
}

void VulkanTextureCache::ProcessDirectResolveParityReadbacks() {
  auto it = direct_resolve_parity_readbacks_.begin();
  while (it != direct_resolve_parity_readbacks_.end()) {
    DirectResolveParityReadback& job = *it;
    if (job.submission > command_processor_.GetCompletedSubmission()) {
      ++it;
      continue;
    }
    if (!job.complete || !job.mapping ||
        vmaInvalidateAllocation(vma_allocator_, job.readback_allocation, 0, VK_WHOLE_SIZE) !=
            VK_SUCCESS) {
      REXGPU_ERROR(
          "[DIRECT_HOST_RESOLVE_PARITY_ERROR] reason=incomplete_or_unreadable frame={} "
          "msaa_class={}",
          job.frame, job.msaa_class);
    } else {
      const auto* candidate = static_cast<const uint8_t*>(job.mapping);
      const auto* reference = candidate + job.bytes;
      uint64_t mismatches = 0;
      uint64_t first_mismatch = UINT64_MAX;
      if (std::memcmp(candidate, reference, size_t(job.bytes))) {
        for (uint64_t i = 0; i < job.bytes; ++i) {
          if (candidate[i] != reference[i]) {
            if (!mismatches)
              first_mismatch = i;
            ++mismatches;
          }
        }
      }
      REXGPU_INFO(
          "[DIRECT_HOST_RESOLVE_PARITY] frame={} msaa_class={} compared_bytes={} "
          "mismatched_bytes={} first_mismatch={} completed_submission={}",
          job.frame, job.msaa_class, job.bytes, mismatches, first_mismatch,
          command_processor_.GetCompletedSubmission());
      if (first_mismatch != UINT64_MAX) {
        const uint64_t context_start = first_mismatch > 8 ? first_mismatch - 8 : 0;
        const uint64_t context_end = first_mismatch + 24;
        for (uint64_t i = context_start; i < job.bytes && i < context_end; ++i) {
          REXGPU_INFO(
              "[DIRECT_HOST_RESOLVE_PARITY_BYTE] frame={} offset={} candidate=0x{:02X} "
              "reference=0x{:02X}",
              job.frame, i, uint32_t(candidate[i]), uint32_t(reference[i]));
        }
      }
    }
    vmaDestroyBuffer(vma_allocator_, job.candidate_buffer, job.candidate_allocation);
    vmaDestroyBuffer(vma_allocator_, job.readback_buffer, job.readback_allocation);
    it = direct_resolve_parity_readbacks_.erase(it);
  }
}

}  // namespace rex::graphics::vulkan
