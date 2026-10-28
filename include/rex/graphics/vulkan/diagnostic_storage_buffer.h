#pragma once
/**
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

#include <rex/graphics/vulkan/fragment_diagnostic.h>
#include <rex/ui/vulkan/util.h>

namespace rex::graphics::vulkan {

// Host-owned, bounded fragment diagnostic transport. The buffer is deliberately
// independent of guest memory; callers must only consume it after its submission
// has completed (see ReadCompleted below).
class VulkanDiagnosticStorageBuffer {
 public:
  static constexpr uint32_t kMaxRecords = diagnostics::kFragmentDiagnosticMaxRecords;

  struct Header {
    uint32_t count;
    uint32_t overflow;
    uint32_t capacity;
    uint32_t valid;
    // Bit 0 arms; bit 1 is the invocation witness; bit 2 enables triples;
    // bit 3 filters to the local candidate ROI; bit 4 captures final RT0 output.
    uint32_t recording_enabled;
    uint32_t abi_version;
    uint32_t entry_invocation_count;
    uint32_t entry_nose_roi_count;
    uint32_t entry_min_x;
    uint32_t entry_min_y;
    uint32_t entry_max_x;
    uint32_t entry_max_y;
    uint32_t eligible_invocation_count;
    uint32_t eligible_nose_roi_count;
  };

  enum HeaderMember : uint32_t {
    kCountMember = 0,
    kOverflowMember = 1,
    kCapacityMember = 2,
    kValidMember = 3,
    kRecordingEnabledMember = 4,
    kAbiVersionMember = 5,
    kEntryInvocationCountMember = 6,
    kEntryNoseRoiCountMember = 7,
    kEntryMinXMember = 8,
    kEntryMinYMember = 9,
    kEntryMaxXMember = 10,
    kEntryMaxYMember = 11,
    kEligibleInvocationCountMember = 12,
    kEligibleNoseRoiCountMember = 13,
    kRecordsMember = 14,
  };

  // Keep these values in lockstep with the SPIR-V member decorations emitted by
  // SpirvShaderTranslator. Float fields are stored as raw IEEE-754 bits.
  enum RecordMember : uint32_t {
    kShaderHashLoOffset = 0,
    kShaderHashHiOffset = 4,
    kFetchSlotOffset = 8,
    kStageMarkerOffset = 12,
    kFragmentXOffset = 16,
    kFragmentYOffset = 20,
    kFrameIdOffset = 24,
    kDrawIdOffset = 28,
    kUvOffset = 32,
    kRgbaOffset = 40,
    kValidityOffset = 56,
    kOutputRgbaOffset = 60,
    kSampleMaskOffset = 76,
    kVertexInterpolator0Offset = 80,
    kVertexInterpolator1Offset = 96,
    kVertexInterpolator2Offset = 112,
    kVertexInterpolator3Offset = 128,
    kVertexFetchInputOffset = 144,
    kRecordSize = 496,
  };

  // Census mode uses only the header counters. Triple mode reserves contiguous
  // tf0/tf2/tf14 records per eligible invocation; index/3 is the group ID.
  struct Record {
    uint32_t shader_hash_lo;
    uint32_t shader_hash_hi;
    uint32_t fetch_slot;
    uint32_t stage_marker;
    uint32_t fragment_x;
    uint32_t fragment_y;
    uint32_t frame_id;  // Explicitly unknown until frame selection is implemented.
    uint32_t draw_id;   // Explicitly unknown until draw selection is implemented.
    uint32_t uv[2];
    uint32_t rgba[4];
    uint32_t validity;
    uint32_t output_rgba[4];  // Face PS output bits, populated only in ABI 6.
    uint32_t sample_mask;     // Rasterization coverage bits; diagnostic ABI 7.
    // Guest vertex shader outputs o0-o3 as float bit patterns, vertex-stage rows only.
    uint32_t vertex_interpolators[4][4];
    // Vertex records: exact decoded targeted vfetch results (position, normal,
    // tangent, UV, weights, indices, then 16 bone vectors). G-buffer pixel
    // records reuse the first 10 vectors for bounded PS input/intermediate
    // registers; only the third record in each fragment triple carries them.
    uint32_t vertex_fetch_inputs[22][4];
  };
  static constexpr uint32_t kAbiVersion = 9;
  static_assert(std::is_standard_layout_v<Header> && sizeof(Header) == 56);
  static_assert(offsetof(Header, count) == 0);
  static_assert(offsetof(Header, overflow) == 4);
  static_assert(offsetof(Header, capacity) == 8);
  static_assert(offsetof(Header, valid) == 12);
  static_assert(offsetof(Header, recording_enabled) == 16);
  static_assert(offsetof(Header, abi_version) == 20);
  static_assert(offsetof(Header, entry_invocation_count) == 24);
  static_assert(offsetof(Header, entry_nose_roi_count) == 28);
  static_assert(offsetof(Header, entry_min_x) == 32);
  static_assert(offsetof(Header, entry_min_y) == 36);
  static_assert(offsetof(Header, entry_max_x) == 40);
  static_assert(offsetof(Header, entry_max_y) == 44);
  static_assert(offsetof(Header, eligible_invocation_count) == 48);
  static_assert(offsetof(Header, eligible_nose_roi_count) == 52);
  static_assert(kRecordsMember == sizeof(Header) / sizeof(uint32_t));
  static_assert(std::is_standard_layout_v<Record> && sizeof(Record) == kRecordSize);
  static_assert(offsetof(Record, shader_hash_lo) == kShaderHashLoOffset);
  static_assert(offsetof(Record, shader_hash_hi) == kShaderHashHiOffset);
  static_assert(offsetof(Record, fetch_slot) == kFetchSlotOffset);
  static_assert(offsetof(Record, stage_marker) == kStageMarkerOffset);
  static_assert(offsetof(Record, fragment_x) == kFragmentXOffset);
  static_assert(offsetof(Record, fragment_y) == kFragmentYOffset);
  static_assert(offsetof(Record, frame_id) == kFrameIdOffset);
  static_assert(offsetof(Record, draw_id) == kDrawIdOffset);
  static_assert(offsetof(Record, uv) == kUvOffset);
  static_assert(offsetof(Record, rgba) == kRgbaOffset);
  static_assert(offsetof(Record, validity) == kValidityOffset);
  static_assert(offsetof(Record, output_rgba) == kOutputRgbaOffset);
  static_assert(offsetof(Record, sample_mask) == kSampleMaskOffset);
  static_assert(offsetof(Record, vertex_interpolators[0]) == kVertexInterpolator0Offset);
  static_assert(offsetof(Record, vertex_interpolators[1]) == kVertexInterpolator1Offset);
  static_assert(offsetof(Record, vertex_interpolators[2]) == kVertexInterpolator2Offset);
  static_assert(offsetof(Record, vertex_interpolators[3]) == kVertexInterpolator3Offset);
  static_assert(offsetof(Record, vertex_fetch_inputs) == kVertexFetchInputOffset);

  VulkanDiagnosticStorageBuffer() = default;
  VulkanDiagnosticStorageBuffer(const VulkanDiagnosticStorageBuffer&) = delete;
  VulkanDiagnosticStorageBuffer& operator=(const VulkanDiagnosticStorageBuffer&) = delete;
  ~VulkanDiagnosticStorageBuffer() { Reset(); }

  bool Initialize(const ui::vulkan::VulkanDevice* device, uint32_t capacity = kMaxRecords) {
    Reset();
    if (!device || !capacity || capacity > kMaxRecords) return false;
    device_ = device;
    capacity_ = capacity;
    const VkDeviceSize size = sizeof(Header) + VkDeviceSize(capacity_) * sizeof(Record);
    if (!ui::vulkan::util::CreateDedicatedAllocationBuffer(
            device, size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            ui::vulkan::util::MemoryPurpose::kReadback, buffer_, memory_, &memory_type_,
            &memory_size_)) {
      Reset();
      return false;
    }
    const auto& functions = device->functions();
    if (functions.vkMapMemory(device->device(), memory_, 0, VK_WHOLE_SIZE, 0,
                              reinterpret_cast<void**>(&mapping_)) != VK_SUCCESS) {
      Reset();
      return false;
    }
    if (!Clear()) {
      Reset();
      return false;
    }
    return true;
  }

  void Reset() {
    if (!device_) {
      buffer_ = VK_NULL_HANDLE;
      memory_ = VK_NULL_HANDLE;
      mapping_ = nullptr;
      capacity_ = 0;
      return;
    }
    const auto& functions = device_->functions();
    if (mapping_) functions.vkUnmapMemory(device_->device(), memory_);
    if (buffer_ != VK_NULL_HANDLE) functions.vkDestroyBuffer(device_->device(), buffer_, nullptr);
    if (memory_ != VK_NULL_HANDLE) functions.vkFreeMemory(device_->device(), memory_, nullptr);
    device_ = nullptr;
    buffer_ = VK_NULL_HANDLE;
    memory_ = VK_NULL_HANDLE;
    mapping_ = nullptr;
    capacity_ = 0;
  }

  // Call only before GPU writes for this buffer have been submitted.
  bool Clear() {
    if (!mapping_) return false;
    std::memset(mapping_, 0, size());
    Header* header = reinterpret_cast<Header*>(mapping_);
    header->capacity = capacity_;
    header->recording_enabled = 0;
    header->abi_version = kAbiVersion;
    header->entry_min_x = UINT32_MAX;
    header->entry_min_y = UINT32_MAX;
    // The utility aligns non-coherent ranges to nonCoherentAtomSize and clamps
    // the final range to the allocation size.
    ui::vulkan::util::FlushMappedMemoryRange(device_, memory_, memory_type_, 0,
                                             memory_size_, size());
    return true;
  }

  VkBuffer buffer() const { return buffer_; }
  VkDeviceSize size() const { return sizeof(Header) + VkDeviceSize(capacity_) * sizeof(Record); }
  uint32_t capacity() const { return capacity_; }

  // Host-controlled gate; only change between completed submissions. Recording
  // remains disabled until a later frame-selection slice explicitly enables it.
  bool SetRecordingEnabled(bool enabled, bool record_triples = true,
                           bool local_candidate_roi = false, bool capture_output = false) {
    if (!mapping_) return false;
    Header* header = reinterpret_cast<Header*>(mapping_);
    header->recording_enabled = enabled ? (1u | (record_triples ? 4u : 0u) |
                                           (local_candidate_roi ? 8u : 0u) |
                                           (capture_output ? 16u : 0u))
                                       : 0u;
    ui::vulkan::util::FlushMappedMemoryRange(device_, memory_, memory_type_, 0,
                                             memory_size_, sizeof(Header));
    return true;
  }

  // The submission-completion condition is explicit at the callsite, preventing
  // accidental reads while the GPU may still be writing.
  bool ReadCompleted(bool submission_completed, std::vector<Record>& records,
                     bool& overflow_out, uint32_t* count_out = nullptr,
                     Header* header_out = nullptr,
                     bool entry_counters_recorded = true) const {
    records.clear();
    overflow_out = false;
    if (count_out) *count_out = 0;
    if (header_out) *header_out = {};
    if (!mapping_ || !submission_completed) return false;
    VkMappedMemoryRange range{};
    range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    range.memory = memory_;
    range.offset = 0;
    range.size = VK_WHOLE_SIZE;
    if (device_->functions().vkInvalidateMappedMemoryRanges(device_->device(), 1, &range) !=
        VK_SUCCESS) {
      return false;
    }
    const Header* header = reinterpret_cast<const Header*>(mapping_);
    if (header_out) *header_out = *header;
    if (header->capacity != capacity_ || header->abi_version != kAbiVersion) return false;
    const uint32_t count = header->count;
    if (count_out) *count_out = count;
    overflow_out = header->overflow != 0;
    if (!diagnostics::FragmentDiagnosticCountersValid(
            header->entry_invocation_count, header->entry_nose_roi_count,
            header->eligible_invocation_count, header->eligible_nose_roi_count,
            entry_counters_recorded)) {
      return false;
    }
    // `valid` is published by the shader after observing a census invocation or
    // storing its first complete triple. An untouched buffer is a meaningful
    // zero-invocation result, not an invalid readback.
    if (!header->valid && !count) return true;
    if (!header->valid) return false;
    // A truncated or overflowing capture is not a valid evidence artifact.
    if (count > capacity_ || overflow_out) return false;
    const Record* source = reinterpret_cast<const Record*>(mapping_ + sizeof(Header));
    for (uint32_t i = 0; i < count; ++i) {
      if (source[i].validity != 1) return false;
    }
    records.assign(source, source + count);
    return true;
  }

 private:
  const ui::vulkan::VulkanDevice* device_ = nullptr;
  VkBuffer buffer_ = VK_NULL_HANDLE;
  VkDeviceMemory memory_ = VK_NULL_HANDLE;
  uint8_t* mapping_ = nullptr;
  uint32_t memory_type_ = 0;
  VkDeviceSize memory_size_ = 0;
  uint32_t capacity_ = 0;
};

}  // namespace rex::graphics::vulkan
