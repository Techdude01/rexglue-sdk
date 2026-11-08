#pragma once
/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2022 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#include <array>
#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rex/graphics/pipeline/texture/cache.h>
#include <rex/graphics/vulkan/shader.h>
#include <rex/graphics/vulkan/direct_storage_image.h>
#include <rex/graphics/vulkan/fragment_diagnostic.h>
#include <rex/graphics/vulkan/shared_memory.h>
#include <rex/graphics/vulkan/resolve_chain_audit.h>
#include <rex/graphics/vulkan/resolved_image_ownership.h>
#include <rex/hash.h>
#include <rex/ui/vulkan/mem_alloc.h>

namespace rex::graphics::draw_util { struct ResolveInfo; }

namespace rex::graphics::vulkan {

class VulkanCommandProcessor;

class VulkanTextureCache final : public TextureCache {
 public:
  // Sampler parameters that can be directly converted to a host sampler or used
  // for checking whether samplers bindings are up to date.
  union SamplerParameters {
    uint32_t value;
    struct {
      xenos::ClampMode clamp_x : 3;         // 3
      xenos::ClampMode clamp_y : 3;         // 6
      xenos::ClampMode clamp_z : 3;         // 9
      xenos::BorderColor border_color : 2;  // 11
      uint32_t mag_linear : 1;              // 12
      uint32_t min_linear : 1;              // 13
      uint32_t mip_linear : 1;              // 14
      xenos::AnisoFilter aniso_filter : 3;  // 17
      uint32_t mip_min_level : 4;           // 21
      uint32_t mip_base_map : 1;            // 22
      // Maximum mip level is in the texture resource itself, but mip_base_map
      // can be used to limit fetching to mip_min_level.
    };

    SamplerParameters() : value(0) { static_assert_size(*this, sizeof(value)); }
    explicit SamplerParameters(uint32_t v) : value(v) {}
    struct Hasher {
      size_t operator()(const SamplerParameters& parameters) const {
        return std::hash<uint32_t>{}(parameters.value);
      }
    };
    bool operator==(const SamplerParameters& parameters) const { return value == parameters.value; }
    bool operator!=(const SamplerParameters& parameters) const { return value != parameters.value; }
  };

  // Transient descriptor set layouts must be initialized in the command
  // processor.
  static std::unique_ptr<VulkanTextureCache> Create(
      const RegisterFile& register_file, VulkanSharedMemory& shared_memory,
      uint32_t draw_resolution_scale_x, uint32_t draw_resolution_scale_y,
      VulkanCommandProcessor& command_processor,
      VkPipelineStageFlags guest_shader_pipeline_stages) {
    std::unique_ptr<VulkanTextureCache> texture_cache(new VulkanTextureCache(
        register_file, shared_memory, draw_resolution_scale_x, draw_resolution_scale_y,
        command_processor, guest_shader_pipeline_stages));
    if (!texture_cache->Initialize()) {
      return nullptr;
    }
    return std::move(texture_cache);
  }

  ~VulkanTextureCache();

  struct FusedResolveImage {
    VkImage image = VK_NULL_HANDLE;
    VkDescriptorSet descriptor = VK_NULL_HANDLE;
    uint32_t width = 0;
    uint32_t height = 0;
  };
  FusedResolveImage PrepareFusedResolveImage(uint32_t base, uint32_t length,
                                            uint32_t width, uint32_t height,
                                            xenos::Endian endian);
  void PublishFusedResolveImage();
  void CancelFusedResolveImage() { pending_fused_resolve_image_.reset(); }

  void ClearCache() override;
  void BeginSubmission(uint64_t new_submission_index) override;
  void BeginFrame() override;
  void EndFrame();

  // Must be called within a frame - creates and untiles textures needed by
  // shaders, and enqueues transitioning them into the sampled usage. This may
  // bind compute pipelines (notifying the command processor about that), and
  // also since it may insert deferred barriers, before flushing the barriers
  // preceding host GPU work.
  void RequestTextures(uint32_t used_texture_mask) override;

  VkImageView GetActiveBindingOrNullImageView(uint32_t fetch_constant_index,
                                              xenos::FetchOpDimension dimension, bool is_signed);

  SamplerParameters GetSamplerParameters(const VulkanShader::SamplerBinding& binding,
                                         bool force_min_mag_linear = false) const;

  // Must be called for every used sampler at least once in a single submission,
  // and a submission must be open for this to be callable.
  // Returns:
  // - The sampler, if obtained successfully - and increases its last usage
  //   submission index - and has_overflown_out = false.
  // - VK_NULL_HANDLE and has_overflown_out = true if there's a total sampler
  //   count overflow in a submission that potentially hasn't completed yet.
  // - VK_NULL_HANDLE and has_overflown_out = false in case of a general failure
  //   to create a sampler.
  VkSampler UseSampler(SamplerParameters parameters, bool& has_overflown_out);
  // Returns the submission index to await (may be the current submission in
  // case of an overflow within a single submission - in this case, it must be
  // ended, and a new one must be started) in case of sampler count overflow, so
  // samplers may be freed, and UseSamplers may take their slots.
  uint64_t GetSubmissionToAwaitOnSamplerOverflow(uint32_t overflowed_sampler_count) const;

  // Returns the 2D view of the front buffer texture (for fragment shader
  // reading - the barrier will be pushed in the command processor if needed),
  // or VK_NULL_HANDLE in case of failure. May call LoadTextureData.
  // If swap_source_needs_rb_swap_out is not nullptr, writes whether the final
  // guest-to-host swizzle requires swapping red and blue (R <- B, B <- R) with
  // green preserved, which is needed by the presentation fallback path on
  // devices without imageViewFormatSwizzle.
  VkImageView RequestSwapTexture(uint32_t& width_scaled_out, uint32_t& height_scaled_out,
                                 xenos::TextureFormat& format_out,
                                 uint32_t* width_unscaled_out = nullptr,
                                 uint32_t* height_unscaled_out = nullptr,
                                 bool* swap_source_needs_rb_swap_out = nullptr,
                                 uint32_t* guest_base_address_out = nullptr);

  bool GetScaledResolveRange(uint32_t start_unscaled, uint32_t length_unscaled,
                             uint32_t length_scaled_alignment_log2, uint64_t& start_scaled_out,
                             uint64_t& length_scaled_out) const;
  bool CommitScaledResolveRange(uint32_t start_unscaled, uint32_t length_unscaled,
                                uint32_t length_scaled_alignment_log2 = 0) {
    return EnsureScaledResolveMemoryCommitted(start_unscaled, length_unscaled,
                                              length_scaled_alignment_log2);
  }
  VkBuffer scaled_resolve_buffer() const { return scaled_resolve_buffer_; }
  void AuditResolveProducer(const draw_util::ResolveInfo& info);
  void UseScaledResolveBufferForRead();

  // Default-off direct storage-image load path accounting (see
  // direct_storage_image.h). Cumulative over the process lifetime.
  struct DirectStorageImageStats {
    uint64_t direct_loads = 0;
    uint64_t fallbacks = 0;
    uint64_t copy_bytes_avoided = 0;
    uint64_t legacy_copy_bytes = 0;
    uint64_t fallbacks_by_reason[size_t(DirectStorageImageFallback::kCount)] = {};
  };
  DirectStorageImageStats GetDirectStorageImageStats() const;
  void UseScaledResolveBufferForWrite(uint64_t written_start_scaled,
                                      uint64_t written_length_scaled);
  size_t BeginDirectResolveParity(uint64_t bytes, uint32_t msaa_class,
                                  VkBuffer& candidate_buffer_out);
  void CompleteDirectResolveParity(size_t job_index, VkDeviceSize reference_offset);

 protected:
  bool IsSignedVersionSeparateForFormat(TextureKey key) const override;
  bool IsScaledResolveSupportedForFormat(TextureKey key) const override;
  uint32_t GetHostFormatSwizzle(TextureKey key) const override;

  uint32_t GetMaxHostTextureWidthHeight(xenos::DataDimension dimension) const override;
  uint32_t GetMaxHostTextureDepthOrArraySize(xenos::DataDimension dimension) const override;

  std::unique_ptr<Texture> CreateTexture(TextureKey key) override;

  bool EnsureScaledResolveMemoryCommitted(uint32_t start_unscaled, uint32_t length_unscaled,
                                          uint32_t length_scaled_alignment_log2 = 0) override;

  bool LoadTextureDataFromResidentMemoryImpl(Texture& texture, bool load_base,
                                             bool load_mips) override;

  void UpdateTextureBindingsImpl(uint32_t fetch_constant_mask) override;

 private:
  ResolveChainAudit resolve_chain_audit_;
  uint64_t audit_next_texture_id_ = 0, audit_next_load_id_ = 0;
  uint64_t audit_resolve_frame_ = UINT64_MAX, audit_resolve_ordinal_ = 0;

  // Default-off direct storage-image load path (direct_storage_image.h).
  // Disabled unless gpu_direct_storage_image_texture_load is set and the
  // device accepts VK_IMAGE_USAGE_STORAGE_BIT for the exact image parameters.
  bool direct_storage_image_enabled_ = false;
  bool direct_storage_image_format_supported_ = false;
  VkDescriptorSetLayout direct_storage_image_descriptor_set_layout_ = VK_NULL_HANDLE;
  VkDescriptorPool direct_storage_image_descriptor_pool_ = VK_NULL_HANDLE;
  VkPipelineLayout direct_storage_image_pipeline_layout_ = VK_NULL_HANDLE;
  VkPipeline direct_storage_image_pipeline_ = VK_NULL_HANDLE;
  VkPipeline direct_storage_image_depth_pipeline_ = VK_NULL_HANDLE;
  // Loaded only when the direct path is enabled; not part of core device init.
  PFN_vkFreeDescriptorSets vk_free_descriptor_sets_ = nullptr;
  std::atomic<uint64_t> direct_storage_image_loads_{0};
  std::atomic<uint64_t> direct_storage_image_fallbacks_{0};
  std::atomic<uint64_t> direct_storage_image_copy_bytes_avoided_{0};
  std::atomic<uint64_t> direct_storage_image_legacy_copy_bytes_{0};
  std::atomic<uint64_t> direct_storage_image_fallbacks_by_reason_[
      size_t(DirectStorageImageFallback::kCount)]{};
  DirectStorageImageStats direct_storage_image_audit_frame_start_{};

  enum LoadDescriptorSetIndex {
    kLoadDescriptorSetIndexDestination,
    kLoadDescriptorSetIndexSource,
    kLoadDescriptorSetCount,
  };

  struct HostFormat {
    LoadShaderIndex load_shader;
    // Do NOT add integer formats to this - they are not filterable, can only be
    // read with ImageFetch, not ImageSample! If any game is seen using
    // num_format 1 for fixed-point formats (for floating-point, it's normally
    // set to 1 though), add a constant buffer containing multipliers for the
    // textures and multiplication to the tfetch implementation.
    VkFormat format;
    // Whether the format is block-compressed on the host (the host block size
    // matches the guest format block size in this case), and isn't decompressed
    // on load.
    bool block_compressed;

    // Set up dynamically based on what's supported by the device.
    bool linear_filterable;
  };

  struct HostFormatPair {
    HostFormat format_unsigned;
    HostFormat format_signed;
    // Mapping of Xenos swizzle components to Vulkan format components.
    uint32_t swizzle;
    // Whether the unsigned and the signed formats are compatible for one image
    // and the same image data (on a portability subset device, this should also
    // take imageViewFormatReinterpretation into account).
    bool unsigned_signed_compatible;
  };

  class VulkanTexture final : public Texture {
    friend class VulkanTextureCache;
   public:
    enum class Usage {
      kUndefined,
      kTransferDestination,
      kParityTransferSource,
      // Written by the default-off direct storage-image load shader
      // (VK_IMAGE_LAYOUT_GENERAL, compute storage write). Leaves the image
      // ready for the guest shader sampled-read transition.
      kComputeStorageWrite,
      kGuestShaderSampled,
      kSwapSampled,
    };

    // Takes ownership of the image and its memory.
    explicit VulkanTexture(VulkanTextureCache& texture_cache, const TextureKey& key, VkImage image,
                           VmaAllocation allocation, bool track_usage = true,
                           bool direct_storage_image = false);
    ~VulkanTexture();

    VkImage image() const { return image_resource_->image; }
    bool direct_storage_image() const { return direct_storage_image_descriptor_set_ != VK_NULL_HANDLE; }
    VkDescriptorSet direct_storage_image_descriptor_set() const {
      return direct_storage_image_descriptor_set_;
    }
    uint64_t audit_texture_id_ = 0;
    std::pair<uint64_t, uint64_t> audit_last_invalidations_[2]{};

    // Owned by the cache; only non-null for the default-off direct
    // storage-image path (created when the image gets VK_IMAGE_USAGE_STORAGE_BIT).
    VkImageView direct_storage_image_view_ = VK_NULL_HANDLE;
    VkDescriptorSet direct_storage_image_descriptor_set_ = VK_NULL_HANDLE;
    // Doesn't transition (the caller must insert the barrier).
    Usage SetUsage(Usage new_usage) {
      Usage old_usage = image_resource_->usage;
      image_resource_->usage = new_usage;
      return old_usage;
    }

    VkImageView GetView(bool is_signed, uint32_t host_swizzle, bool is_array = true);
    VkImageView GetOrCreate3DAs2DImageView(bool is_signed, uint32_t host_swizzle);

   private:
    union ViewKey {
      uint32_t key;
      struct {
        uint32_t is_signed_separate_view : 1;
        uint32_t host_swizzle : 12;
        uint32_t is_array : 1;
      };

      ViewKey() : key(0) { static_assert_size(*this, sizeof(key)); }

      struct Hasher {
        size_t operator()(const ViewKey& key) const {
          return std::hash<decltype(key.key)>{}(key.key);
        }
      };
      bool operator==(const ViewKey& other_key) const { return key == other_key.key; }
      bool operator!=(const ViewKey& other_key) const { return !(*this == other_key); }
    };

    static constexpr VkComponentSwizzle GetComponentSwizzle(uint32_t texture_swizzle,
                                                            uint32_t component_index) {
      xenos::XE_GPU_TEXTURE_SWIZZLE texture_component_swizzle =
          xenos::XE_GPU_TEXTURE_SWIZZLE((texture_swizzle >> (3 * component_index)) & 0b111);
      if (texture_component_swizzle == xenos::XE_GPU_TEXTURE_SWIZZLE(component_index)) {
        // The portability subset requires all swizzles to be IDENTITY, return
        // IDENTITY specifically, not R, G, B, A.
        return VK_COMPONENT_SWIZZLE_IDENTITY;
      }
      switch (texture_component_swizzle) {
        case xenos::XE_GPU_TEXTURE_SWIZZLE_R:
          return VK_COMPONENT_SWIZZLE_R;
        case xenos::XE_GPU_TEXTURE_SWIZZLE_G:
          return VK_COMPONENT_SWIZZLE_G;
        case xenos::XE_GPU_TEXTURE_SWIZZLE_B:
          return VK_COMPONENT_SWIZZLE_B;
        case xenos::XE_GPU_TEXTURE_SWIZZLE_A:
          return VK_COMPONENT_SWIZZLE_A;
        case xenos::XE_GPU_TEXTURE_SWIZZLE_0:
          return VK_COMPONENT_SWIZZLE_ZERO;
        case xenos::XE_GPU_TEXTURE_SWIZZLE_1:
          return VK_COMPONENT_SWIZZLE_ONE;
        default:
          // An invalid value.
          return VK_COMPONENT_SWIZZLE_IDENTITY;
      }
    }

    struct ImageResource {
      VmaAllocator allocator;
      VkImage image;
      VmaAllocation allocation;
      Usage usage = Usage::kUndefined;
      ImageResource(VmaAllocator allocator_in, VkImage image_in, VmaAllocation allocation_in)
          : allocator(allocator_in), image(image_in), allocation(allocation_in) {}
      ~ImageResource() { vmaDestroyImage(allocator, image, allocation); }
      ImageResource(const ImageResource&) = delete;
      ImageResource& operator=(const ImageResource&) = delete;
    };
    std::shared_ptr<ImageResource> image_resource_;

    std::unordered_map<ViewKey, VkImageView, ViewKey::Hasher> views_;
    std::unique_ptr<VulkanTexture> texture_3d_as_2d_;
    VkImageView image_view_3d_as_2d_unsigned_ = VK_NULL_HANDLE;
    VkImageView image_view_3d_as_2d_signed_ = VK_NULL_HANDLE;
  };

  void InvalidateResolvedImages(uint32_t address, uint32_t length) override;
  void DrainResolvedImageInvalidations();
  bool TryReuseFusedResolveImage(VulkanTexture& texture, bool load_base, bool load_mips);
  bool MakeFusedResolveImageWritable(VulkanTexture& texture);
  void SwapTextureImage(VulkanTexture& a, VulkanTexture& b);
  bool RecordFusedImageParity(VulkanTexture& candidate, VulkanTexture& legacy);
  void CompleteFusedImageParity(uint64_t completed);
  struct FusedImageParityReadback {
    VkBuffer buffer;
    VmaAllocation allocation;
    void* mapping;
    uint64_t submission;
    VkDeviceSize bytes;
    uint64_t frame;
    uint32_t base;
  };
  std::vector<FusedImageParityReadback> fused_image_parity_readbacks_;
  bool fused_image_parity_reentry_ = false;
  uint32_t fused_image_parity_recorded_ = 0;
  ResolvedImageOwnership<TextureKey, VulkanTexture::ImageResource> resolved_images_{64,
                                                                                   128 * 1024 * 1024};
  std::unique_ptr<Texture> pending_fused_resolve_image_;
  std::vector<std::pair<uint64_t, std::unique_ptr<Texture>>> retired_fused_resolve_images_;
  std::vector<std::unique_ptr<Texture>> fused_image_pool_;
  std::unordered_map<const VulkanTexture::ImageResource*, VulkanTexture*> fused_producer_by_resource_;
  std::unique_ptr<Texture> CreateTextureResource(TextureKey key, bool track_usage);
  std::mutex resolved_image_invalidation_mutex_;
  std::atomic<bool> has_resolved_image_invalidations_{false};
  std::vector<std::pair<uint32_t, uint32_t>> resolved_image_invalidations_;

  struct VulkanTextureBinding {
    VkImageView image_view_unsigned;
    VkImageView image_view_signed;

    VulkanTextureBinding() { Reset(); }

    void Reset() {
      image_view_unsigned = VK_NULL_HANDLE;
      image_view_signed = VK_NULL_HANDLE;
    }
  };

  struct Sampler {
    VkSampler sampler;
    bool uses_custom_border_color;
    uint64_t last_usage_submission;
    std::pair<const SamplerParameters, Sampler>* used_previous;
    std::pair<const SamplerParameters, Sampler>* used_next;
  };

  static constexpr bool AreDimensionsCompatible(xenos::FetchOpDimension binding_dimension,
                                                xenos::DataDimension resource_dimension) {
    switch (binding_dimension) {
      case xenos::FetchOpDimension::k1D:
      case xenos::FetchOpDimension::k2D:
        return resource_dimension == xenos::DataDimension::k1D ||
               resource_dimension == xenos::DataDimension::k2DOrStacked ||
               resource_dimension == xenos::DataDimension::k3D;
      case xenos::FetchOpDimension::k3DOrStacked:
        return resource_dimension == xenos::DataDimension::k3D;
      case xenos::FetchOpDimension::kCube:
        return resource_dimension == xenos::DataDimension::kCube;
      default:
        return false;
    }
  }

  explicit VulkanTextureCache(const RegisterFile& register_file, VulkanSharedMemory& shared_memory,
                              uint32_t draw_resolution_scale_x, uint32_t draw_resolution_scale_y,
                              VulkanCommandProcessor& command_processor,
                              VkPipelineStageFlags guest_shader_pipeline_stages);

  bool Initialize();
  bool InitializeScaledResolveBuffer();
  void ShutdownScaledResolveBuffer();

  const HostFormatPair& GetHostFormatPair(TextureKey key) const;

  struct ParityReadback {
    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    void* mapping = nullptr;
    uint64_t bytes = 0, submission = 0, frame = 0;
    uint32_t width = 0, height = 0, format = 0;
    bool complete = false;
  };
  bool parity_reentry_ = false;
  uint32_t parity_requested_ = 0;
  std::vector<ParityReadback> parity_readbacks_;
  struct DirectResolveParityReadback {
    VkBuffer candidate_buffer = VK_NULL_HANDLE;
    VmaAllocation candidate_allocation = VK_NULL_HANDLE;
    VkBuffer readback_buffer = VK_NULL_HANDLE;
    VmaAllocation readback_allocation = VK_NULL_HANDLE;
    void* mapping = nullptr;
    uint64_t bytes = 0, submission = 0, frame = 0;
    uint32_t msaa_class = 0;
    bool complete = false;
  };
  uint32_t direct_resolve_parity_requested_ = 0;
  uint32_t direct_resolve_parity_msaa_mask_ = 0;
  std::vector<DirectResolveParityReadback> direct_resolve_parity_readbacks_;
  size_t BeginParityReadback(Texture& texture, bool load_base, bool load_mips);
  void CaptureParityImage(VulkanTexture& texture, ParityReadback& job, bool direct);
  void ProcessParityReadbacks();
  void ProcessDirectResolveParityReadbacks();

  void GetTextureUsageMasks(VulkanTexture::Usage usage, VkPipelineStageFlags& stage_mask,
                            VkAccessFlags& access_mask, VkImageLayout& layout);
  // vkGetPhysicalDeviceImageFormatProperties for the exact image parameters plus
  // VK_IMAGE_USAGE_STORAGE_BIT. Cached per host format for the process lifetime.
  bool QueryDirectStorageImageSupport(const VkImageCreateInfo& image_create_info);
  // Records one load against the direct path counters (hit or fallback reason).
  void CountDirectStorageImageResult(DirectStorageImageFallback fallback, uint64_t copy_bytes);
  bool IsInGpuFrameAuditWindow() const;
  bool EnsureScaledResolveBufferAllocated(uint64_t start_scaled, uint64_t length_scaled);
  void GetScaledResolveUsageMasks(VkPipelineStageFlags& stage_mask_out,
                                  VkAccessFlags& access_mask_out, bool write) const;

  xenos::ClampMode NormalizeClampMode(xenos::ClampMode clamp_mode) const;

  VulkanCommandProcessor& command_processor_;
  diagnostics::FragmentDiagnosticDescriptorLog diagnostic_descriptor_log_;
  VkPipelineStageFlags guest_shader_pipeline_stages_;

  // Using the Vulkan Memory Allocator because texture count in games is
  // naturally pretty much unbounded, while Vulkan implementations, especially
  // on Windows versions before 10, may have an allocation count limit as low as
  // 4096.
  VmaAllocator vma_allocator_ = VK_NULL_HANDLE;

  static const HostFormatPair kBestHostFormats[64];
  static const HostFormatPair kHostFormatGBGRUnaligned;
  static const HostFormatPair kHostFormatBGRGUnaligned;
  static const HostFormatPair kHostFormatDXT1Unaligned;
  static const HostFormatPair kHostFormatDXT2_3Unaligned;
  static const HostFormatPair kHostFormatDXT4_5Unaligned;
  static const HostFormatPair kHostFormatDXNUnaligned;
  static const HostFormatPair kHostFormatDXT5AUnaligned;
  HostFormatPair host_formats_[64];

  VkPipelineLayout load_pipeline_layout_ = VK_NULL_HANDLE;
  std::array<VkPipeline, kLoadShaderCount> load_pipelines_{};
  std::array<VkPipeline, kLoadShaderCount> load_pipelines_scaled_{};

  // If both images can be placed in the same allocation, it's one allocation,
  // otherwise it's two separate.
  std::array<VkDeviceMemory, 2> null_images_memory_{};
  VkImage null_image_2d_array_cube_ = VK_NULL_HANDLE;
  VkImage null_image_3d_ = VK_NULL_HANDLE;
  VkImageView null_image_view_2d_array_ = VK_NULL_HANDLE;
  VkImageView null_image_view_cube_ = VK_NULL_HANDLE;
  VkImageView null_image_view_3d_ = VK_NULL_HANDLE;
  bool null_images_cleared_ = false;

  std::array<VulkanTextureBinding, xenos::kTextureFetchConstantCount> vulkan_texture_bindings_;

  // Unsupported texture formats used during this frame (for research and
  // testing).
  enum : uint8_t {
    kUnsupportedResourceBit = 1,
    kUnsupportedUnormBit = kUnsupportedResourceBit << 1,
    kUnsupportedSnormBit = kUnsupportedUnormBit << 1,
  };
  uint8_t unsupported_format_features_used_[64] = {};

  uint32_t sampler_max_count_;

  xenos::AnisoFilter max_anisotropy_;

  std::unordered_map<SamplerParameters, Sampler, SamplerParameters::Hasher> samplers_;
  std::pair<const SamplerParameters, Sampler>* sampler_used_first_ = nullptr;
  std::pair<const SamplerParameters, Sampler>* sampler_used_last_ = nullptr;
  SamplerParameters last_sampler_parameters_{UINT32_MAX};
  VkSampler last_sampler_ = VK_NULL_HANDLE;
  uint32_t custom_border_color_sampler_count_ = 0;

  VkBuffer scaled_resolve_buffer_ = VK_NULL_HANDLE;
  uint64_t scaled_resolve_buffer_size_ = 0;
  bool scaled_resolve_buffer_sparse_ = false;
  uint32_t scaled_resolve_buffer_memory_type_ = UINT32_MAX;
  std::vector<VkDeviceMemory> scaled_resolve_buffer_memory_;
  uint32_t scaled_resolve_sparse_granularity_log2_ = UINT32_MAX;
  std::vector<uint64_t> scaled_resolve_sparse_allocated_;
  bool scaled_resolve_last_usage_write_ = false;
  std::pair<uint64_t, uint64_t> scaled_resolve_last_written_range_{0, 0};
};

}  // namespace rex::graphics::vulkan
