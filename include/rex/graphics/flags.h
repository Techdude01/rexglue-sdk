/**
 ******************************************************************************
 * Xenia : Xbox 360 Emulator Research Project                                 *
 ******************************************************************************
 * Copyright 2020 Ben Vanik. All rights reserved.                             *
 * Released under the BSD license - see LICENSE in the root for more details. *
 ******************************************************************************
 *
 * @modified    Tom Clay, 2026 - Adapted for ReXGlue runtime
 */

#pragma once

#include <rex/cvar.h>

// GPU Core
REXCVAR_DECLARE(bool, vsync);
REXCVAR_DECLARE(bool, clear_memory_page_state);
REXCVAR_DECLARE(bool, half_pixel_offset);
REXCVAR_DECLARE(bool, async_shader_compilation);

// GPU Resolution / Readback / Queries
REXCVAR_DECLARE(int32_t, resolution_scale);
REXCVAR_DECLARE(int32_t, draw_resolution_scale_x);
REXCVAR_DECLARE(int32_t, draw_resolution_scale_y);
REXCVAR_DECLARE(bool, resolve_resolution_scale_fill_half_pixel_offset);
REXCVAR_DECLARE(bool, draw_resolution_scaled_texture_offsets);
REXCVAR_DECLARE(std::string, readback_resolve);
REXCVAR_DECLARE(bool, readback_resolve_half_pixel_offset);
REXCVAR_DECLARE(bool, readback_memexport);
REXCVAR_DECLARE(bool, readback_memexport_fast);
REXCVAR_DECLARE(bool, occlusion_query_enable);
REXCVAR_DECLARE(int32_t, query_occlusion_fake_sample_count);

// GPU Depth / Render Target Behavior
REXCVAR_DECLARE(bool, depth_float24_round);
REXCVAR_DECLARE(bool, depth_float24_convert_in_pixel_shader);
REXCVAR_DECLARE(bool, depth_transfer_not_equal_test);
REXCVAR_DECLARE(bool, native_stencil_value_output);
REXCVAR_DECLARE(bool, native_stencil_value_output_d3d12_intel);
REXCVAR_DECLARE(bool, gamma_render_target_as_unorm16);
REXCVAR_DECLARE(bool, native_2x_msaa);
REXCVAR_DECLARE(bool, snorm16_render_target_full_range);
REXCVAR_DECLARE(bool, mrt_edram_used_range_clamp_to_min);
REXCVAR_DECLARE(bool, direct_host_resolve);
REXCVAR_DECLARE(bool, gpu_direct_host_resolve_fused);
REXCVAR_DECLARE(bool, gpu_direct_host_resolve_audit);
REXCVAR_DECLARE(bool, vulkan_lazy_guest_byte_materialization);
REXCVAR_DECLARE(int32_t, gpu_direct_host_resolve_parity_resolves);
REXCVAR_DECLARE(bool, gpu_fused_resolve_image_parity);

// GPU Textures
REXCVAR_DECLARE(bool, gpu_resolve_chain_audit);
REXCVAR_DECLARE(bool, gpu_profile_scope_timestamps);
REXCVAR_DECLARE(bool, gpu_work_labels);
REXCVAR_DECLARE(uint64_t, gpu_work_label_start);
REXCVAR_DECLARE(uint64_t, gpu_work_label_frames);
REXCVAR_DECLARE(bool, gpu_direct_storage_image_texture_load);
REXCVAR_DECLARE(int32_t, gpu_direct_storage_image_parity_loads);
REXCVAR_DECLARE(bool, gpu_allow_invalid_fetch_constants);
REXCVAR_DECLARE(bool, gpu_3d_to_2d_texture);
REXCVAR_DECLARE(int32_t, anisotropic_override);
REXCVAR_DECLARE(int32_t, texture_cache_memory_limit_render_to_texture);
REXCVAR_DECLARE(int32_t, texture_cache_memory_limit_soft);
REXCVAR_DECLARE(int32_t, texture_cache_memory_limit_hard);
REXCVAR_DECLARE(int32_t, texture_cache_memory_limit_soft_lifetime);
REXCVAR_DECLARE(bool, non_seamless_cube_map);

// GPU Primitive Processing
REXCVAR_DECLARE(bool, execute_unclipped_draw_vs_on_cpu);
REXCVAR_DECLARE(bool, execute_unclipped_draw_vs_on_cpu_for_psi_render_backend);
REXCVAR_DECLARE(bool, execute_unclipped_draw_vs_on_cpu_with_scissor);
REXCVAR_DECLARE(bool, force_convert_line_loops_to_strips);
REXCVAR_DECLARE(bool, force_convert_quad_lists_to_triangle_lists);
REXCVAR_DECLARE(bool, force_convert_triangle_fans_to_lists);
REXCVAR_DECLARE(int32_t, primitive_processor_cache_min_indices);

// GPU Debug
REXCVAR_DECLARE(bool, gpu_debug_markers);
bool IsGpuDebugMarkersEnabled();

// GPU Alpha Test
REXCVAR_DECLARE(bool, use_fuzzy_alpha_epsilon);

// GPU Shader Translation
REXCVAR_DECLARE(std::string, dump_shaders);
REXCVAR_DECLARE(std::string, swap_post_effect);

#if REX_HAS_VULKAN
REXCVAR_DECLARE(bool, gpu_profile_scope_timestamps);
REXCVAR_DECLARE(bool, vulkan_sparse_shared_memory);
REXCVAR_DECLARE(bool, vulkan_hoist_shared_memory_uploads);
REXCVAR_DECLARE(bool, vulkan_prune_unused_attachment_load_store);
REXCVAR_DECLARE(bool, vulkan_elide_dead_edram_transfers);
REXCVAR_DECLARE(bool, vulkan_elide_intertile_transfers);
REXCVAR_DECLARE(bool, vulkan_merge_matching_attachment_passes);
REXCVAR_DECLARE(bool, vulkan_prune_transfer_stencil);
REXCVAR_DECLARE(bool, vulkan_reuse_transfer_scope_for_guest);
REXCVAR_DECLARE(bool, vulkan_defer_non_conflicting_barriers);
REXCVAR_DECLARE(bool, vulkan_promote_viewport_resolve_clears);
REXCVAR_DECLARE(bool, vulkan_coalesce_adjacent_clears);
REXCVAR_DECLARE(bool, vulkan_alias_4x_depth_to_1x);
REXCVAR_DECLARE(bool, vulkan_elide_intermediate_depth_transfers);
REXCVAR_DECLARE(bool, vulkan_early_z_opaque);
REXCVAR_DECLARE(double, vulkan_ablate_viewport_scale);
REXCVAR_DECLARE(bool, vulkan_ablate_skip_transfers);
REXCVAR_DECLARE(bool, vulkan_ablate_skip_depth_transfers);
REXCVAR_DECLARE(int32_t, vulkan_ablate_depth_transfer_mask);
REXCVAR_DECLARE(int32_t, vulkan_ablate_skip_depth_transfer_width);
REXCVAR_DECLARE(bool, vulkan_submit_on_primary_buffer_end);
REXCVAR_DECLARE(bool, vulkan_dynamic_rendering);
REXCVAR_DECLARE(bool, vulkan_async_skip_incomplete_frames);
REXCVAR_DECLARE(int32_t, vulkan_pipeline_creation_threads);
REXCVAR_DECLARE(bool, vulkan_tessellation_wireframe);
REXCVAR_DECLARE(bool, vulkan_force_expand_point_sprites_in_vs);
REXCVAR_DECLARE(bool, vulkan_force_expand_rectangle_lists_in_vs);
REXCVAR_DECLARE(bool, vulkan_force_convert_quad_lists_to_triangle_lists);
REXCVAR_DECLARE(std::string, render_target_path_vulkan);
// Legacy backend compatibility aliases for shared readback controls.
REXCVAR_DECLARE(bool, vulkan_readback_resolve);
REXCVAR_DECLARE(bool, vulkan_readback_memexport);
#endif  // REX_HAS_VULKAN

#if REX_HAS_D3D12
REXCVAR_DECLARE(bool, dxbc_switch);
REXCVAR_DECLARE(bool, dxbc_source_map);
REXCVAR_DECLARE(bool, d3d12_bindless);
REXCVAR_DECLARE(bool, d3d12_submit_on_primary_buffer_end);
REXCVAR_DECLARE(bool, d3d12_dxbc_disasm);
REXCVAR_DECLARE(bool, d3d12_dxbc_disasm_dxilconv);
REXCVAR_DECLARE(int32_t, d3d12_pipeline_creation_threads);
REXCVAR_DECLARE(bool, d3d12_tessellation_wireframe);
REXCVAR_DECLARE(bool, d3d12_tiled_shared_memory);
REXCVAR_DECLARE(std::string, render_target_path_d3d12);
// Legacy backend compatibility aliases for shared readback controls.
REXCVAR_DECLARE(bool, d3d12_readback_memexport);
REXCVAR_DECLARE(bool, d3d12_readback_resolve);
#endif  // REX_HAS_D3D12

#define XE_GPU_FINE_GRAINED_DRAW_SCOPES 1
