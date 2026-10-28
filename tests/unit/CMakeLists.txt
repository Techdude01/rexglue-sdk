# Unit Tests for ReXGlue core components
add_executable(transfer_scope_reuse_tests graphics/vulkan/transfer_scope_reuse_test.cpp)
target_include_directories(transfer_scope_reuse_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include ${PROJECT_SOURCE_DIR}/thirdparty/vulkan-headers/include)
target_link_libraries(transfer_scope_reuse_tests PRIVATE Catch2::Catch2WithMain)
add_executable(render_target_accumulation_tests graphics/render_target_accumulation_test.cpp)
target_include_directories(render_target_accumulation_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(render_target_accumulation_tests PRIVATE Catch2::Catch2WithMain fmt::fmt)
add_executable(frame_capture_schedule_tests graphics/vulkan/frame_capture_schedule_test.cpp)
target_include_directories(frame_capture_schedule_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(frame_capture_schedule_tests PRIVATE Catch2::Catch2WithMain)

add_executable(presenter_repaint_tests ui/presenter_repaint_test.cpp)
target_include_directories(presenter_repaint_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/imgui)
target_link_libraries(presenter_repaint_tests PRIVATE rexruntime Catch2::Catch2WithMain)

add_executable(frame_cpu_timeline_tests graphics/frame_cpu_timeline_test.cpp)
target_include_directories(frame_cpu_timeline_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(frame_cpu_timeline_tests PRIVATE Catch2::Catch2WithMain)

add_executable(render_area_tests graphics/vulkan/render_area_test.cpp)
target_include_directories(render_area_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/vulkan-headers/include)
target_link_libraries(render_area_tests PRIVATE Catch2::Catch2WithMain)

add_executable(edram_depth_column_tests graphics/vulkan/edram_depth_column_test.cpp)
target_include_directories(edram_depth_column_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(edram_depth_column_tests PRIVATE Catch2::Catch2WithMain)

add_executable(shared_memory_hoist_tests graphics/vulkan/shared_memory_hoist_test.cpp)
target_include_directories(shared_memory_hoist_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(shared_memory_hoist_tests PRIVATE Catch2::Catch2WithMain)

add_executable(input_trace_tests input/input_trace_test.cpp
    ${PROJECT_SOURCE_DIR}/src/input/input_trace.cpp)
target_include_directories(input_trace_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(input_trace_tests PRIVATE Catch2::Catch2WithMain)

add_executable(spirv_declaration_harness
    graphics/vulkan/spirv_declaration_harness.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/format/ucode.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/pipeline/shader/shader.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/pipeline/shader/translator.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/pipeline/shader/translator_disasm.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/pipeline/shader/spirv_builder.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/pipeline/shader/spirv.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/pipeline/shader/spirv_translator.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/pipeline/shader/spirv_translator_alu.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/pipeline/shader/spirv_translator_fetch.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/pipeline/shader/spirv_translator_memexport.cpp
    ${PROJECT_SOURCE_DIR}/src/graphics/pipeline/shader/spirv_translator_rb.cpp
)
target_include_directories(spirv_declaration_harness PRIVATE
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/thirdparty/fmt/include
    ${PROJECT_SOURCE_DIR}/thirdparty/glslang
    ${PROJECT_SOURCE_DIR}/thirdparty/xxhash
    ${PROJECT_SOURCE_DIR}/thirdparty/vulkan-headers/include
    ${PROJECT_SOURCE_DIR}/thirdparty/vulkan-memory-allocator/include
    ${PROJECT_SOURCE_DIR}/thirdparty/spirv-tools/include
)
target_precompile_headers(spirv_declaration_harness PRIVATE <rex/logging/macros.h>)
target_link_libraries(spirv_declaration_harness PRIVATE rexgpu-xenos rexcore)

add_executable(spirv_modification_tests
    graphics/vulkan/spirv_modification_test.cpp
)
target_include_directories(spirv_modification_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/thirdparty/glslang
    ${PROJECT_SOURCE_DIR}/thirdparty/fmt/include
    ${PROJECT_SOURCE_DIR}/thirdparty/xxhash
    ${PROJECT_SOURCE_DIR}/thirdparty/vulkan-headers/include
    ${PROJECT_SOURCE_DIR}/thirdparty/vulkan-memory-allocator/include
    ${PROJECT_SOURCE_DIR}/thirdparty/spirv-tools/include
)
target_link_libraries(spirv_modification_tests PRIVATE Catch2::Catch2WithMain)

add_executable(fragment_diagnostic_tests
    graphics/vulkan/fragment_diagnostic_test.cpp
)
target_include_directories(fragment_diagnostic_tests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(fragment_diagnostic_tests PRIVATE Catch2::Catch2WithMain)

add_executable(unit_tests
    test_memory.cpp
    memory/heap_allocation_test.cpp
    kernel/object_table_test.cpp
    core/cvar_test.cpp
    core/fiber_test.cpp
    core/filesystem_test.cpp
    core/hash_test.cpp
    core/stream_test.cpp
    core/chrono_test.cpp
    graphics/vulkan/fp32_probe_transport_test.cpp
    graphics/vulkan/fragment_diagnostic_test.cpp
    graphics/vulkan/spirv_modification_test.cpp
    ppc/import_function_test.cpp
    ppc/stack_test.cpp
    system/guest_path_test.cpp
    system/achievement_manager_test.cpp
    system/function_dispatcher_test.cpp
    input/input_assignment_test.cpp
    input/input_trace_test.cpp
    codegen/codegen_writer_test.cpp
    codegen/output_partition_test.cpp
    codegen/output_stamp_test.cpp
)

target_sources(unit_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/src/rexglue/ui/ui.cpp
    ${PROJECT_SOURCE_DIR}/src/rexglue/ui/progress.cpp
)

target_include_directories(unit_tests PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/thirdparty/glslang
    ${PROJECT_SOURCE_DIR}/thirdparty/xxhash
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(unit_tests PRIVATE
    rexcodegen
    rexruntime
    Catch2::Catch2WithMain
)

# ReXApp is shipped as source for recomp projects, rather than compiled into
# rexui. Keep its SDK-consumer compilation path covered by the test build.
add_library(rexapp_compile_test OBJECT
    ${PROJECT_SOURCE_DIR}/src/ui/rex_app.cpp
)
target_link_libraries(rexapp_compile_test PRIVATE
    rexruntime
    rexui
    rexinput
    rexaudio
)
add_dependencies(unit_tests rexapp_compile_test)

# Strip Tracy/profiling instrumentation from test targets
target_compile_options(unit_tests PRIVATE
    -UTRACY_ENABLE -UREXGLUE_ENABLE_PROFILING
    -UREXGLUE_ENABLE_PERF_COUNTERS -UREXGLUE_PROFILE_GUEST_FUNCTIONS
)

set_target_properties(unit_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:rexruntime>
)
if(UNIX AND NOT APPLE)
    set_target_properties(unit_tests PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH ON
    )
endif()

# Discover tests
include(Catch)
catch_discover_tests(transfer_scope_reuse_tests TEST_PREFIX "transfer_scope.")
catch_discover_tests(render_target_accumulation_tests TEST_PREFIX "attachment_retention.")
catch_discover_tests(frame_capture_schedule_tests TEST_PREFIX "capture_schedule.")
catch_discover_tests(presenter_repaint_tests TEST_PREFIX "presenter_repaint.")
catch_discover_tests(frame_cpu_timeline_tests TEST_PREFIX "frame_cpu_timeline.")
catch_discover_tests(render_area_tests TEST_PREFIX "render_area.")
catch_discover_tests(edram_depth_column_tests TEST_PREFIX "edram_depth_column.")
catch_discover_tests(input_trace_tests TEST_PREFIX "input_trace.")
catch_discover_tests(fragment_diagnostic_tests TEST_PREFIX "fragment_diagnostic.")
catch_discover_tests(unit_tests
    TEST_PREFIX "unit."
    PROPERTIES
        LABELS "unit"
        TIMEOUT 60
)
