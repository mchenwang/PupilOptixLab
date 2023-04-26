set(target_name ReSTIR_DI)

cuda_compile_and_embed(g_restir_di_gbuffer_ptx GBufferPass/gbuffer.cu)
cuda_compile_and_embed(g_restir_di_shading_ptx ShadingPass/shading.cu)
cuda_compile_and_embed(g_restir_di_shadow_ptx ShadowRayPass/shadow.cu)
cuda_compile_and_embed(g_restir_di_temp_reuse_ptx TemporalReusePass/temporal_reuse.cu)
cuda_compile_and_embed(g_restir_di_spat_reuse_ptx SpatialReusePass/spatial_reuse.cu)

file(GLOB_RECURSE ALL_INCLUDE CONFIGURE_DEPENDS "*.h")
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS "*.cpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_INCLUDE} ${ALL_SOURCES})

add_executable(${target_name}
    ${g_restir_di_gbuffer_ptx}
    ${g_restir_di_shading_ptx}
    ${g_restir_di_shadow_ptx}
    ${g_restir_di_temp_reuse_ptx}
    ${g_restir_di_spat_reuse_ptx}
    ${ALL_INCLUDE}
    ${ALL_SOURCES}
)

target_link_libraries(${target_name} PUBLIC ${pupil_framework_name})
set_target_properties(${target_name} PROPERTIES FOLDER ${pupil_application})
