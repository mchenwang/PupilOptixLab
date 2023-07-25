
configure_file(static.h.in ${CMAKE_CURRENT_LIST_DIR}/static.h)
set(STATIC_DATA_FILE ${CMAKE_CURRENT_LIST_DIR}/static.h)

file(GLOB_RECURSE ALL_INCLUDE CONFIGURE_DEPENDS "*.h")
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.inl")

set(pupil_cuda_kernel_files
    system/gui/buffer_to_canvas.cu
    system/gui/buffer_to_canvas.cuh
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_INCLUDE} ${ALL_SOURCES} ${pupil_cuda_kernel_files})

cuda_compile_and_embed(g_pupil_material_embedded_ptx_code render/material/bsdf/bsdf.cu)

set(pupil_builtin_embeding_cu
    ${g_pupil_material_embedded_ptx_code}
)

add_library(${pupil_framework_name} STATIC ${ALL_INCLUDE} ${ALL_SOURCES} ${pupil_builtin_embeding_cu} ${pupil_cuda_kernel_files})
target_include_directories(${pupil_framework_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)

add_dependencies(${pupil_framework_name} file-copy)
target_link_libraries(${pupil_framework_name} PUBLIC ${3rdparty_target})
target_link_libraries(${pupil_framework_name} PUBLIC CUDA::cudart CUDA::cuda_driver)
target_link_libraries(${pupil_framework_name} PUBLIC OptiX)

target_compile_definitions(${pupil_framework_name} PUBLIC _UNICODE UNICODE NOMINMAX)

target_compile_options(${pupil_framework_name} BEFORE PUBLIC "$<$<COMPILE_LANGUAGE:C>:/utf-8>")
target_compile_options(${pupil_framework_name} BEFORE PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/await>")
target_compile_options(${pupil_framework_name} BEFORE PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
target_compile_options(${pupil_framework_name} BEFORE PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:/MP>")
target_compile_options(${pupil_framework_name} BEFORE PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler "/utf-8 /wd4819 /await" >)
target_compile_options(${pupil_framework_name} BEFORE PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>)

target_compile_definitions(${pupil_framework_name} PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:PUPIL_CPP>")
target_compile_definitions(${pupil_framework_name} PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:PUPIL_CUDA>")
