set(target_name ddgi)

cuda_compile_and_embed(ddgi_pt_pass_embedded_ptx_code pt/pt.cu)
cuda_compile_and_embed(ddgi_gbuffer_pass_embedded_ptx_code gbuffer/gbuffer.cu)

add_executable(${target_name}
    ${ddgi_pt_pass_embedded_ptx_code}
    ${ddgi_gbuffer_pass_embedded_ptx_code}
    main.cpp
    gbuffer/pass.h
    gbuffer/pass.cpp
    gbuffer/type.h
    pt/pass.h
    pt/pass.cpp
    pt/type.h
)

target_link_libraries(${target_name} PUBLIC ${pupil_framework_name})
set_target_properties(${target_name} PROPERTIES FOLDER ${pupil_application})
