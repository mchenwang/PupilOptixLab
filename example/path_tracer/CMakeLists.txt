set(target_name path_tracer)

cuda_compile_and_embed(embedded_ptx_code main.cu)

add_executable(${target_name}
    ${embedded_ptx_code}
    main.cpp
    pt_pass.h
    pt_pass.cpp
    type.h
)

target_link_libraries(${target_name} PUBLIC ${pupil_framework_name})
set_target_properties(${target_name} PROPERTIES FOLDER ${pupil_application_example})
