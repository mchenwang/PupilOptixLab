set(target_name path_tracer)

add_executable(${target_name}
    main.cpp
    pt_pass.h
    pt_pass.cpp
    type.h
)

target_link_libraries(${target_name} PUBLIC ${pupil_framework_name})
