set(target_name path_tracer)

add_executable(${target_name}
    main.cpp
    pt_pass.h
    pt_pass.cpp
    type.h
)

target_link_libraries(${target_name} PUBLIC ${pupil_framework_name})

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/main.ptx
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
        ${pupil_framework_root}/optix/*.h
        ${pupil_framework_root}/material/*.h
    COMMAND
        nvcc ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/main.ptx -std=c++17
        -I ${OptiX_ROOT_DIR}/include
        -I ${CMAKE_CURRENT_SOURCE_DIR}
        -I ${pupil_framework_root}
)
add_custom_target(path_tracer-ptx
    DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/main.ptx
)

add_dependencies(${target_name} path_tracer-ptx)
