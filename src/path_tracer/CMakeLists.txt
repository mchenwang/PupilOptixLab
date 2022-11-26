add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/main.ptx
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/type.h
    COMMAND nvcc ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/main.ptx -std=c++17
        -I ${OptiX_ROOT_DIR}/include
        -I ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_target(raytracing-ptx
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/main.ptx
        # ${CMAKE_CURRENT_SOURCE_DIR}/spatial.ptx
        # ${CMAKE_CURRENT_SOURCE_DIR}/lighting.ptx
)

set(path_tracer_source
    type.h
)

target_sources(${PROJECT_NAME} PRIVATE ${path_tracer_source})
add_dependencies(${PROJECT_NAME} raytracing-ptx)