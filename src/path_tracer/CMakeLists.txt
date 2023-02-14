set(path_tracer_name path_tracer)

set(path_tracer_source
    ../static.h
    main.cpp
    type.h
)

add_executable(${path_tracer_name})

if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(${path_tracer_name} PRIVATE /Zc:preprocessor)
endif ()

target_sources(${path_tracer_name} PRIVATE ${path_tracer_source})

target_include_directories(${path_tracer_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${path_tracer_name} PRIVATE common)
target_link_libraries(${path_tracer_name} PRIVATE gui)
target_link_libraries(${path_tracer_name} PRIVATE device_api)
target_link_libraries(${path_tracer_name} PRIVATE scene)
target_link_libraries(${path_tracer_name} PRIVATE material)
target_link_libraries(${path_tracer_name} PRIVATE optix_util)

# compile for Optix
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/main.ptx
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
    COMMAND nvcc ${CMAKE_CURRENT_SOURCE_DIR}/main.cu
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/main.ptx -std=c++17
        -I ${OptiX_ROOT_DIR}/include
        -I ${CMAKE_CURRENT_SOURCE_DIR}
        -I ${pupil_optix_root}
)
add_custom_target(path_tracer-ptx
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/main.ptx
        # ${CMAKE_CURRENT_SOURCE_DIR}/spatial.ptx
        # ${CMAKE_CURRENT_SOURCE_DIR}/lighting.ptx
)

add_dependencies(${path_tracer_name} path_tracer-ptx)