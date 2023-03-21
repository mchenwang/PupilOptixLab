set(ddgi_name ddgi)

set(ddgi_source
    ../static.h
    main.cpp
    pt/type.h
    pt/pass.h
)

add_executable(${ddgi_name})

if(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
    target_compile_options(${ddgi_name} PRIVATE /Zc:preprocessor)
endif()

target_sources(${ddgi_name} PRIVATE ${ddgi_source})

target_include_directories(${ddgi_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${ddgi_name} PRIVATE common)
target_link_libraries(${ddgi_name} PRIVATE gui)
target_link_libraries(${ddgi_name} PRIVATE device_api)
target_link_libraries(${ddgi_name} PRIVATE scene)
target_link_libraries(${ddgi_name} PRIVATE material)
target_link_libraries(${ddgi_name} PRIVATE optix_util)

# add_subdirectory(pt)
# add_subdirectory(gbuffer)
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/pt/pt_main.ptx
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/pt/main.cu
        ${pupil_optix_root}/optix_util/*.h
        ${pupil_optix_root}/material/*.h
    COMMAND
        nvcc ${CMAKE_CURRENT_SOURCE_DIR}/pt/main.cu
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/pt/pt_main.ptx -std=c++17
        -I ${OptiX_ROOT_DIR}/include
        -I ${CMAKE_CURRENT_SOURCE_DIR}
        -I ${pupil_optix_root}
)

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/gbuffer/gbuffer_main.ptx
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/gbuffer/main.cu
        ${pupil_optix_root}/optix_util/*.h
        ${pupil_optix_root}/material/*.h
    COMMAND
        nvcc ${CMAKE_CURRENT_SOURCE_DIR}/gbuffer/main.cu
        -ptx -o ${CMAKE_CURRENT_SOURCE_DIR}/gbuffer/gbuffer_main.ptx -std=c++17
        -I ${OptiX_ROOT_DIR}/include
        -I ${CMAKE_CURRENT_SOURCE_DIR}
        -I ${pupil_optix_root}
)

add_custom_target(ddgi-ptx
    DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/pt/pt_main.ptx
    DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/gbuffer/gbuffer_main.ptx
)

add_dependencies(${ddgi_name} ddgi-ptx)