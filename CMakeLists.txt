cmake_minimum_required(VERSION 3.18)
project(PupilOptixLab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options("/MP")
    add_compile_options("/await") # for WinRT
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    add_definitions(-DUNICODE -D_UNICODE)
    add_definitions(-DNOMINMAX)
endif(MSVC)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# set target folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/target/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/target/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/target/lib)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(pupil_framework_root ${CMAKE_CURRENT_SOURCE_DIR}/framework)
set(3rdparty_target third-party-targets)
set(pupil_framework_name PupilOptixLab)

find_program(BIN2C bin2c
    DOC "Path to the cuda-sdk bin2c executable.")

macro(cuda_compile_and_embed output_var cuda_file)
    set(c_var_name ${output_var})
    add_custom_command(
        OUTPUT ${cuda_file}.ptx
        DEPENDS
            ${cuda_file}
            ${pupil_framework_name}
        COMMAND
            nvcc ${CMAKE_CURRENT_LIST_DIR}/${cuda_file}
            -ptx -o ${CMAKE_CURRENT_BINARY_DIR}/${cuda_file}.ptx
            -std=c++17 --generate-line-info -use_fast_math --keep
            -I ${OptiX_ROOT_DIR}/include
            -I ${CMAKE_CURRENT_SOURCE_DIR}
            -I ${pupil_framework_root}
    )
    set(ptx_file ${CMAKE_CURRENT_BINARY_DIR}/${cuda_file}.ptx)
    set(embedded_file ${CMAKE_CURRENT_BINARY_DIR}/${cuda_file}.embedded.c)
    add_custom_command(
        OUTPUT ${embedded_file}
        COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
        DEPENDS ${ptx_file}
        COMMENT "compiling (and embedding ptx from) ${cuda_file} (to ${embedded_file})"
    )
    set(${output_var} ${embedded_file})
endmacro()

add_subdirectory(3rdparty)
add_subdirectory(framework)
add_subdirectory(app)