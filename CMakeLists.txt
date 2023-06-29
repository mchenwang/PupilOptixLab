cmake_minimum_required(VERSION 3.25.2)
project(PupilOptixLab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

if(NOT DEFINED pupil_target_output_dir)
    set(pupil_target_output_dir ${PROJECT_SOURCE_DIR}/target)
endif()

# set target folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${pupil_target_output_dir}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${pupil_target_output_dir}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${pupil_target_output_dir}/lib)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(pupil_framework_root ${CMAKE_CURRENT_SOURCE_DIR}/framework CACHE INTERNAL "")
set(3rdparty_target third-party-targets CACHE INTERNAL "")
set(pupil_framework_name PupilOptixLab CACHE INTERNAL "")

find_package(OptiX REQUIRED)

set(pupil_cuda_architectures ${CMAKE_CUDA_ARCHITECTURES} CACHE INTERNAL "")
set(pupil_cuda_version ${CUDA_VERSION} CACHE INTERNAL "")
message(STATUS "cuda arch: ${pupil_cuda_architectures} ")
message(STATUS "cuda version: ${pupil_cuda_version} ")

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
            -std=c++20 --generate-line-info -use_fast_math --keep
            --keep-device-functions --relocatable-device-code=true # for separate .cu files
            -D PUPIL_OPTIX
            -Xcompiler "/utf-8"
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

option(pupil_build_example "build framework examples" ON)
add_subdirectory(example)