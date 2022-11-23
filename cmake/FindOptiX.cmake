if(DEFINED ENV{OptiX_INSTALL_DIR})
    message(STATUS "Detected the OptiX_INSTALL_DIR env variable (pointing to $ENV{OptiX_INSTALL_DIR}; going to use this for finding optix.h")
    find_path(OptiX_ROOT_DIR NAMES include/optix.h PATHS $ENV{OptiX_INSTALL_DIR})
elseif(WIN32)
    find_path(OptiX_ROOT_DIR
        NAMES include/optix.h
        PATHS
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.5.0"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.4.0"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.3.0"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.2.0"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.1.0"
            "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.0.0"
    )
else()
    find_path(OptiX_ROOT_DIR NAMES include/optix.h)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OptiX
    FAIL_MESSAGE "Failed to find OptiX install dir. Please instal OptiX or set OptiX_INSTALL_DIR env variable."
    REQUIRED_VARS OptiX_ROOT_DIR
)

add_library(OptiX INTERFACE IMPORTED)
target_include_directories(OptiX INTERFACE ${OptiX_ROOT_DIR}/include)

enable_language(CUDA)
find_package(CUDA 11.6 REQUIRED)
find_package(CUDAToolkit 11.6 REQUIRED)

find_program(BIN2C bin2c
  DOC "Path to the cuda-sdk bin2c executable.")

# code from https://github.com/ingowald/optix7course/blob/master/common/gdt/cmake/configure_optix.cmake
macro(cuda_compile_and_embed output_var cuda_file)
  set(c_var_name ${output_var})
  cuda_compile_ptx(ptx_files ${cuda_file} OPTIONS --generate-line-info -use_fast_math --keep)
  list(GET ptx_files 0 ptx_file)
  set(embedded_file ${ptx_file}_embedded.c)
  add_custom_command(
    OUTPUT ${embedded_file}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_var_name} ${ptx_file} > ${embedded_file}
    DEPENDS ${ptx_file}
    COMMENT "compiling (and embedding ptx from) ${cuda_file}"
  )
  set(${output_var} ${embedded_file})
endmacro()