if(DEFINED ENV{OptiX_INSTALL_DIR})
    message(STATUS "Detected the OptiX_INSTALL_DIR env variable (pointing to $ENV{OptiX_INSTALL_DIR}; going to use this for finding optix.h")
    find_path(OptiX_ROOT_DIR NAMES include/optix.h PATHS $ENV{OptiX_INSTALL_DIR})
elseif(WIN32)
    find_path(OptiX_ROOT_DIR
        NAMES include/optix.h
        PATHS "C:/ProgramData/NVIDIA Corporation/OptiX SDK 7.5.0"
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
find_package(CUDAToolkit 12.0 REQUIRED)

# Adapted from the CMake source code at https://github.com/Kitware/CMake/blob/master/Modules/FindCUDA/select_compute_arch.cmake
# Simplified to return a semicolon-separated list of the compute capabilities of installed devices
function(TCNN_AUTODETECT_CUDA_ARCHITECTURES OUT_VARIABLE)
    if(NOT TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT)
        if(CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
            set(file "${PROJECT_BINARY_DIR}/detect_tcnn_cuda_architectures.cu")
        else()
            set(file "${PROJECT_BINARY_DIR}/detect_tcnn_cuda_architectures.cpp")
        endif()

        file(WRITE ${file} ""
            "#include <cuda_runtime.h>\n"
            "#include <cstdio>\n"
            "int main() {\n"
            "	int count = 0;\n"
            "	if (cudaSuccess != cudaGetDeviceCount(&count)) return -1;\n"
            "	if (count == 0) return -1;\n"
            "	for (int device = 0; device < count; ++device) {\n"
            "		cudaDeviceProp prop;\n"
            "		if (cudaSuccess == cudaGetDeviceProperties(&prop, device)) {\n"
            "			std::printf(\"%d%d\", prop.major, prop.minor);\n"
            "			if (device < count - 1) std::printf(\";\");\n"
            "		}\n"
            "	}\n"
            "	return 0;\n"
            "}\n"
        )

        if(CMAKE_CUDA_COMPILER_LOADED) # CUDA as a language
            try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${file} RUN_OUTPUT_VARIABLE compute_capabilities)
        else()
            try_run(
                run_result compile_result ${PROJECT_BINARY_DIR} ${file}
                CMAKE_FLAGS "-DINCLUDE_DIRECTORIES=${CUDA_INCLUDE_DIRS}"
                LINK_LIBRARIES ${CUDA_LIBRARIES}
                RUN_OUTPUT_VARIABLE compute_capabilities
            )
        endif()

        if(run_result EQUAL 0)
            # If the user has multiple GPUs with the same compute capability installed, list that capability only once.
            list(REMOVE_DUPLICATES compute_capabilities)
            set(TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT ${compute_capabilities} CACHE INTERNAL "Returned GPU architectures from detect_gpus tool" FORCE)
        endif()
    endif()

    if(NOT TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT)
        message(STATUS "Automatic GPU detection failed. Building for Turing and Ampere as a best guess.")
        set(${OUT_VARIABLE} "75;86" PARENT_SCOPE)
    else()
        set(${OUT_VARIABLE} ${TCNN_AUTODETECT_CUDA_ARCHITECTURES_OUTPUT} PARENT_SCOPE)
    endif()
endfunction()

message(STATUS "Obtained CUDA architectures automatically from installed GPUs")
TCNN_AUTODETECT_CUDA_ARCHITECTURES(CMAKE_CUDA_ARCHITECTURES)
