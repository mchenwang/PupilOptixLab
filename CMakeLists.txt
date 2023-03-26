cmake_minimum_required(VERSION 3.14)
project(PupilOptixLab LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(MSVC)
    add_compile_options("/MP")
    add_compile_options("/await") # for WinRT
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
endif(MSVC)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# set target folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/target/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/target/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/target/lib)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(pupil_optix_root ${CMAKE_CURRENT_SOURCE_DIR})
set(3rdparty_target third-party-targets)
set(pupil_framework_name PupilOptixLab)

add_subdirectory(3rdparty)
add_subdirectory(framework)
add_subdirectory(example)