find_package(OptiX REQUIRED)

configure_file(static.h.in ${CMAKE_CURRENT_LIST_DIR}/static.h)
set(STATIC_DATA_FILE ${CMAKE_CURRENT_LIST_DIR}/static.h)

if(WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
    add_definitions(-DNOMINMAX)
endif()

file(GLOB_RECURSE ALL_INCLUDE CONFIGURE_DEPENDS "*.h" "*.cuh")
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS "*.cpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_INCLUDE} ${ALL_SOURCES})

add_library(${pupil_framework_name} STATIC ${ALL_INCLUDE} ${ALL_SOURCES})
target_include_directories(${pupil_framework_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)

add_dependencies(${pupil_framework_name} file-copy)
target_link_libraries(${pupil_framework_name} PUBLIC ${3rdparty_target})
target_link_libraries(${pupil_framework_name} PUBLIC CUDA::cudart CUDA::cuda_driver)
target_link_libraries(${pupil_framework_name} PUBLIC OptiX)

cuda_add_library(post_process
    ${CMAKE_CURRENT_SOURCE_DIR}/system/post_process.cuh
    ${CMAKE_CURRENT_SOURCE_DIR}/system/post_process.cu)
set_target_properties(post_process PROPERTIES CUDA_ARCHITECTURES "35;50;75")
target_link_libraries(${pupil_framework_name} PUBLIC post_process)
