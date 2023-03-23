find_package(OptiX REQUIRED)

configure_file(static.h.in ${CMAKE_CURRENT_LIST_DIR}/static.h)
set(STATIC_DATA_FILE ${CMAKE_CURRENT_LIST_DIR}/static.h)

if(WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
    add_definitions(-DNOMINMAX)
endif()

file(GLOB_RECURSE ALL_INCLUDE CONFIGURE_DEPENDS "*.h")
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS "*.cpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_INCLUDE} ${ALL_SOURCES})

set(pupil_optix_target_name PupilOptixLab)
add_library(${pupil_optix_target_name} STATIC ${ALL_INCLUDE} ${ALL_SOURCES})
target_include_directories(${pupil_optix_target_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)

add_dependencies(${pupil_optix_target_name} file-copy)
target_link_libraries(${pupil_optix_target_name} PUBLIC ${3rdparty_target})
target_link_libraries(${pupil_optix_target_name} PUBLIC CUDA::cudart CUDA::cuda_driver)
target_link_libraries(${pupil_optix_target_name} PUBLIC OptiX)
