find_package(OptiX REQUIRED)
FIND_PACKAGE(CUDA REQUIRED)
INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})

configure_file(static.h.in ${CMAKE_CURRENT_LIST_DIR}/static.h)
set(STATIC_DATA_FILE ${CMAKE_CURRENT_LIST_DIR}/static.h)

if(WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
    add_definitions(-DNOMINMAX)
endif()

# file(GLOB_RECURSE ALL_INCLUDE CONFIGURE_DEPENDS "*.h" "*.cuh")
file(GLOB_RECURSE ALL_INCLUDE CONFIGURE_DEPENDS "*.h" "*.cuh")
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS "*.cpp" "*.cu")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_INCLUDE} ${ALL_SOURCES})

add_library(${pupil_framework_name} STATIC ${ALL_INCLUDE} ${ALL_SOURCES})
target_include_directories(${pupil_framework_name} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/)

add_dependencies(${pupil_framework_name} file-copy)
target_link_libraries(${pupil_framework_name} PUBLIC ${3rdparty_target})
target_link_libraries(${pupil_framework_name} PUBLIC CUDA::cudart CUDA::cuda_driver)
target_link_libraries(${pupil_framework_name} PUBLIC OptiX)

# cuda_add_library(post_process ${CMAKE_CURRENT_SOURCE_DIR}/system/post_process.cu)
# set_target_properties(post_process PROPERTIES PUBLIC_HEADER ${CMAKE_CURRENT_SOURCE_DIR}/system/post_process.cuh)
# set_target_properties(post_process PROPERTIES POSITION_INDEPENDENT_CODE ON)
# target_link_directories(post_process PUBLIC ${pupil_framework_root}/cuda)
# target_include_directories(post_process PUBLIC ${pupil_framework_root}/cuda)
# target_compile_options(post_process PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:–std=c++17>)
set_target_properties(${pupil_framework_name} PROPERTIES CUDA_ARCHITECTURES "50;75")
set_target_properties(${pupil_framework_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${pupil_framework_name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# target_link_libraries(${pupil_framework_name} PUBLIC post_process ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})

# cuda_add_library(post_process SHARED
# ${pupil_framework_root}/cuda/preprocessor.h
# ${pupil_framework_root}/cuda/vec_math.h
# ${pupil_framework_root}/cuda/data_view.h
# ${CMAKE_CURRENT_SOURCE_DIR}/system/post_process.cuh
# ${CMAKE_CURRENT_SOURCE_DIR}/system/post_process.cu)
# target_include_directories(post_process PUBLIC ${pupil_framework_root}/cuda)
# target_compile_options(post_process PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:–std=c++17>)

# # target_link_directories(post_process PUBLIC ${pupil_framework_root})
# set_target_properties(post_process PROPERTIES CUDA_ARCHITECTURES "50;75")
# set_target_properties(post_process PROPERTIES POSITION_INDEPENDENT_CODE ON)

# # set_target_properties(post_process PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
# add_dependencies(${pupil_framework_name} post_process)
# target_link_libraries(${pupil_framework_name} PUBLIC post_process)
