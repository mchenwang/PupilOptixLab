# cuda toolkit
find_package(CUDAToolkit REQUIRED)

# optix
find_package(OptiX REQUIRED)

configure_file(static.h.in ${CMAKE_CURRENT_LIST_DIR}/static.h)

if (WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
endif ()

set(source_folder src)

file(GLOB_RECURSE ALL_INCLUDE CONFIGURE_DEPENDS "*.h")
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS "*.cpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_INCLUDE} ${ALL_SOURCES})

add_subdirectory(common)
add_subdirectory(gui)

add_executable(${PROJECT_NAME})
target_sources(${PROJECT_NAME}
    PRIVATE
        static.h
        main.cpp
)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# target_link_libraries(${PROJECT_NAME} PRIVATE OptiX::OptiX)
# target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cuda_driver)
target_link_libraries(${PROJECT_NAME} PRIVATE gui common)