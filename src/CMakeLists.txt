find_package(OptiX REQUIRED)

configure_file(static.h.in ${CMAKE_CURRENT_LIST_DIR}/static.h)

if (WIN32)
    add_definitions(-DUNICODE -D_UNICODE)
    add_definitions(-DNOMINMAX)
endif ()

set(source_folder src)

file(GLOB_RECURSE ALL_INCLUDE CONFIGURE_DEPENDS "*.h")
file(GLOB_RECURSE ALL_SOURCES CONFIGURE_DEPENDS "*.cpp")
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${ALL_INCLUDE} ${ALL_SOURCES})

add_subdirectory(common)
add_subdirectory(cuda_util)
add_subdirectory(device)
add_subdirectory(gui)
add_subdirectory(scene)
add_subdirectory(material)

# executable target
add_subdirectory(path_tracer)