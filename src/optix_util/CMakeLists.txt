set(optix_util_source
    util.h
    geometry.h
    geometry.cpp
    camera.h
    camera.cpp
    emitter.h
)

add_library(optix_util STATIC ${optix_util_source})
target_include_directories(optix_util PRIVATE ../)

target_link_libraries(optix_util PRIVATE common)
target_link_libraries(optix_util PRIVATE cuda_util)
target_link_libraries(optix_util PRIVATE scene)
target_link_libraries(optix_util PRIVATE device_api)

set_target_properties(optix_util PROPERTIES FOLDER ${source_folder})