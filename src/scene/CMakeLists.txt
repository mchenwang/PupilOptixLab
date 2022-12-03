set(scene_source
    loader.cpp
    loader.h
)

add_library(scene STATIC ${scene_source})
target_include_directories(scene PRIVATE ../)

target_link_libraries(scene INTERFACE common)
target_link_libraries(scene PRIVATE pugixml)

set_target_properties(scene PROPERTIES FOLDER ${source_folder})