set(scene_source
    loader.cpp
    scene.h
    shape.h
)

file(GLOB_RECURSE scene_xml_source CONFIGURE_DEPENDS "xml/*.*")

add_library(scene STATIC ${scene_source} ${scene_xml_source})
target_include_directories(scene PRIVATE ../)

target_link_libraries(scene INTERFACE common)
target_link_libraries(scene PRIVATE pugixml)

set_target_properties(scene PROPERTIES FOLDER ${source_folder})