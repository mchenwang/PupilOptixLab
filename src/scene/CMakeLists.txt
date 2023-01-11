set(scene_source
    scene.cpp
    scene.h
    shape.h
    shape.cpp
    transform.h
    transform.cpp
    texture.h
    texture.cpp
)

file(GLOB_RECURSE scene_xml_source CONFIGURE_DEPENDS "xml/*.*")

add_library(scene STATIC ${scene_source} ${scene_xml_source})
target_include_directories(scene PRIVATE ../)

target_link_libraries(scene PRIVATE common)
target_link_libraries(scene PRIVATE cuda_util)
target_link_libraries(scene PRIVATE stb)
target_link_libraries(scene PRIVATE pugixml)

set_target_properties(scene PROPERTIES FOLDER ${source_folder})