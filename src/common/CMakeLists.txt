set(common_src
    util.h
    type.h
    util.cpp
    macro_map.h
    enum.h
    texture.h
    transform.h
    transform.cpp
    camera.h
    camera.cpp
)
add_library(common STATIC ${common_src})

target_include_directories(common PUBLIC ../)
set_target_properties(common PROPERTIES FOLDER ${source_folder})