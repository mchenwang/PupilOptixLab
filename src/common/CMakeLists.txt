set(common_src
    util.h
    util.cpp
    macro_map.h
    enum.h
    texture.h
)
add_library(common STATIC ${common_src})

target_include_directories(common PUBLIC ../)
set_target_properties(common PROPERTIES FOLDER ${source_folder})