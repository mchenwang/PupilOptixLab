add_library(common STATIC util.h util.cpp macro_map.h enum.h)

target_include_directories(common PUBLIC ../)
set_target_properties(common PROPERTIES FOLDER ${source_folder})