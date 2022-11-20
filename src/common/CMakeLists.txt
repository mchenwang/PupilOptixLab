add_library(common INTERFACE util.h)
target_include_directories(common INTERFACE ../)
set_target_properties(common PROPERTIES FOLDER ${source_folder})