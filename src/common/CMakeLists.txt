add_library(common INTERFACE util.h macro_map.h)
target_include_directories(common INTERFACE ../)
set_target_properties(common PROPERTIES FOLDER ${source_folder})

file(GLOB_RECURSE cuda_util_src CONFIGURE_DEPENDS "cuda_util/*.*")
add_library(cuda_util INTERFACE ${cuda_util_src})
target_include_directories(cuda_util INTERFACE "cuda_util/")
set_target_properties(cuda_util PROPERTIES FOLDER ${source_folder})