add_library(common STATIC util.h util.cpp macro_map.h enum.h)

# TODO: use PUBLIC
target_include_directories(common PUBLIC ../)
set_target_properties(common PROPERTIES FOLDER ${source_folder})

file(GLOB_RECURSE cuda_util_src CONFIGURE_DEPENDS "cuda_util/*.*")
add_library(cuda_util STATIC ${cuda_util_src})
target_include_directories(cuda_util PUBLIC "cuda_util/")
set_target_properties(cuda_util PROPERTIES FOLDER ${source_folder})