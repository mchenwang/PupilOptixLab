set(cuda_util_src
    preprocessor.h
    util.h
    vec_math.h
    cuda_util.cpp
)

add_library(cuda_util STATIC ${cuda_util_src})

target_include_directories(cuda_util PUBLIC ../)
set_target_properties(cuda_util PROPERTIES FOLDER ${source_folder})