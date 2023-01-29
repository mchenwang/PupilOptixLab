set(cuda_util_src
    preprocessor.h
    util.h
    vec_math.h
    cuda_util.cpp
    texture.h
    camera.h
    camera.cpp
    random.h
)

add_library(cuda_util STATIC ${cuda_util_src})

target_include_directories(cuda_util PUBLIC ../)

target_link_libraries(cuda_util PRIVATE common)
target_link_libraries(cuda_util PUBLIC CUDA::cudart CUDA::cuda_driver)

set_target_properties(cuda_util PROPERTIES FOLDER ${source_folder})