file(GLOB_RECURSE optix_wrap_src CONFIGURE_DEPENDS "*.*")

set(device_api_source
    error_handle.h
    d3dx12.h
    dx12_device.h
    dx12_device.cpp
    optix_device.h
    optix_device.cpp
    wsa.h wsa.cpp
    cuda_texture.h
    cuda_texture.cpp
    ${optix_wrap_src}
)

add_library(device_api STATIC ${device_api_source})
target_include_directories(device_api PRIVATE ../)

target_link_libraries(device_api PRIVATE DirectXTK12)
target_link_libraries(device_api PUBLIC CUDA::cudart CUDA::cuda_driver)
target_link_libraries(device_api PUBLIC OptiX)
target_link_libraries(device_api PRIVATE common)
target_link_libraries(device_api PRIVATE cuda_util)
target_link_libraries(device_api PRIVATE scene)

set_target_properties(device_api PROPERTIES FOLDER ${source_folder})