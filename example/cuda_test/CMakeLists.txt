set(target_name cuda_test)

add_executable(${target_name}
    main.cpp
    pass.h
    pass.cpp
    kernel.h
    kernel.cu
)

set_target_properties(${target_name} PROPERTIES CUDA_ARCHITECTURES ${pupil_cuda_architectures})
target_link_libraries(${target_name} PUBLIC ${pupil_framework_name})

set_target_properties(${target_name} PROPERTIES FOLDER ${pupil_application_example})
