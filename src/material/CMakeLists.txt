set(material_source
    material.h
    material.cpp
    ior.h
    optix_material.h
)

add_library(material STATIC ${material_source})
target_include_directories(material PRIVATE ../)

target_link_libraries(material PRIVATE common)
target_link_libraries(material PRIVATE cuda_util)
target_link_libraries(material PRIVATE scene)
target_link_libraries(material PRIVATE device_api)

set_target_properties(material PROPERTIES FOLDER ${source_folder})