set(material_source
    material.h
)

add_library(material STATIC ${material_source})
target_include_directories(material PRIVATE ../)

target_link_libraries(material PRIVATE cuda_util)

set_target_properties(material PROPERTIES FOLDER ${source_folder})