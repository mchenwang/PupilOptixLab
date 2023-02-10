set(material_source
    predefine.h
    material.h
    material.cpp
    ior.h
    fresnel.h
    optix_material.h
    optix_material.cpp

    bsdf/bsdf.h
    bsdf/diffuse.h
    bsdf/dielectric.h
    bsdf/conductor.h
    bsdf/rough_conductor.h
)

add_library(material STATIC ${material_source})
target_include_directories(material PRIVATE ../)

target_link_libraries(material PRIVATE common)
target_link_libraries(material PRIVATE cuda_util)
target_link_libraries(material PRIVATE scene)
target_link_libraries(material PRIVATE device_api)
target_link_libraries(material PRIVATE optix_util)

set_target_properties(material PROPERTIES FOLDER ${source_folder})