set(gui_source
    ${STATIC_DATA_FILE}
    window.h
    window.cpp
    dx12_backend.h
    dx12_backend.cpp
)

add_library(gui STATIC ${gui_source})
target_include_directories(gui PRIVATE ../)

target_link_libraries(gui PUBLIC imgui)
target_link_libraries(gui PRIVATE device_api)
target_link_libraries(gui PRIVATE common)
target_link_libraries(gui PRIVATE DirectXTK12)
target_link_libraries(gui PRIVATE stb)
target_link_libraries(gui PRIVATE cuda_util)

set_target_properties(gui PROPERTIES FOLDER ${source_folder})