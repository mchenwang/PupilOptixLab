set(gui_source
    window.h
    window.cpp
    dx12_backend.h
    dx12_backend.cpp
)

add_library(gui STATIC ${gui_source})
target_include_directories(gui PRIVATE ../)

target_link_libraries(gui PRIVATE imgui)
target_link_libraries(gui PRIVATE device_api)
target_link_libraries(gui PRIVATE common)

set_target_properties(gui PROPERTIES FOLDER ${source_folder})