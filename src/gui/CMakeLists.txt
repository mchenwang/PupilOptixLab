add_library(gui SHARED)

target_sources(gui
    PUBLIC
        window.h
        window.cpp
        dx12_backend.h
        dx12_backend.cpp
)
target_link_libraries(gui PUBLIC imgui)
set_target_properties(gui PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

set_target_properties(gui PROPERTIES FOLDER ${source_folder})