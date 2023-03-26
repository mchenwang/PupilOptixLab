set(imgui_header
    imgui/imconfig.h
    imgui/imgui.h
    imgui/backends/imgui_impl_dx12.h
    imgui/backends/imgui_impl_win32.h
    imgui-filebrowser/imfilebrowser.h
)

set(imgui_source
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_dx12.cpp
    imgui/backends/imgui_impl_win32.cpp
)
add_library(imgui SHARED ${imgui_header} ${imgui_source})

target_include_directories(imgui INTERFACE imgui-filebrowser/)
target_link_libraries(imgui PUBLIC d3d12 d3dcompiler dxgi dxguid)
set_target_properties(imgui PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)