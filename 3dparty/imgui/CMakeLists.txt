add_library(imgui SHARED)

set(imgui_header
    imgui/imconfig.h
    imgui/imgui.h
    imgui/backends/imgui_impl_dx12.h
    imgui/backends/imgui_impl_win32.h
)

set(imgui_source
    imgui/imgui.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_dx12.cpp
    imgui/backends/imgui_impl_win32.cpp
)

target_sources(imgui PRIVATE ${imgui_header} ${imgui_source})

list(APPEND files ${imgui_header})
set(imgui_include_dir ${CMAKE_CURRENT_BINARY_DIR}/include)
file(MAKE_DIRECTORY ${imgui_include_dir}/imgui)
foreach(i IN LISTS files)
    configure_file(${i} ${imgui_include_dir}/imgui COPYONLY)
endforeach()
target_include_directories(imgui PUBLIC ${imgui_include_dir})

target_link_libraries(imgui PUBLIC d3d12 d3dcompiler dxgi dxguid)
set_target_properties(imgui PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)