
set(3rdparty_folder third-party)

add_library(${3rdparty_target} SHARED dummy.cpp)
set_target_properties(${3rdparty_target} PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ########################################################
# common function
function(set_ide_folder target_name folder_name)
    if(TARGET ${target_name})
        set_target_properties(${target_name} PROPERTIES FOLDER ${folder_name})
    else()
        message(STATUS "${target_name} does not exist.")
    endif()
endfunction()

function(add_sub_target lib_name target_name folder_name include_dir)
    if(NOT TARGET ${target_name})
        message(STATUS "============start config ${target_name}============")
        add_subdirectory(${lib_name})

        target_link_libraries(${3rdparty_target} PUBLIC ${target_name})

        if(NOT(include_dir EQUAL ""))
            target_include_directories(${target_name} PUBLIC ${include_dir})
        endif()

        set_ide_folder(${target_name} ${folder_name})
    endif()
endfunction()

# ########################################################
set_ide_folder(${3rdparty_target} ${3rdparty_folder})

add_custom_target(file-copy COMMENT "Copy necessary files for third-party libraries")
set_ide_folder(file-copy ${3rdparty_folder})

set(ASSIMP_BUILD_OBJ_IMPORTER ON CACHE INTERNAL "" FORCE)
set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE INTERNAL "" FORCE)
set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE INTERNAL "" FORCE)
set(ASSIMP_BUILD_TESTS OFF CACHE INTERNAL "" FORCE)
set(ASSIMP_INSTALL OFF CACHE INTERNAL "" FORCE)
add_sub_target(assimp assimp ${3rdparty_folder}/assimp "")
set_ide_folder(UpdateAssimpLibsDebugSymbolsAndDLLs ${3rdparty_folder}/assimp)
set_ide_folder(zlibstatic ${3rdparty_folder}/assimp)

target_link_libraries(${3rdparty_target} PUBLIC assimp)
target_include_directories(${3rdparty_target} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/assimp)

add_sub_target(stb stb ${3rdparty_folder} ${CMAKE_CURRENT_SOURCE_DIR}/stb)
add_sub_target(imgui imgui ${3rdparty_folder} ${CMAKE_CURRENT_SOURCE_DIR}/imgui/imgui)

set(BUILD_TESTING OFF CACHE INTERNAL "" FORCE)
add_sub_target(DirectXTK12 DirectXTK12 ${3rdparty_folder} "")

set_property(GLOBAL PROPERTY CTEST_TARGETS_ADDED 1)
set(BUILD_SHARED_LIBS ON CACHE INTERNAL "" FORCE)
add_sub_target(pugixml pugixml-shared ${3rdparty_folder} "")

add_sub_target(spdlog spdlog ${3rdparty_folder} "")

add_sub_target(exr tinyexr ${3rdparty_folder}/tinyexr "")
set_ide_folder(miniz ${3rdparty_folder}/tinyexr)