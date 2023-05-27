set(target_name empty_gui)

add_executable(${target_name} main.cpp)

target_link_libraries(${target_name} PUBLIC ${pupil_framework_name})
set_target_properties(${target_name} PROPERTIES FOLDER ${pupil_application_example})
