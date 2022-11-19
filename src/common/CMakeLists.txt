add_library(common INTERFACE)

target_sources(common
    PUBLIC
        util.h
)
set_target_properties(common PROPERTIES FOLDER ${source_folder})