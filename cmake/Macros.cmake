macro(shiva_add_code_checks)

    set(options)
    set(singleValueArgs PREFIX UNCRUSTIFY_CFG_FILE )
    set(multiValueArgs EXCLUDES )

    # Parse the arguments to the macro
    cmake_parse_arguments(arg
         "${options}" "${singleValueArgs}" "${multiValueArgs}" ${ARGN})

    set(_all_sources)
    file(GLOB_RECURSE _all_sources
         "*.cpp" "*.hpp" "*.cxx" "*.hxx" "*.cc" "*.c" "*.h" "*.hh" )

    # Check for excludes
    if (NOT DEFINED arg_EXCLUDES)
        set(_sources ${_all_sources})
    else()
        set(_sources)
        foreach(_source ${_all_sources})
            set(_to_be_excluded FALSE)
            foreach(_exclude ${arg_EXCLUDES})
                if (${_source} MATCHES ${_exclude})
                    set(_to_be_excluded TRUE)
                    break()
                endif()
            endforeach()

            if (NOT ${_to_be_excluded})
                list(APPEND _sources ${_source})
            endif()
        endforeach()
    endif()

    set( CPPCHECK_FLAGS --std=c++17 
                        --enable=all 
                        --suppress=missingIncludeSystem 
                        --suppress=missingInclude 
                        --suppress=noConstructor 
                        --suppress=unusedFunction 
                        --suppress=constStatement 
                        --suppress=unusedStructMember )
                        
    blt_add_code_checks( PREFIX    ${arg_PREFIX}
                         SOURCES   ${_sources}
                         UNCRUSTIFY_CFG_FILE ${PROJECT_SOURCE_DIR}/src/uncrustify.cfg
                         CPPCHECK_FLAGS ${CPPCHECK_FLAGS}
                         D
                         )


endmacro(shiva_add_code_checks)