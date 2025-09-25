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
                        --quiet
                        --suppress=missingIncludeSystem
                        --suppress=unmatchedSuppression
                        --suppress=missingInclude 
                        --suppress=noConstructor 
                        --suppress=noExplicitConstructor
                        --suppress=unusedFunction 
                        --suppress=constStatement 
                        --suppress=unusedStructMember
                        --suppress=unknownMacro )
                        
    blt_add_code_checks( PREFIX    ${arg_PREFIX}
                         SOURCES   ${_sources}
                         UNCRUSTIFY_CFG_FILE ${PROJECT_SOURCE_DIR}/src/uncrustify.cfg
                         CPPCHECK_FLAGS ${CPPCHECK_FLAGS}
                         )

    if( CPPCHECK_FOUND )
        add_test( NAME testCppCheck
                COMMAND bash -c "make cppcheck_check 2> >(tee cppcheck.err) >/dev/null && exit $(cat cppcheck.err | wc -l)"
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                )
    endif()

    if( CLANGTIDY_FOUND )
        add_test( NAME testClangTidy
                COMMAND bash -c "make clang_tidy_check 2> >(tee tidyCheck.err) >/dev/null && exit $(cat tidyCheck.err | wc -l)"
                WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
                )
    endif()


    if (ENABLE_COVERAGE)
        blt_add_code_coverage_target(NAME   ${arg_PREFIX}_coverage
                                     RUNNER ctest -E 'blt_gtest_smoke|testCppCheck|testClangTidy|testUncrustifyCheck|testDoxygenCheck|testCppCheck|testClangTidy'
                                     SOURCE_DIRECTORIES ${PROJECT_SOURCE_DIR}/src )
    endif()
endmacro(shiva_add_code_checks)