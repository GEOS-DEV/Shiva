#
# Specify list of tests
#

set( unit_tests_sources
     testCellHexahedron.cpp
     testCellUtilities.cpp
     )

set( dependencyList ${parallelDeps} gtest geometry )

#
# Add gtest C++ based tests
#
foreach(test ${unit_tests_sources})
    message("test is ${test}")
    get_filename_component( test_name ${test} NAME_WE )
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test}
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON ${dependencyList}
                        )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name}
                  )
endforeach()
