#
# Specify list of tests
#

set( unit_tests_sources
     testUniformScaling.cpp
     testLinearTransform.cpp
     testScaling.cpp
     )

set( dependencyList ${parallelDeps} gtest geometry )

if( ENABLE_HIP )
    list( APPEND dependencyList blt::hip )
endif()

if( ENABLE_CUDA )
    list( APPEND dependencyList cuda )
endif()

#
# Add gtest C++ based tests
#
foreach(test ${unit_tests_sources})
#    message("test is ${test}")
    set( header ${test} )
    string(REPLACE "test" "../" header ${header})
    string(REPLACE ".cpp" ".hpp" header ${header})
#    message("header is ${header}")

    get_filename_component( test_name ${test} NAME_WE )
    blt_add_executable( NAME ${test_name}
                        SOURCES ${test}
                        HEADERS ${header}
                        OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
                        DEPENDS_ON ${dependencyList}
                        )

    blt_add_test( NAME ${test_name}
                  COMMAND ${test_name}
                  )
endforeach()
