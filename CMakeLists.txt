###################################################################################################
# Copyright (c) 2021, Lawrence Livermore National Security, LLC and LvArray contributors.
# All rights reserved.
# See the LICENSE file for details.
# SPDX-License-Identifier: (BSD-3-Clause)
###################################################################################################

cmake_minimum_required( VERSION 3.23.1 )

# Set version number
set( SHIVA_VERSION_MAJOR 0 )
set( SHIVA_VERSION_MINOR 1 )
set( SHIVA_VERSION_PATCHLEVEL 0 )

# check if Shiva is build as a submodule or a separate project
get_directory_property( parent_dir PARENT_DIRECTORY )
if(parent_dir)
    set( is_submodule ON )
else()
    set( is_submodule OFF )
endif()

if( NOT is_submodule )
    message( "not a submodule")
    project( Shiva LANGUAGES CXX C )

    set( BLT_CXX_STD "c++17" CACHE STRING "Version of C++ standard" FORCE )
    set( ENABLE_WARNINGS_AS_ERRORS "ON" CACHE PATH "" )

    option( SHIVA_ENABLE_UNIT_TESTS "Builds tests" ON )
    option( SHIVA_ENABLE_EXAMPLES "Builds examples" ON )
    option( SHIVA_ENABLE_BENCHMARKS "Builds benchmarks" ON )
    option( SHIVA_ENABLE_DOCS "Builds documentation" ON )

    option( ENABLE_CUDA "Build with CUDA" OFF )
    option( ENABLE_HIP "Build with HIP" OFF )

    if( NOT BLT_LOADED )
        if( DEFINED BLT_SOURCE_DIR )
            if( NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake )
                message( FATAL_ERROR "Given BLT_SOURCE_DIR does not contain SetupBLT.cmake" )
            endif()
        else ()
            set( BLT_SOURCE_DIR ${PROJECT_SOURCE_DIR}/cmake/blt CACHE PATH "" )

            if( NOT EXISTS ${BLT_SOURCE_DIR}/SetupBLT.cmake )
                message( FATAL_ERROR "The BLT submodule is not present. If in git repository run the following two commands:\n \
                git submodule init\n \
                git submodule update" )
            endif ()
        endif ()

        include( ${BLT_SOURCE_DIR}/SetupBLT.cmake )
    endif()

    include( cmake/CMakeBasics.cmake )
    #include( cmake/SetupTPL.cmake )
else()
    if( NOT BLT_LOADED )
        message( FATAL_ERROR "When using LvArray as a submodule you must have already loaded BLT." )
    endif()
    include( cmake/CMakeBasics.cmake )
endif()

include( cmake/Macros.cmake )
include( cmake/Config.cmake )


add_subdirectory( src )
add_subdirectory( tpl/camp )
target_compile_options( camp PRIVATE "-Wno-shadow") 

# if( SHIVA_ENABLE_DOCS )
     add_subdirectory( docs )
# endif()


