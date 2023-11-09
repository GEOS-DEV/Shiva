
set( common_headers
     IndexTypes.hpp
     LinearIndex.hpp
     MathUtilities.hpp
     MultiIndex.hpp
     pmpl.hpp
     SequenceUtilities.hpp
     NestedSequenceUtilities.hpp
     ShivaMacros.hpp
     types.hpp
     )

set( common_sources
     )

if( ENABLE_HIP )
    list( APPEND common_dependencies blt::hip )
endif()

if( ENABLE_CUDA )
    list( APPEND common_dependencies cuda )
endif()

blt_add_library( NAME             common
#                 SOURCES          ${common_sources}
                 HEADERS          ${common_headers}
                 DEPENDS_ON       ${common_dependencies}
                 )

target_include_directories( common 
                            INTERFACE 
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
                            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src> )

target_include_directories( common 
                            SYSTEM INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/tpl/camp/include>
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/tpl/camp/include> )


install( FILES ${common_headers} 
         DESTINATION include/common )

install( TARGETS common
         EXPORT shiva-targets
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib ) 

add_subdirectory( unitTests )
