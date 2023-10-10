
set( common_headers
     MathUtilities.hpp
     pmpl.hpp
     SequenceUtilities.hpp
     ShivaMacros.hpp
     )

set( common_sources
     )

blt_add_library( NAME             common
#                 SOURCES          ${common_sources}
                 HEADERS          ${common_headers}
                 DEPENDS_ON       ${common_dependencies}
                 )

target_include_directories( common 
                            INTERFACE 
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
                            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src> )


install( FILES ${common_headers} 
         DESTINATION include/common )

install( TARGETS common
         EXPORT shiva-targets
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib ) 

add_subdirectory( unitTests )
