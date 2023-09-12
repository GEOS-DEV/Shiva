
set( common_headers
     MathUtilities.hpp
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
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> )


install( FILES ${common_headers} 
         DESTINATION include/common )

install( TARGETS common
         EXPORT common
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib ) 

install( EXPORT common
         DESTINATION share/shiva/cmake/ )


#add_subdirectory( unitTests )
