
set( types_headers
     MultiIndex.hpp
     )

set( types_sources
     )

blt_add_library( NAME             types
#                 SOURCES          ${types_sources}
                 HEADERS          ${types_headers}
                 DEPENDS_ON       ${types_dependencies}
                 )

target_include_directories( types 
                            INTERFACE 
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> )


install( FILES ${types_headers} 
         DESTINATION include/types )

install( TARGETS types
         EXPORT types
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib ) 

install( EXPORT types
         DESTINATION share/shiva/cmake/ )


#add_subdirectory( unitTests )
