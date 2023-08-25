
set( geometry_headers
     CellHexahedron.hpp
     )

set( geometry_sources
     )

blt_add_library( NAME             geometry
#                 SOURCES          ${geometry_sources}
                 HEADERS          ${geometry_headers}
                 DEPENDS_ON       ${geometry_dependencies}
                 )

target_include_directories( geometry 
                            INTERFACE 
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> )


install( FILES ${geometry_headers} 
         DESTINATION include/geometry )

install( TARGETS geometry
         EXPORT geometry
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib ) 

install( EXPORT geometry
         DESTINATION share/shiva/cmake/ )


add_subdirectory( unitTests )
