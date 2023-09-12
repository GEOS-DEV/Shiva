
set( shiva_headers
     Shiva.hpp
     )

set( shiva_sources
     )

set( shiva_dependencies
     geometry
     )

add_subdirectory( common )
add_subdirectory( discretizations )
add_subdirectory( types )
add_subdirectory( geometry )

message( "${shiva_headers}" )

blt_add_library( NAME             shiva
#                 SOURCES          ${shiva_sources}
                 HEADERS          ${shiva_headers}
                 DEPENDS_ON       ${shiva_dependencies}
                )

target_include_directories( shiva
                            INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
                            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
                            $<INSTALL_INTERFACE:include> )

foreach( header ${shiva_headers} )
    get_filename_component(DIR ${header} DIRECTORY)
    INSTALL( FILES ${header} 
             DESTINATION include/${DIR} )
endforeach( header )



install( TARGETS shiva
         EXPORT shiva
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib )

install( EXPORT shiva
         DESTINATION share/shiva/cmake/ )

shiva_add_code_checks( PREFIX shiva
                       EXCLUDES "blt/*" )
