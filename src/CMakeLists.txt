
set( shiva_headers
     Shiva.hpp
     )

set( shiva_sources
     )

set( shiva_dependencies
     geometry
     )

add_subdirectory( common )
add_subdirectory( functions )
add_subdirectory( discretizations )
add_subdirectory( geometry )

blt_add_library( NAME             shiva
#                 SOURCES          ${shiva_sources}
                 HEADERS          ${shiva_headers}
                 DEPENDS_ON       ${shiva_dependencies}
                )

target_include_directories( shiva
                            INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include>
                            $<INSTALL_INTERFACE:include> )

install( FILES ${shiva_headers} 
         DESTINATION include )


install( TARGETS shiva
         EXPORT shiva-targets
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib )

install( EXPORT shiva-targets
         DESTINATION lib/cmake/shiva )

shiva_add_code_checks( PREFIX shiva
                       EXCLUDES "blt/*" )
