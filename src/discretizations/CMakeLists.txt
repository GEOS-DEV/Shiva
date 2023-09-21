
set( discretization_headers
     finiteElementMethod/bases/LagrangeBasis.hpp
     finiteElementMethod/parentElements/ParentElement.hpp
     spacing/Spacing.hpp
     quadrature/Quadrature.hpp
     )

set( discretization_sources
     )

set( discretization_dependencies
     common
     )

blt_add_library( NAME             discretization
#                 SOURCES          ${discretization_sources}
                 HEADERS          ${discretization_headers}
                 DEPENDS_ON       ${discretization_dependencies}
                 )

target_include_directories( discretization 
                            INTERFACE 
                            $<BUILD_INTERFACE:${CMAKE_BINARY_DIR}/include> )

target_include_directories( discretization 
                            INTERFACE 
                            $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src> )

install( FILES ${discretization_headers} 
         DESTINATION include/discretization )

install( TARGETS discretization
         EXPORT discretization
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib ) 

install( EXPORT discretization
         DESTINATION share/shiva/cmake/ )


add_subdirectory( unitTests )
