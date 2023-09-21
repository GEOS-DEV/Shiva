
set( discretization_headers
     finiteElementMethod/bases/LagrangeBasis.hpp
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

install( FILES ${discretization_headers} 
         DESTINATION include/discretization )

install( TARGETS discretization
         EXPORT shiva-targets
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib ) 


add_subdirectory( unitTests )
