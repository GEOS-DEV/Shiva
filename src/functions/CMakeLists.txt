
set( functions_headers
     bases/LagrangeBasis.hpp
     quadrature/Quadrature.hpp
     spacing/Spacing.hpp
     )

set( functions_sources
     )

set( functions_dependencies
     common
     )

blt_add_library( NAME             functions
#                 SOURCES          ${functions_sources}
                 HEADERS          ${functions_headers}
                 DEPENDS_ON       ${functions_dependencies}
                 )

install( FILES ${functions_headers} 
         DESTINATION include/functions )

install( TARGETS functions
         EXPORT shiva-targets
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib ) 


add_subdirectory( unitTests )
