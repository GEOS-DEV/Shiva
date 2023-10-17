
set( discretizations_headers
     finiteElementMethod/bases/LagrangeBasis.hpp
     finiteElementMethod/parentElements/ParentElement.hpp
     spacing/Spacing.hpp
     quadrature/Quadrature.hpp
     )

set( discretizations_sources
     )

set( discretizations_dependencies
     common
     )

blt_add_library( NAME             discretizations
#                 SOURCES          ${discretizations_sources}
                 HEADERS          ${discretizations_headers}
                 DEPENDS_ON       ${discretizations_dependencies}
                 )

install( FILES ${discretizations_headers} 
         DESTINATION include/discretizations )

install( TARGETS discretizations
         EXPORT shiva-targets
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib ) 


add_subdirectory( unitTests )
