
set( discretizations_headers
     finiteElementMethod/parentElements/ParentElement.hpp
     )

set( discretizations_sources
     )

set( discretizations_dependencies
     common
     functions
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
