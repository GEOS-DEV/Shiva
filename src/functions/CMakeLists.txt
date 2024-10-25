
set( functions_headers
     bases/BasisProduct.hpp
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

foreach( _header ${functions_headers} )
     get_filename_component( _header_dir ${_header} DIRECTORY )
     install( FILES ${_header} 
              DESTINATION include/functions/${_header_dir} )
endforeach()

install( TARGETS functions
         EXPORT shiva
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib ) 


if( SHIVA_ENABLE_UNIT_TESTS )
     add_subdirectory( unitTests )
endif()