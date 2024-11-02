
set( geometry_headers
     mapping/JacobianTransforms.hpp
     mapping/LinearTransform.hpp
     mapping/Scaling.hpp
     mapping/UniformScaling.hpp
     shapes/InterpolatedShape.hpp
     shapes/NCube.hpp
     shapes/NSimplex.hpp
     )

set( geometry_sources
     )

set( geometry_dependencies
     common
     )


blt_add_library( NAME             geometry
#                 SOURCES          ${geometry_sources}
                 HEADERS          ${geometry_headers}
                 DEPENDS_ON       ${geometry_dependencies}
                 )

foreach( _header ${geometry_headers} )
     get_filename_component( _header_dir ${_header} DIRECTORY )
     install( FILES ${_header} 
              DESTINATION include/geometry/${_header_dir} )
endforeach()
            
# install( FILES ${geometry_headers} 
#          DESTINATION include/geometry )

install( TARGETS geometry
         EXPORT shiva
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib ) 


if( SHIVA_ENABLE_UNIT_TESTS )
     add_subdirectory( mapping/unitTests )
     add_subdirectory( shapes/unitTests )
endif()