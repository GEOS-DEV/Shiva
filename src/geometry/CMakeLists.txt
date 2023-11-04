
set( geometry_headers
     geometryUtilities.hpp
     mapping/LinearTransform.hpp
     mapping/Scaling.hpp
     mapping/UniformScaling.hpp
     shapes/nCube.hpp
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

install( FILES ${geometry_headers} 
         DESTINATION include/geometry )

install( TARGETS geometry
         EXPORT shiva-targets
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib ) 

add_subdirectory( mapping/unitTests )
add_subdirectory( shapes/unitTests )
