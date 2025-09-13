
set( shiva_headers
     geometry/mapping/JacobianTransforms.hpp
     geometry/mapping/LinearTransform.hpp
     geometry/mapping/Scaling.hpp
     geometry/mapping/UniformScaling.hpp
     geometry/shapes/InterpolatedShape.hpp
     geometry/shapes/NCube.hpp
     geometry/shapes/NSimplex.hpp
     PARENT_SCOPE 
     )

if( SHIVA_ENABLE_UNIT_TESTS )
     add_subdirectory( mapping/unitTests )
     add_subdirectory( shapes/unitTests )
endif()