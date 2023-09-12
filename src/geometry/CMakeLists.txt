

set( shiva_headers
     ${shiva_headers}
     geometry/Cube.hpp
     geometry/Cuboid.hpp
     geometry/RectangularCuboid.hpp
     geometry/geometryUtilities.hpp
     PARENT_SCOPE 
     )

add_subdirectory( unitTests )
