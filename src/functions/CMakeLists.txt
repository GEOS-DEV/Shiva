
set( shiva_headers
     functions/bases/BasisProduct.hpp
     functions/bases/LagrangeBasis.hpp
     functions/quadrature/Quadrature.hpp
     functions/spacing/Spacing.hpp
     PARENT_SCOPE 
     )

if( SHIVA_ENABLE_UNIT_TESTS )
     add_subdirectory( unitTests )
endif()