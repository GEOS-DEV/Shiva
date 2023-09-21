
set( shiva_headers
     ${shiva_headers}
     discretizations/finiteElementMethod/bases/LagrangeBasis.hpp
     discretizations/spacing/Spacing.hpp
     discretizations/quadrature/Quadrature.hpp
     PARENT_SCOPE 
     )

add_subdirectory( unitTests )
