
set( types_headers
     IndexTypes.hpp
     LinearIndex.hpp
     MultiIndex.hpp
     types.hpp
     )

set( shiva_headers
     ${shiva_headers}
     types/MultiIndex.hpp
     types/types.hpp
     PARENT_SCOPE 
     )


add_subdirectory( unitTests )
