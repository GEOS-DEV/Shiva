
set( shiva_headers
     discretizations/finiteElementMethod/parentElements/ParentElement.hpp
     PARENT_SCOPE
     )

set( shiva_sources
     )

if( SHIVA_ENABLE_UNIT_TESTS )
     add_subdirectory( unitTests )
endif()