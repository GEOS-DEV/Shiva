
set( shiva_headers
     Shiva.hpp
     )

set( shiva_dependencies )


if( SHIVA_ENABLE_CAMP )
  list( APPEND shiva_dependencies camp )
endif()

if( SHIVA_ENABLE_BLT AND ENABLE_HIP )
    list( APPEND shiva_dependencies blt::hip )
endif()

if( ENABLE_CUDA )
    list( APPEND shiva_dependencies cuda )
endif()

add_subdirectory( common )
add_subdirectory( functions )
add_subdirectory( discretizations )
add_subdirectory( geometry )




add_library( shiva INTERFACE ${shiva_headers} )

target_include_directories( shiva
                            INTERFACE
                            ${shiva_headers} )

install( FILES ${shiva_headers} 
         DESTINATION include )


install( TARGETS shiva
         EXPORT shiva
         ARCHIVE DESTINATION lib
         LIBRARY DESTINATION lib
         RUNTIME DESTINATION lib )

if( SHIVA_ENABLE_BLT )

  install( EXPORT shiva
           DESTINATION lib/cmake/shiva )

  shiva_add_code_checks( PREFIX shiva
                        EXCLUDES "blt/*" )
endif()