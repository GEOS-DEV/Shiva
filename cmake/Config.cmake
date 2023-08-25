#
set( PREPROCESSOR_DEFINES CUDA
                          HIP
                        )

set( USE_CONFIGFILE ON CACHE BOOL "" )
foreach( DEP in ${PREPROCESSOR_DEFINES})
    if( ${DEP}_FOUND OR ENABLE_${DEP} )
        set( SHIVA_USE_${DEP} TRUE )
    endif()
endforeach()

if( ENABLE_ADDR2LINE )
    if ( NOT DEFINED ADDR2LINE_EXEC )
        set( ADDR2LINE_EXEC /usr/bin/addr2line CACHE PATH "" )
    endif()

    if ( NOT EXISTS ${ADDR2LINE_EXEC} )
        message( FATAL_ERROR "The addr2line executable does not exist: ${ADDR2LINE_EXEC}" )
    endif()

    set( SHIVA_ADDR2LINE_EXEC ${ADDR2LINE_EXEC} )
endif()


configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/src/ShivaConfig.hpp.in
                ${CMAKE_BINARY_DIR}/include/ShivaConfig.hpp )

configure_file( ${CMAKE_CURRENT_SOURCE_DIR}/src/ShivaConfig.hpp.in
                ${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen/ShivaConfig.hpp )

# Install the generated header.
install( FILES ${CMAKE_BINARY_DIR}/include/ShivaConfig.hpp
         DESTINATION include )

# Configure and install the CMake config
configure_file( ${CMAKE_CURRENT_LIST_DIR}/shiva-config.cmake.in
                ${PROJECT_BINARY_DIR}/share/shiva/cmake/shiva-config.cmake)

install( FILES ${PROJECT_BINARY_DIR}/share/shiva/cmake/shiva-config.cmake
         DESTINATION share/shiva/cmake/)
