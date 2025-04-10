#
set( PREPROCESSOR_DEFINES CUDA
                          HIP
                          CAMP
                          BOUNDS_CHECK
                        )

set( USE_CONFIGFILE ON CACHE BOOL "" )
foreach( DEP in ${PREPROCESSOR_DEFINES})
    if( ${DEP}_FOUND OR ENABLE_${DEP} OR SHIVA_ENABLE_${DEP} )
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

# Set up cmake package config file

set(SHIVA_INSTALL_INCLUDE_DIR "include" CACHE STRING "")
set(SHIVA_INSTALL_CONFIG_DIR "lib" CACHE STRING "")
set(SHIVA_INSTALL_LIB_DIR "lib" CACHE STRING "")
set(SHIVA_INSTALL_BIN_DIR "bin" CACHE STRING "")
set(SHIVA_INSTALL_CMAKE_MODULE_DIR "${SHIVA_INSTALL_CONFIG_DIR}/cmake" CACHE STRING "")
set(SHIVA_INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX} CACHE STRING "" FORCE)


include(CMakePackageConfigHelpers)
configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/shiva-config.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/shiva-config.cmake
  INSTALL_DESTINATION
    ${SHIVA_INSTALL_CONFIG_DIR}
  PATH_VARS
    SHIVA_INSTALL_INCLUDE_DIR
    SHIVA_INSTALL_LIB_DIR
    SHIVA_INSTALL_BIN_DIR
    SHIVA_INSTALL_CMAKE_MODULE_DIR
  )


install( FILES ${CMAKE_CURRENT_BINARY_DIR}/shiva-config.cmake
         DESTINATION share/shiva/cmake/)
