cmake_minimum_required(VERSION 3.23.1)
project(Shiva VERSION 0.1.0)

# Portable install dir vars: ${CMAKE_INSTALL_INCLUDEDIR}, ${CMAKE_INSTALL_LIBDIR}, etc.
include(GNUInstallDirs)

# -------- Options (lean) -------------------------------------------------------
option( SHIVA_ENABLE_BLT           "Enable BLT integration (testing, packaging, etc.)" ON )
option( SHIVA_ENABLE_CUDA          "Enable CUDA code paths in headers (interface define only)" OFF )
option( SHIVA_ENABLE_HIP           "Enable HIP code paths in headers  (interface define only)" OFF )
option( SHIVA_ENABLE_UNIT_TESTS    "Build unit tests (standalone only)" ON )
option( SHIVA_ENABLE_CAMP          "Link against CAMP if available" OFF )
option( SHIVA_BUILD_OBJ_LIBS       "Build object libraries...useful for dependency trees" OFF )
option( SHIVA_ENABLE_BOUNDS_CHECK  "Enable bounds checking in shiva::CArray" OFF )

if( ENABLE_CUDA )
  set( SHIVA_ENABLE_CUDA ON )
endif()

if( ENABLE_HIP )
  set( SHIVA_ENABLE_HIP ON )
endif()

if( SHIVA_ENABLE_CUDA AND SHIVA_ENABLE_HIP )
  message( FATAL_ERROR "CUDA and HIP are mutually exclusive; enable only one." )
endif()


# Top-level or subproject?
set( SHIVA_IS_TOPLEVEL "${PROJECT_IS_TOP_LEVEL}" )

# Convenience dirs
set( SHIVA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}" )
set( SHIVA_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}" )

# -------- Generate config header ----------------------------------------------
# Expects: include/shiva/ShivaConfig.hpp.in
include( "${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake" )

# -------- Header-only target ---------------------------------------------------
add_library(Shiva INTERFACE)
add_library(Shiva::shiva ALIAS Shiva)

target_include_directories( Shiva INTERFACE
                            "$<BUILD_INTERFACE:${SHIVA_SOURCE_DIR}/include>"   # include/shiva/...
                            "$<BUILD_INTERFACE:${SHIVA_BINARY_DIR}/include>"   # build/include/shiva/ShivaConfig.hpp
                            "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>" )

target_compile_features(Shiva INTERFACE cxx_std_17)




# -------- BLT (optional; usable as top-level or submodule) ---------------------
if( SHIVA_ENABLE_BLT )
  set( BLT_CXX_STD "c++17" )
  # Load BLT if Shiva is standalone
  if( SHIVA_IS_TOPLEVEL )
    set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt" CACHE PATH "Path to BLT")
    if (NOT EXISTS "${BLT_SOURCE_DIR}/SetupBLT.cmake")
      message(FATAL_ERROR "BLT not found at '${BLT_SOURCE_DIR}'.")
    endif()
    include("${BLT_SOURCE_DIR}/SetupBLT.cmake")
  else()

  endif( SHIVA_IS_TOPLEVEL )


  if( BLT_LOADED )
    message( STATUS "BLT loaded successfully." )
  else()
    message( FATAL_ERROR "BLT requested but failed to load." )
  endif( BLT_LOADED )



  if( SHIVA_ENABLE_CAMP )
      add_subdirectory(tpl/camp)          # vendored fallback when standalone
      target_link_libraries(Shiva INTERFACE camp)
  endif()

  if( SHIVA_ENABLE_UNIT_TESTS )
    # Enable GPU languages **only** for tests/examples at the top level
    if ( SHIVA_ENABLE_CUDA AND NOT CMAKE_CUDA_COMPILER )
      enable_language ( CUDA )
    endif ()
    if ( SHIVA_ENABLE_HIP  AND NOT CMAKE_HIP_COMPILER )
      enable_language ( HIP )
    endif ()

    add_subdirectory(tests)
  endif()

  include( "${CMAKE_CURRENT_LIST_DIR}/cmake/Macros.cmake" )
  shiva_add_code_checks( PREFIX shiva
                         EXCLUDES "blt/*" )

endif( SHIVA_ENABLE_BLT)


# -------- Install & package ----------------------------------------------------
install(TARGETS Shiva EXPORT ShivaTargets)

# Install public headers and generated config header tree
install( DIRECTORY "${SHIVA_SOURCE_DIR}/include/"
         DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" )
install( DIRECTORY "${SHIVA_BINARY_DIR}/include/"
         DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
         OPTIONAL )

include( CMakePackageConfigHelpers )

write_basic_package_version_file( "${CMAKE_CURRENT_BINARY_DIR}/ShivaConfigVersion.cmake"
                                  VERSION "${PROJECT_VERSION}"
                                  COMPATIBILITY SameMajorVersion )

configure_package_config_file( "${CMAKE_CURRENT_LIST_DIR}/cmake/ShivaConfig.cmake.in"
                               "${CMAKE_CURRENT_BINARY_DIR}/ShivaConfig.cmake"
                               INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Shiva" )

install( FILES "${CMAKE_CURRENT_BINARY_DIR}/ShivaConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/ShivaConfigVersion.cmake"
         DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Shiva" )
install( EXPORT ShivaTargets
         FILE ShivaTargets.cmake
         NAMESPACE Shiva::
         DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Shiva" )
