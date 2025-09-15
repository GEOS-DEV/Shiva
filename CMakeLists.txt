cmake_minimum_required(VERSION 3.23.1)

# Standalone project metadata
project(Shiva VERSION 0.1.0 )

# Are we the top-level project?
# (Reliable with CMake >= 3.21)
if (DEFINED PROJECT_IS_TOP_LEVEL)
  set(SHIVA_IS_TOPLEVEL "${PROJECT_IS_TOP_LEVEL}")
else()
  # Fallback (shouldn’t be needed given min version)
  if (CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    set(SHIVA_IS_TOPLEVEL TRUE)
  else()
    set(SHIVA_IS_TOPLEVEL FALSE)
  endif()
endif()

# Options
option( SHIVA_ENABLE_BLT       "Enable BLT usage (standalone default ON)" ON )
option( SHIVA_ENABLE_CAMP      "Enable CAMP usage (requires BLT in this repo layout)" ON )
option( CAMP_ENABLE_TESTS      "Enable CAMP tests (if CAMP is added here)" OFF )

# Shiva’s public config toggles (exported in INTERFACE)
option(SHIVA_ENABLE_UNIT_TESTS "Builds tests (standalone only)" ON)
option(SHIVA_ENABLE_EXAMPLES   "Builds examples (standalone only)" ON)
option(SHIVA_ENABLE_BENCHMARKS "Builds benchmarks (standalone only)" ON)
option(SHIVA_ENABLE_DOCS       "Builds documentation (standalone only)" ON)

# Device toggles are *interface* concerns for consumers; we do not enable languages here.
option(SHIVA_USE_CUDA "Consumer intends to use CUDA with Shiva headers" OFF)
option(SHIVA_USE_HIP  "Consumer intends to use HIP with Shiva headers"  OFF)





# BLT (standalone mode only, and only if requested)
if (SHIVA_IS_TOPLEVEL)
  message(STATUS "Building Shiva as a standalone project")
  
  enable_language(CXX)
  if( SHIVA_USE_CUDA )
    enable_language(CUDA)
  endif()

  set( SHIVA_ENABLE_BLT ON CACHE BOOL "Force ON in standalone mode" )  
  set( SHIVA_ENABLE_CAMP ON CACHE BOOL "Force ON in standalone mode" ) 

  # Where to find BLT if vendored
  set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt" CACHE PATH "Path to BLT")
  if (NOT EXISTS "${BLT_SOURCE_DIR}/SetupBLT.cmake")
    message(FATAL_ERROR
      "BLT not found at '${BLT_SOURCE_DIR}'. If this repository vendors BLT as a submodule, run:\n"
      "  git submodule update --init --recursive\n"
      "Or set -DBLT_SOURCE_DIR=/path/to/blt")
  endif()

  # Standardized C++ level for BLT toolchains
  set(BLT_CXX_STD "c++17" CACHE STRING "Version of C++ standard" FORCE)
  include("${BLT_SOURCE_DIR}/SetupBLT.cmake")
  message(STATUS "BLT loaded")
else()
  message(STATUS "Using Shiva as a subproject/submodule")
  if (SHIVA_ENABLE_CAMP OR SHIVA_ENABLE_BLT)
    if (NOT BLT_LOADED)
      message(FATAL_ERROR
        "Parent requested BLT/CAMP features, but BLT is not loaded. "
        "Please load BLT in the top-level project before adding Shiva.")
    endif()
  endif()
endif()

# Basic include paths (source + generated)
set(SHIVA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SHIVA_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# Generate Shiva’s config header (your script)
include("${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake")

# Create the header-only target unconditionally.
add_library(Shiva INTERFACE)
add_library(Shiva::shiva ALIAS Shiva)

# Public include dirs (source + current binary for generated headers)
target_include_directories( Shiva
                            INTERFACE
                              "$<BUILD_INTERFACE:${SHIVA_SOURCE_DIR}/src>"
                              "$<BUILD_INTERFACE:${SHIVA_BINARY_DIR}/include>"
                              "$<INSTALL_INTERFACE:include>"
)

# Public compile features and interface defines (but no language enable here)
target_compile_features(Shiva INTERFACE cxx_std_17)

# Device feature flags are exported as interface macros for consumers to react to in headers
if (SHIVA_USE_CUDA)
  target_compile_definitions(Shiva INTERFACE SHIVA_USE_CUDA=1)
endif()
if (SHIVA_USE_HIP)
  target_compile_definitions(Shiva INTERFACE SHIVA_USE_HIP=1)
endif()

# CAMP (only wire it up here if explicitly requested)
if (SHIVA_ENABLE_CAMP)
  # Option A (vendored via BLT/this repo): add_subdirectory
  # Option B (preferred long-term): find_package(camp CONFIG REQUIRED) and link camp::camp
  # Keeping your current approach but safer paths:
  add_subdirectory(tpl/camp)
  # Generated header into *current* binary dir to avoid collisions when subproject
  configure_file(
    tpl/camp/include/camp/config.in.hpp
    "${SHIVA_BINARY_DIR}/include/camp/config.hpp"
    @ONLY
  )
  # The 'camp' target comes from the subdir. Be gentle with warnings:
  if (TARGET camp)
    # Example: silence -Wshadow on GCC/Clang only
    target_compile_options(camp PRIVATE
      $<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-Wno-shadow>
    )
    # Export dependency to consumers
    target_link_libraries(Shiva INTERFACE camp)
  endif()
endif()

# Put the rest of your target wiring (headers-only structure, subdirs) under src/
# Keep 'src' responsible for populating the header tree and optional tests/examples/etc.
add_subdirectory(include)

# Standalone-only extras
if (SHIVA_IS_TOPLEVEL)
  if (SHIVA_ENABLE_DOCS)
    add_subdirectory(docs)
  endif()


  # Minimal install/export so consumers can find_package(Shiva)
  include(GNUInstallDirs)
  install(TARGETS Shiva
    EXPORT ShivaTargets
  )
  install(DIRECTORY "${SHIVA_SOURCE_DIR}/src/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
  )
  # Also install generated headers if any
  install(DIRECTORY "${SHIVA_BINARY_DIR}/include/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    OPTIONAL
  )

  include(CMakePackageConfigHelpers)
  write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ShivaConfigVersion.cmake"
    VERSION "${PROJECT_VERSION}"
    COMPATIBILITY SameMajorVersion
  )
  configure_package_config_file(
    "${CMAKE_CURRENT_LIST_DIR}/cmake/ShivaConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ShivaConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Shiva"
  )
  install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ShivaConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ShivaConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Shiva"
  )
  install(EXPORT ShivaTargets
    NAMESPACE Shiva::
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Shiva"
  )
endif()


if( SHIVA_ENABLE_UNIT_TESTS )
  add_subdirectory(tests)
endif()
