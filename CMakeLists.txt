cmake_minimum_required(VERSION 3.23.1)
project(Shiva VERSION 0.1.0)

# Portable install dir vars: ${CMAKE_INSTALL_INCLUDEDIR}, ${CMAKE_INSTALL_LIBDIR}, etc.
include(GNUInstallDirs)

# -------- Options (lean) -------------------------------------------------------
option(SHIVA_USE_CUDA          "Enable CUDA code paths in headers (interface define only)" OFF)
option(SHIVA_USE_HIP           "Enable HIP code paths in headers  (interface define only)" OFF)
option(SHIVA_ENABLE_UNIT_TESTS "Build unit tests (standalone only)" ON)
option(SHIVA_ENABLE_CAMP       "Link against CAMP if available" OFF)

# Top-level or subproject?
set(SHIVA_IS_TOPLEVEL "${PROJECT_IS_TOP_LEVEL}")

# Convenience dirs
set(SHIVA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
set(SHIVA_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

# -------- Generate config header ----------------------------------------------
# Expects: include/shiva/ShivaConfig.hpp.in
include("${CMAKE_CURRENT_LIST_DIR}/cmake/Config.cmake")

# -------- Header-only target ---------------------------------------------------
add_library(Shiva INTERFACE)
add_library(Shiva::shiva ALIAS Shiva)

target_include_directories(Shiva INTERFACE
  "$<BUILD_INTERFACE:${SHIVA_SOURCE_DIR}/include>"   # include/shiva/...
  "$<BUILD_INTERFACE:${SHIVA_BINARY_DIR}/include>"   # build/include/shiva/ShivaConfig.hpp
  "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

target_compile_features(Shiva INTERFACE cxx_std_17)

if (SHIVA_USE_CUDA)
  target_compile_definitions(Shiva INTERFACE SHIVA_USE_CUDA=1)
endif()
if (SHIVA_USE_HIP)
  target_compile_definitions(Shiva INTERFACE SHIVA_USE_HIP=1)
endif()

# Optional CAMP wiring (no BLT required here)
if (SHIVA_ENABLE_CAMP)
  find_package(camp CONFIG QUIET)
  if (TARGET camp)
    target_link_libraries(Shiva INTERFACE camp)
  elseif (SHIVA_IS_TOPLEVEL)
    add_subdirectory(tpl/camp)          # vendored fallback when standalone
    target_link_libraries(Shiva INTERFACE camp)
  else()
    message(FATAL_ERROR "CAMP not found; enable it only if the parent provides it or build standalone.")
  endif()
endif()

# -------- Tests (BLT only when standalone) -------------------------------------
if (SHIVA_IS_TOPLEVEL AND SHIVA_ENABLE_UNIT_TESTS)
  include(CTest)
  # Load BLT you vendor under cmake/blt
  set(BLT_SOURCE_DIR "${PROJECT_SOURCE_DIR}/cmake/blt" CACHE PATH "Path to BLT")
  if (NOT EXISTS "${BLT_SOURCE_DIR}/SetupBLT.cmake")
    message(FATAL_ERROR "BLT not found at '${BLT_SOURCE_DIR}'.")
  endif()
  include("${BLT_SOURCE_DIR}/SetupBLT.cmake")
  add_subdirectory(tests)
endif()


# -------- Install & package ----------------------------------------------------
install(TARGETS Shiva EXPORT ShivaTargets)

# Install public headers and generated config header tree
install(DIRECTORY "${SHIVA_SOURCE_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
install(DIRECTORY "${SHIVA_BINARY_DIR}/include/"
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
        OPTIONAL)

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
  FILE ShivaTargets.cmake
  NAMESPACE Shiva::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/Shiva"
)
