# cmake/Config.cmake
include_guard(DIRECTORY)

# Expect these to be set by the parent CMakeLists:
# - SHIVA_SOURCE_DIR
# - SHIVA_BINARY_DIR

# ---------- Feature switches (normalize to SHIVA_USE_* booleans) ----------
# Start from user options if you expose them; fall back to CMake signals.

# CUDA
if (DEFINED SHIVA_USE_CUDA)
  # keep as-is
elseif (CMAKE_CUDA_COMPILER)
  set(SHIVA_USE_CUDA TRUE)
else()
  set(SHIVA_USE_CUDA FALSE)
endif()

# HIP
if (DEFINED SHIVA_USE_HIP)
  # keep as-is
elseif (DEFINED HIP_FOUND OR CMAKE_HIP_COMPILER)
  set(SHIVA_USE_HIP TRUE)
else()
  set(SHIVA_USE_HIP FALSE)
endif()

# CAMP
if (DEFINED SHIVA_USE_CAMP)
  # keep as-is
elseif (DEFINED SHIVA_ENABLE_CAMP)
  set(SHIVA_USE_CAMP "${SHIVA_ENABLE_CAMP}")
else()
  set(SHIVA_USE_CAMP FALSE)
endif()

# BOUNDS_CHECK (provide an option upstream if you want this user-tunable)
if (NOT DEFINED SHIVA_USE_BOUNDS_CHECK)
  set(SHIVA_USE_BOUNDS_CHECK FALSE)
endif()

# ---------- CUDA version numbers (if available) ----------
set(SHIVA_CUDA_MAJOR 0)
set(SHIVA_CUDA_MINOR 0)
if (SHIVA_USE_CUDA AND DEFINED CMAKE_CUDA_COMPILER_VERSION)
  # CMAKE_CUDA_COMPILER_VERSION like "12.4"
  string(REPLACE "." ";" _cuda_ver_list "${CMAKE_CUDA_COMPILER_VERSION}")
  list(GET _cuda_ver_list 0 SHIVA_CUDA_MAJOR)
  list(LENGTH _cuda_ver_list _cuda_len)
  if (_cuda_len GREATER 1)
    list(GET _cuda_ver_list 1 SHIVA_CUDA_MINOR)
  endif()
endif()

# ---------- Version numbers expected by the header ----------
# Your header expects *_PATCHLEVEL; map from project() version.
set(SHIVA_VERSION_MAJOR      "${PROJECT_VERSION_MAJOR}")
set(SHIVA_VERSION_MINOR      "${PROJECT_VERSION_MINOR}")
set(SHIVA_VERSION_PATCHLEVEL "${PROJECT_VERSION_PATCH}")

# ---------- Emit the generated header(s) ----------
set(_shiva_gen_inc "${SHIVA_BINARY_DIR}/include/shiva")
message( "SHIVA_BINARY_DIR = ${SHIVA_BINARY_DIR}" )
file(MAKE_DIRECTORY "${_shiva_gen_inc}")

configure_file( "${SHIVA_SOURCE_DIR}/include/shiva/ShivaConfig.hpp.in"
                "${_shiva_gen_inc}/ShivaConfig.hpp" )

# Optional: a copy for Doxygen without touching the source tree
set(_shiva_gen_doc "${SHIVA_BINARY_DIR}/docs/doxygen")
file(MAKE_DIRECTORY "${_shiva_gen_doc}")
configure_file( "${SHIVA_SOURCE_DIR}/include/shiva/ShivaConfig.hpp.in"
                "${_shiva_gen_doc}/ShivaConfig.hpp" )

# Install the generated header (binary include tree)
install( FILES "${_shiva_gen_inc}/ShivaConfig.hpp"
         DESTINATION include )


message( STATUS "Shiva config header -> ${SHIVA_BINARY_DIR}/include/ShivaConfig.hpp" )
