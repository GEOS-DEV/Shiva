# cmake/Config.cmake
include_guard(DIRECTORY)

# ---------- Version numbers expected by the header ----------
# Your header expects *_PATCHLEVEL; map from project() version.
set(SHIVA_VERSION_MAJOR      "${PROJECT_VERSION_MAJOR}")
set(SHIVA_VERSION_MINOR      "${PROJECT_VERSION_MINOR}")
set(SHIVA_VERSION_PATCHLEVEL "${PROJECT_VERSION_PATCH}")


# ---------- CUDA version numbers (if available) ----------
set(SHIVA_CUDA_MAJOR 0)
set(SHIVA_CUDA_MINOR 0)
if (SHIVA_ENABLE_CUDA AND DEFINED CMAKE_CUDA_COMPILER_VERSION)
  # CMAKE_CUDA_COMPILER_VERSION like "12.4"
  string(REPLACE "." ";" _cuda_ver_list "${CMAKE_CUDA_COMPILER_VERSION}")
  list(GET _cuda_ver_list 0 SHIVA_CUDA_MAJOR)
  list(LENGTH _cuda_ver_list _cuda_len)
  if (_cuda_len GREATER 1)
    list(GET _cuda_ver_list 1 SHIVA_CUDA_MINOR)
  endif()
endif()


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
