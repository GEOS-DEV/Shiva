set(CMAKE_ENABLE_EXPORTS ON)

if( CMAKE_BUILD_TYPE MATCHES "Debug" )

else()

endif()

set( SHIVA_BUILD_OBJ_LIBS OFF CACHE BOOL "" )


option( SHIVA_ENABLE_BOUNDS_CHECK "Enable bounds checking in shiva::CArray" ON )

if( CMAKE_CXX_STANDARD IN_LIST "98; 11; 14" )
    MESSAGE(FATAL_ERROR "Shiva requires at least c++17")
endif()


blt_append_custom_compiler_flag( FLAGS_VAR CMAKE_CXX_FLAGS DEFAULT "${OpenMP_CXX_FLAGS}")
blt_append_custom_compiler_flag( FLAGS_VAR CMAKE_CXX_FLAGS
                                 GNU   "-Wpedantic -pedantic-errors -Wshadow -Wfloat-equal -Wcast-align -Wcast-qual"
                                 CLANG "-Wpedantic -pedantic-errors -Wshadow -Wfloat-equal -Wcast-align -Wcast-qual"
                               )

blt_append_custom_compiler_flag( FLAGS_VAR CMAKE_CXX_FLAGS_DEBUG
                                 GNU ""
                                 CLANG "-fstandalone-debug"
                                )

option( SHIVA_ENABLE_CAMP OFF )
option( CAMP_ENABLE_TESTS OFF )


if( ENABLE_CUDA )
  # Extract CUDA version from CMake’s variables
  set(SHIVA_CUDA_VERSION ${CUDAToolkit_VERSION})

  # Also normalize to an integer for easy comparison (e.g. 12040 for 12.4.0)
  string(REPLACE "." ";" CUDA_VERSION_LIST ${CUDAToolkit_VERSION})
  list(GET CUDA_VERSION_LIST 0 CUDA_MAJOR)
  list(GET CUDA_VERSION_LIST 1 CUDA_MINOR)
  list(GET CUDA_VERSION_LIST 2 CUDA_PATCH)

  math(EXPR CUDA_VERSION_INT "${CUDA_MAJOR}*1000 + ${CUDA_MINOR}*10 + ${CUDA_PATCH}")

  target_compile_definitions( shiva PUBLIC
                              SHIVA_CUDA_VERSION_STR="${CUDAToolkit_VERSION}"
                              SHIVA_CUDA_VERSION_INT=${CUDA_VERSION_INT}
                              SHIVA_CUDA_MAJOR=${CUDA_MAJOR}
                              SHIVA_CUDA_MINOR=${CUDA_MINOR}
                            )
endif()
