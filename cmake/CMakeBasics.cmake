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
  if(CUDAToolkit_FOUND AND CUDAToolkit_VERSION)
    set(SHIVA_CUDA_VERSION ${CUDAToolkit_VERSION})
    string(REPLACE "." ";" _ver_list ${CUDAToolkit_VERSION})
    list(GET _ver_list 0 SHIVA_CUDA_MAJOR)
    list(GET _ver_list 1 SHIVA_CUDA_MINOR)
    list(GET _ver_list 2 SHIVA_CUDA_PATCHLEVEL)
    math(EXPR SHIVA_CUDA_VERSION_INT "${SHIVA_CUDA_MAJOR}*1000 + ${SHIVA_CUDA_MINOR}*10 + ${SHIVA_CUDA_PATCHLEVEL}")
  else()
    message(FATAL_ERROR "Could not determine CUDA version. Please set CUDAToolkit_ROOT to the location of your CUDA installation.")
  endif()
else()
  set(SHIVA_CUDA_VERSION "0.0.0")
  set(SHIVA_CUDA_MAJOR 0)
  set(SHIVA_CUDA_MINOR 0)
  set(SHIVA_CUDA_PATCHLEVEL 0)
  set(SHIVA_CUDA_VERSION_INT 0)
endif()
