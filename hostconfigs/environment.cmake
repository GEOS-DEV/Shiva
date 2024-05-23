site_name(HOST_NAME)

set(CMAKE_C_COMPILER "$ENV{CC}" CACHE PATH "" FORCE)
set(CMAKE_CXX_COMPILER "$ENV{CXX}" CACHE PATH "" FORCE)
set(ENABLE_FORTRAN OFF CACHE BOOL "" FORCE)
set(ENABLE_MPI OFF CACHE PATH "" FORCE)

set(ENABLE_GTEST_DEATH_TESTS ON CACHE BOOL "" FORCE)

set(ENABLE_CUDA "$ENV{ENABLE_CUDA}" CACHE BOOL "" FORCE)
if(ENABLE_CUDA)
  set(CUDA_TOOLKIT_ROOT_DIR "$ENV{CUDA_TOOLKIT_ROOT_DIR}" CACHE PATH "" FORCE)
  if(NOT CUDA_TOOLKIT_ROOT_DIR)
    set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda" CACHE PATH "" FORCE)
  endif()

  set(CMAKE_CUDA_ARCHITECTURES "$ENV{CMAKE_CUDA_ARCHITECTURES}" CACHE STRING "" FORCE)

  set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "")
  set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc CACHE STRING "")
  set(CMAKE_CUDA_FLAGS "-restrict --expt-extended-lambda --expt-relaxed-constexpr -Werror cross-execution-space-call,reorder,deprecated-declarations" CACHE STRING "")

endif()