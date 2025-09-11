set(CONFIG_NAME "maple_rocky9" CACHE PATH "") 

set(COMPILER_DIR  /opt/rh/gcc-toolset-13/root/ )
set(CMAKE_C_COMPILER ${COMPILER_DIR}/bin/gcc CACHE PATH "")
set(CMAKE_CXX_COMPILER ${COMPILER_DIR}/bin/g++ CACHE PATH "")

# C++ options
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -mtune=native -march=native" CACHE STRING "")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g ${CMAKE_CXX_FLAGS_RELEASE}" CACHE STRING "")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g" CACHE STRING "")

# Cuda options
set(ENABLE_CUDA ON CACHE BOOL "")
set(CUDA_TOOLKIT_ROOT_DIR /hrtc/apps/cuda/12.6.20/aarch64/rocky9 CACHE STRING "")
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE STRING "")
set(CMAKE_CUDA_COMPILER ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvcc CACHE STRING "")
set(CMAKE_CUDA_ARCHITECTURES 90 CACHE STRING "")
set(CMAKE_CUDA_STANDARD 17 CACHE STRING "")
set(CMAKE_CUDA_FLAGS "-restrict --expt-extended-lambda --expt-relaxed-constexpr -Werror cross-execution-space-call,reorder,deprecated-declarations" CACHE STRING "")
#set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG -Xcompiler -DNDEBUG -Xcompiler -O3 -Xcompiler -mcpu=powerpc64le -Xcompiler -mtune=powerpc64le" CACHE STRING "")
#set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-g -lineinfo ${CMAKE_CUDA_FLAGS_RELEASE}" CACHE STRING "")
#set(CMAKE_CUDA_FLAGS_DEBUG "-g -G -O0 -Xcompiler -O0" CACHE STRING "")

set( SHIVA_ENABLE_CAMP OFF CACHE BOOL "Disable CAMP support" FORCE )