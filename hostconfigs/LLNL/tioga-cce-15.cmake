
set(CONFIG_NAME "tioga-cce@15.0.1" CACHE PATH "")
include( ${CMAKE_CURRENT_LIST_DIR}/tioga-base.cmake )

# C++ options
set(CRAYPE_VERSION "2.7.23")
set(CMAKE_C_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/cc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/CC" CACHE PATH "")
set(CMAKE_Fortran_COMPILER "/opt/cray/pe/craype/${CRAYPE_VERSION}/bin/ftn" CACHE PATH "")

set( ENABLE_CLANG_HIP ON CACHE BOOL "" FORCE )

set( HIP_VERSION_STRING "5.4.3" CACHE STRING "" )
set( HIP_ROOT "/opt/rocm-${HIP_VERSION_STRING}" CACHE PATH "" )
set( ROCM_PATH ${HIP_ROOT} CACHE PATH "" )

set( CMAKE_HIP_ARCHITECTURES "gfx90a" CACHE STRING "" FORCE )
set( CMAKE_CXX_FLAGS "-fgpu-rdc" CACHE STRING "" FORCE )
set( CMAKE_CXX_LINK_FLAGS "-fgpu-rdc --hip-link" CACHE STRING "" FORCE )
