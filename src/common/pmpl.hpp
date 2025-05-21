/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2023  Lawrence Livermore National Security LLC
 * Copyright (c) 2023  TotalEnergies
 * Copyright (c) 2023- Shiva Contributors
 * All rights reserved
 *
 * See Shiva/LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */


/**
 * @file pmpl.hpp
 * @brief This file contains the implementation of the
 * "Poor Man's Portablity Layer" (pmpl) used for constructing unit tests. This
 * should NOT be used in any code that is not a unit test.
 */

#pragma once


#include "ShivaMacros.hpp"

#include <utility>
namespace shiva
{
#if defined(SHIVA_USE_DEVICE)
  #if defined(SHIVA_USE_CUDA)
    #define deviceMalloc( PTR, BYTES ) cudaMalloc( PTR, BYTES );
    #define deviceMallocManaged( PTR, BYTES ) cudaMallocManaged( PTR, BYTES );
    #define deviceDeviceSynchronize() cudaDeviceSynchronize();
    #define deviceMemCpy( DST, SRC, BYTES, KIND ) cudaMemcpy( DST, SRC, BYTES, KIND );
    #define deviceFree( PTR ) cudaFree( PTR );
    #define deviceError_t cudaError_t
    #define deviceSuccess cudaSuccess
    #define deviceGetErrorString    cudaGetErrorString
    #elif defined(SHIVA_USE_HIP)
    #define deviceMalloc( PTR, BYTES ) hipMalloc( PTR, BYTES );
    #define deviceMallocManaged( PTR, BYTES ) hipMallocManaged( PTR, BYTES );
    #define deviceDeviceSynchronize() hipDeviceSynchronize();
    #define deviceMemCpy( DST, SRC, BYTES, KIND ) hipMemcpy( DST, SRC, BYTES, KIND );
    #define deviceFree( PTR ) hipFree( PTR );
    #define deviceError_t hipError_t
    #define deviceSuccess = hipSuccess;
    #define deviceGetErrorString    hipGetErrorString
    #endif
#endif

/**
 * @namespace shiva::pmpl
 * @brief The pmpl namespace contains all of the pmpl classes and functions
 * used to provide a portablity layer in unit testing.
 */
namespace pmpl
{

/**
 * @brief This function checks if two floating point numbers are equal within a
 * tolerance.
 * @tparam REAL_TYPE This is the type of the floating point numbers to compare.
 * @param a This is the first floating point number to compare.
 * @param b This is the second floating point number to compare.
 * @param tolerance This is the tolerance to use when comparing the two numbers.
 * @return This returns true if the two numbers are equal within the tolerance.
 */
template< typename REAL_TYPE >
static constexpr bool check( REAL_TYPE const a, REAL_TYPE const b, REAL_TYPE const tolerance )
{
  return ( a - b ) * ( a - b ) < tolerance * tolerance;
}


/**
 * @brief This function provides a generic kernel execution mechanism that can
 * be called on either host or device.
 * @tparam LAMBDA The type of the lambda function to execute.
 * @param func The lambda function to execute.
 */
template< typename LAMBDA >
SHIVA_GLOBAL void genericKernel( LAMBDA func )
{
  func();
}

/**
 * @brief This function provides a wrapper to the genericKernel function.
 * @tparam LAMBDA The type of the lambda function to execute.
 * @param func The lambda function to execute.
 * @param abortOnError If true, the program will abort if the kernel fails.
 *
 * This function will execute the lambda through a kernel launch of
 * genericKernel.
 */
template< typename LAMBDA >
void genericKernelWrapper( LAMBDA && func, bool const abortOnError = true )
{
#if defined(SHIVA_USE_DEVICE)
  // UNCRUSTIFY-OFF
  genericKernel <<< 1, 1 >>> ( std::forward< LAMBDA >( func ) );
  // UNCRUSTIFY-ON
  deviceError_t err = deviceDeviceSynchronize();
  if ( err != cudaSuccess )
  {
    printf( "Kernel failed: %s\n", deviceGetErrorString( err ));
    if ( abortOnError )
    {
      // LCOV_EXCL_START
      printf( "Aborting...\n" );
      std::abort();
      // LCOV_EXCL_STOP
    }
  }
#else
  genericKernel( std::forward< LAMBDA >( func ) );
  SHIVA_UNUSED_VAR( abortOnError );
#endif
}



/**
 * @brief This function provides a generic kernel execution mechanism that can
 * be called on either host or device.
 * @tparam DATA_TYPE The type of the data pointer.
 * @tparam LAMBDA The type of the lambda function to execute.
 * @param func The lambda function to execute.
 * @param data A general data pointer to pass to the lambda function that should
 * hold all data required to execute the lambda function, aside from what is
 * captured.
 */
template< typename DATA_TYPE, typename LAMBDA >
SHIVA_GLOBAL void genericKernel( LAMBDA func, DATA_TYPE * const data )
{
  func( data );
}

/**
 * @brief This function provides a wrapper to the genericKernel function.
 * @tparam DATA_TYPE The type of the data pointer.
 * @tparam LAMBDA The type of the lambda function to execute.
 * @param N The size of the data array.
 * @param hostData The data pointer to pass to the lambda function.
 * @param func The lambda function to execute.
 * @param abortOnError If true, the program will abort if the kernel fails.
 *
 * This function will allocate the data pointer on the device, execute the
 * lambda through a kernel launch of genericKernel, and then synchronize the
 * device.
 */
template< typename DATA_TYPE, typename LAMBDA >
void genericKernelWrapper( int const N, DATA_TYPE * const hostData, LAMBDA && func, bool const abortOnError = true )
{

#if defined(SHIVA_USE_DEVICE)
  DATA_TYPE * deviceData;
  deviceMalloc( &deviceData, N * sizeof(DATA_TYPE) );
  deviceMemCpy( deviceData, hostData, N * sizeof(DATA_TYPE), cudaMemcpyHostToDevice );
  // UNCRUSTIFY-OFF
  genericKernel <<< 1, 1 >>> ( std::forward< LAMBDA >( func ), deviceData );
  // UNCRUSTIFY-ON
  deviceError_t err = deviceDeviceSynchronize();
  deviceMemCpy( hostData, deviceData, N * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost );
  deviceFree( deviceData );
  if ( err != cudaSuccess )
  {
    printf( "Kernel failed: %s\n", deviceGetErrorString( err ));
    if ( abortOnError )
    {
      // LCOV_EXCL_START
      printf( "Aborting...\n" );
      std::abort();
      // LCOV_EXCL_STOP
    }
  }
#else
  SHIVA_UNUSED_VAR( N, abortOnError );
  genericKernel( std::forward< LAMBDA >( func ), hostData );

#endif
}

/**
 * @brief convenience function for allocating data allocated on a pointer
 * @tparam DATA_TYPE The type of the data pointer.
 * @param data The data pointer to deallocate.
 */
template< typename DATA_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE void deallocateData( DATA_TYPE * data )
{
#if defined(SHIVA_USE_DEVICE)
  deviceFree( data );
#else
  delete[] data;
#endif
}

} // namespace pmpl
} // namespace shiva
