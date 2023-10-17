#pragma once

#include "ShivaMacros.hpp"

namespace shiva
{
#if defined(SHIVA_USE_DEVICE)
  #if defined(SHIVA_USE_CUDA)
    #define deviceMallocManaged( PTR, BYTES ) cudaMallocManaged( PTR, BYTES );
    #define deviceDeviceSynchronize() cudaDeviceSynchronize();
    #define deviceFree( PTR ) cudaFree(PTR);
  #elif defined(SHIVA_USE_HIP)
    #define deviceMallocManaged( PTR, BYTES ) hipMallocManaged( PTR, BYTES );
    #define deviceDeviceSynchronize() hipDeviceSynchronize();
    #define deviceFree( PTR ) hipFree(PTR);
  #endif
#endif

namespace pmpl
{
static constexpr bool check( double const a, double const b, double const tolerance )
{
  return ( a - b ) * ( a - b ) < tolerance * tolerance;
}

template< typename DATA_TYPE, typename LAMBDA >
SHIVA_GLOBAL void genericKernel( LAMBDA func, DATA_TYPE * const data )
{
  func( data );
}

template< typename DATA_TYPE, typename LAMBDA >
void genericKernelWrapper( int const N, DATA_TYPE * & data, LAMBDA && func )
{

#if defined(SHIVA_USE_DEVICE)
  deviceMallocManaged( &data, N * sizeof(DATA_TYPE) );
  genericKernel<<<1,1>>>( std::forward<LAMBDA>(func), data );
  deviceDeviceSynchronize();
#else
  data = new DATA_TYPE[N];
  genericKernel( std::forward<LAMBDA>(func), data );
#endif
}

template< typename DATA_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE void deallocateData( DATA_TYPE * data )
{
#if defined(SHIVA_USE_DEVICE)
  deviceFree(data);
#else
  delete[] data;
#endif
}

} // namespace pmpl
} // namespace shiva