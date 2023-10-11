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

static constexpr bool check( double const a, double const b, double const tolerance )
{
  return ( a - b ) * ( a - b ) < tolerance * tolerance;
}

}