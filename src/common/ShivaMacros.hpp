#pragma once

#include "ShivaConfig.hpp"

#if defined(SHIVA_USE_CUDA) || defined(SHIVA_USE_HIP)
#define SHIVA_USE_DEVICE
#endif

#if defined(SHIVA_USE_DEVICE)
#define SHIVA_HOST __host__
#define SHIVA_DEVICE __device__
#define SHIVA_HOST_DEVICE __host__ __device__
#define SHIVA_FORCE_INLINE __forceinline__
#define SHIVA_GLOBAL __global__
#else
#define SHIVA_HOST
#define SHIVA_DEVICE
#define SHIVA_HOST_DEVICE
#define SHIVA_FORCE_INLINE inline
#define SHIVA_GLOBAL
#endif


#define SHIVA_S_CEXPR_HD_I static constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE
#if __cplusplus >= 202302L
#define SHIVA_S_CEVAL_HD_I static consteval SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE
#else
#define SHIVA_S_CEVAL_HD_I SHIVA_S_CEXPR_HD_I
#endif


template< typename ... ARGS >
SHIVA_HOST_DEVICE inline constexpr
void i_g_n_o_r_e( ARGS const & ... ) {}

/// Mark an unused variable and silence compiler warnings.
#define SHIVA_UNUSED_VAR( ... ) i_g_n_o_r_e( __VA_ARGS__ )
