#pragma once


#if defined(SHIVA_USE_DEVICE)
#define SHIVA_HOST __host__
#define SHIVA_DEVICE __device__
#define SHIVA_HOST_DEVICE __host__ __device__
#define SHIVA_FORCE_INLINE __forceinline__
#else
/// Marks a host-only function.
#define SHIVA_HOST
/// Marks a device-only function.
#define SHIVA_DEVICE
/// Marks a host-device function.
#define SHIVA_HOST_DEVICE
/// Marks a function or lambda for inlining
#define SHIVA_FORCE_INLINE inline
/// Compiler directive specifying to unroll the loop.
#endif

template< typename ... ARGS >
SHIVA_HOST_DEVICE inline constexpr
void i_g_n_o_r_e( ARGS const & ... ) {}

/// Mark an unused variable and silence compiler warnings.
#define SHIVA_UNUSED_VAR( ... ) i_g_n_o_r_e( __VA_ARGS__ )
