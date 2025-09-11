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
 * @file ShivaMacros.hpp
 * @brief This file contains macros used throughout the Shiva library.
 */

#pragma once

#include "ShivaConfig.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cinttypes>
#include <cstdarg>

#if defined( SHIVA_USE_HIP )
#include <hip/hip_runtime.h>
#endif

#if defined(SHIVA_USE_CUDA) || defined(SHIVA_USE_HIP)
/// This macro is used to indicate that the code is being compiled for device.
#define SHIVA_USE_DEVICE
#endif

#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
#define SHIVA_DEVICE_CONTEXT
#endif


#if defined(SHIVA_USE_DEVICE)
/// This macro is used to indicate that the code is being compiled for host
/// execution.
#define SHIVA_HOST __host__
/// This macro is used to indicate that the code is being compiled for device
/// execution.
#define SHIVA_DEVICE __device__
/// This macro is used to indicate that the code is being compiled for host
/// or device execution.
#define SHIVA_HOST_DEVICE __host__ __device__
/// This macro us used to indicate that a function or variable should be
/// inlined.
#define SHIVA_FORCE_INLINE __forceinline__
/// This macro is used to indicate that a function is a global function.
#define SHIVA_GLOBAL __global__
#else
/// This macro is used to indicate that the code is being compiled for host
/// execution.
#define SHIVA_HOST
/// This macro is used to indicate that the code is being compiled for device
/// execution.
#define SHIVA_DEVICE
/// This macro is used to indicate that the code is being compiled for host
/// or device execution.
#define SHIVA_HOST_DEVICE
/// This macro us used to indicate that a function or variable should be
/// inlined.
#define SHIVA_FORCE_INLINE inline
/// This macro is used to indicate that a function is a global function.
#define SHIVA_GLOBAL
#endif

/// This macro is used to indicate that a function is a
/// constexpr __host__ __device__ __forceinline__ function.
#define SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE

/// This macro is used to indicate that a function is a
/// static constexpr __host__ __device__ function.
#define SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE static constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE
#if __cplusplus >= 202302L
/// This macro is used to indicate that a function is a
/// consteval __host__ __device__ function.
#define SHIVA_S_CEVAL_HD_I static consteval SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE
#else
/// This macro is used to indicate that a function is a
/// consteval __host__ __device__ function.
/// This is a workaround for compilers that do not support consteval.
#define SHIVA_S_CEVAL_HD_I SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE
#endif

/**
 * @brief This function is used to facilitate ignoring compiler warnings for
 * unused varaibles.
 * @tparam ARGS The types of the arguments to ignore.
 */
template< typename ... ARGS >
constexpr SHIVA_HOST_DEVICE inline
void i_g_n_o_r_e( ARGS const & ... ) {}

/// This macro is used to ignore warnings that that a variable is
/// unused.
#define SHIVA_UNUSED_VAR( ... ) i_g_n_o_r_e( __VA_ARGS__ )



/**
 * @brief This macro is used to implement an assertion.
 * @param cond The condition to assert is true.
 * @param ... The message to print if the assertion fails.
 */
#define SHIVA_ASSERT_MSG( cond, ... ) \
        do { \
          if ( !(cond)) { \
            if ( !__builtin_is_constant_evaluated()) { \
              shivaAssertionFailed( __FILE__, __LINE__, true, __VA_ARGS__ ); \
            } \
          } \
        } while ( 0 )
