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
 * @file types.hpp
 * @brief Wrappers and definitions for the types used in shiva.
 */
#pragma once

#include "common/ShivaMacros.hpp"

/// @brief Macro to define whether or not to use camp.
#if defined(SHIVA_USE_CAMP)
#include <camp/camp.hpp>
#else

#if defined(SHIVA_USE_CUDA)
#include <cuda/std/tuple>
#else
#include <tuple>
#endif

#endif

namespace shiva
{

#if defined(SHIVA_USE_CAMP)

/**
 * @brief Wrapper for camp::tuple.
 * @tparam T Types of the elements of the tuple.
 */
template< typename ... T >
using tuple = camp::tuple< T ... >;

/**
 * @brief Wrapper for camp::make_tuple.
 * @tparam T Types of the elements of the tuple.
 * @param t Elements of the tuple.
 * @return A tuple with the elements passed as arguments.
 */
template< typename ... T >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto make_tuple( T && ... t )
{
  return camp::make_tuple( std::forward< T >( t ) ... );
}

#else
#if defined(SHIVA_USE_CUDA)
/**
 * @brief Wrapper for cuda::std::tuple.
 * @tparam T Types of the elements of the tuple.
 */
template< typename ... T >
using tuple = cuda::std::tuple< T ... >;

/**
 * @brief Wrapper for cuda::std::make_tuple.
 * @tparam T Types of the elements of the tuple.
 * @param t Elements of the tuple.
 * @return A tuple with the elements passed as arguments.
 */
template< typename ... T >
auto make_tuple( T && ... t )
{
  return cuda::std::make_tuple( std::forward< T >( t ) ... );
}
#else
/**
 * @brief Wrapper for std::tuple.
 * @tparam T Types of the elements of the tuple.
 */
template< typename ... T >
using tuple = std::tuple< T ... >;

/**
 * @brief Wrapper for std::make_tuple.
 * @tparam T Types of the elements of the tuple.
 * @param t Elements of the tuple.
 * @return A tuple with the elements passed as arguments.
 */
template< typename ... T >
auto make_tuple( T && ... t )
{
  return std::make_tuple( std::forward< T >( t ) ... );
}
#endif
#endif

/**
 * @brief alias for std::integer_sequence<int, T...>.
 * @tparam T Types of the elements of the sequence.
 */
template< int ... T >
using int_sequence = std::integer_sequence< int, T ... >;

/**
 * @brief alias for std::make_integer_sequence<int, N>.
 * @tparam N Size of the sequence.
 */
template< int N >
using make_int_sequence = std::make_integer_sequence< int, N >;

}
