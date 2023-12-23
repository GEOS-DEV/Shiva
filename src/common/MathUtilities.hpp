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
 * @file MathUtilities.hpp
 */

#pragma once

#include "common/ShivaMacros.hpp"

#include <utility>

namespace shiva
{

/**
 * @namespace shiva::mathUtilities
 * @brief Namespace for math utilities inside of shiva
 */
namespace mathUtilities
{


/**
 * @brief Computes the power of a number at compile time
 * @tparam T The type of the number
 * @tparam EXPONENT The exponent
 * @param base The base of the power
 * @return result^EXPONENT
 */
template< typename T, int EXPONENT >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE
T pow( T const base )
{
  T result = 1;
  for ( int i = 0; i < EXPONENT; ++i )
  {
    result *= base;
  }
  return result;
}

/**
 * @brief helper struct for computing the factorial of a number
 * @tparam T The type of the number
 * @tparam N The number
 * @tparam I type for the integer sequence
 */
template< typename T, T N, typename I = std::make_integer_sequence< T, N > >
struct factorial;

/**
 * @brief Specialization that computes the factorial of a number
 * @tparam T The type of the number
 * @tparam N The number
 * @tparam ISEQ The integer sequence type/values
 */
template< typename T, T N, T ... ISEQ >
struct factorial< T, N, std::integer_sequence< T, ISEQ... > >
{
  /**
   * @brief The factorial of the number
   */
  static constexpr T value = (static_cast< T >(1) * ... *(ISEQ + 1));
};

/**
 * @brief Struct that computes the binomial coefficient N over K
 * @tparam T The type of the number
 * @tparam N The number of objects in the set
 * @tparam K The number of objects to be chose (irrespective of the order) from the set
 */
template< typename T, T N, T K >
struct binomialCoefficient
{
  static_assert( 0 <= K, "K must be greater or equal than 0" );
  static_assert( K <= N, "N must be greater or equal than K" );

  /**
   * @brief The binomial coefficient N over K
   */
  static constexpr T value = factorial< T, N >::value / ( factorial< T, K >::value * factorial< T, N - K >::value );
};

/**
 * @brief Inverse of a 3x3 matrix
 * @tparam REAL_TYPE The data type of the matrix
 * @param matrix The matrix to invert
 * @param det The determinant of the matrix
 */
template< typename REAL_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
inverse( REAL_TYPE (&matrix)[3][3], REAL_TYPE & det )
{

  REAL_TYPE const srcMatrix00 = matrix[ 0 ][ 0 ];
  REAL_TYPE const srcMatrix01 = matrix[ 0 ][ 1 ];
  REAL_TYPE const srcMatrix02 = matrix[ 0 ][ 2 ];
  REAL_TYPE const srcMatrix10 = matrix[ 1 ][ 0 ];
  REAL_TYPE const srcMatrix11 = matrix[ 1 ][ 1 ];
  REAL_TYPE const srcMatrix12 = matrix[ 1 ][ 2 ];
  REAL_TYPE const srcMatrix20 = matrix[ 2 ][ 0 ];
  REAL_TYPE const srcMatrix21 = matrix[ 2 ][ 1 ];
  REAL_TYPE const srcMatrix22 = matrix[ 2 ][ 2 ];

  matrix[ 0 ][ 0 ] = srcMatrix11 * srcMatrix22 - srcMatrix12 * srcMatrix21;
  matrix[ 0 ][ 1 ] = srcMatrix02 * srcMatrix21 - srcMatrix01 * srcMatrix22;
  matrix[ 0 ][ 2 ] = srcMatrix01 * srcMatrix12 - srcMatrix02 * srcMatrix11;

  det = srcMatrix00 * matrix[ 0 ][ 0 ] +
        srcMatrix10 * matrix[ 0 ][ 1 ] +
        srcMatrix20 * matrix[ 0 ][ 2 ];
  REAL_TYPE const invDet = REAL_TYPE( 1 ) / det;

  matrix[ 0 ][ 0 ] *= invDet;
  matrix[ 0 ][ 1 ] *= invDet;
  matrix[ 0 ][ 2 ] *= invDet;
  matrix[ 1 ][ 0 ] = ( srcMatrix12 * srcMatrix20 - srcMatrix10 * srcMatrix22 ) * invDet;
  matrix[ 1 ][ 1 ] = ( srcMatrix00 * srcMatrix22 - srcMatrix02 * srcMatrix20 ) * invDet;
  matrix[ 1 ][ 2 ] = ( srcMatrix02 * srcMatrix10 - srcMatrix00 * srcMatrix12 ) * invDet;
  matrix[ 2 ][ 0 ] = ( srcMatrix10 * srcMatrix21 - srcMatrix11 * srcMatrix20 ) * invDet;
  matrix[ 2 ][ 1 ] = ( srcMatrix01 * srcMatrix20 - srcMatrix00 * srcMatrix21 ) * invDet;
  matrix[ 2 ][ 2 ] = ( srcMatrix00 * srcMatrix11 - srcMatrix01 * srcMatrix10 ) * invDet;
}



/**
 * @brief Inverse of a 3x3 matrix
 * @tparam REAL_TYPE The data type of the matrix
 * @param matrix The matrix to invert
 * @param det The determinant of the matrix
 */
template< typename MATRIX_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
inverse( MATRIX_TYPE & matrix, typename MATRIX_TYPE::value_type & det )
{
  using REAL_TYPE = typename MATRIX_TYPE::value_type;
  REAL_TYPE const srcMatrix00 = matrix( 0, 0 );
  REAL_TYPE const srcMatrix01 = matrix( 0, 1 );
  REAL_TYPE const srcMatrix02 = matrix( 0, 2 );
  REAL_TYPE const srcMatrix10 = matrix( 1, 0 );
  REAL_TYPE const srcMatrix11 = matrix( 1, 1 );
  REAL_TYPE const srcMatrix12 = matrix( 1, 2 );
  REAL_TYPE const srcMatrix20 = matrix( 2, 0 );
  REAL_TYPE const srcMatrix21 = matrix( 2, 1 );
  REAL_TYPE const srcMatrix22 = matrix( 2, 2 );

  matrix( 0, 0 ) = srcMatrix11 * srcMatrix22 - srcMatrix12 * srcMatrix21;
  matrix( 0, 1 ) = srcMatrix02 * srcMatrix21 - srcMatrix01 * srcMatrix22;
  matrix( 0, 2 ) = srcMatrix01 * srcMatrix12 - srcMatrix02 * srcMatrix11;

  det = srcMatrix00 * matrix( 0, 0 ) +
        srcMatrix10 * matrix( 0, 1 ) +
        srcMatrix20 * matrix( 0, 2 );
  REAL_TYPE const invDet = REAL_TYPE( 1 ) / det;

  matrix( 0, 0 ) *= invDet;
  matrix( 0, 1 ) *= invDet;
  matrix( 0, 2 ) *= invDet;
  matrix( 1, 0 ) = ( srcMatrix12 * srcMatrix20 - srcMatrix10 * srcMatrix22 ) * invDet;
  matrix( 1, 1 ) = ( srcMatrix00 * srcMatrix22 - srcMatrix02 * srcMatrix20 ) * invDet;
  matrix( 1, 2 ) = ( srcMatrix02 * srcMatrix10 - srcMatrix00 * srcMatrix12 ) * invDet;
  matrix( 2, 0 ) = ( srcMatrix10 * srcMatrix21 - srcMatrix11 * srcMatrix20 ) * invDet;
  matrix( 2, 1 ) = ( srcMatrix01 * srcMatrix20 - srcMatrix00 * srcMatrix21 ) * invDet;
  matrix( 2, 2 ) = ( srcMatrix00 * srcMatrix11 - srcMatrix01 * srcMatrix10 ) * invDet;
}


} // namespace mathUtilities
} // namespace shiva
