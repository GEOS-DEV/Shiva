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

template < typename T, T N, typename I = std::make_integer_sequence<T, N>>
struct factorial;

template < typename T, T N, T ... ISEQ>
struct factorial< T, N, std::integer_sequence< T, ISEQ... > > {
   static constexpr T value = (static_cast<T>(1)* ... *(ISEQ + 1));
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

} // namespace mathUtilities
} // namespace shiva
