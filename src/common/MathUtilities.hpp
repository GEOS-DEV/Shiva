

#pragma once

namespace shiva
{
namespace mathUtilities
{
template< typename REAL_TYPE >
inline
static void inverse( REAL_TYPE (&matrix)[3][3], REAL_TYPE & det )
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
