#pragma once

#include "common/ShivaMacros.hpp"
#include "types/types.hpp"
#include "common/MathUtilities.hpp"

namespace shiva
{

namespace geometry
{

template< typename REAL_TYPE >
class Cuboid
{
public:
  using JacobianType = CArray2d<REAL_TYPE,3,3>;
  using DataType = REAL_TYPE[8][3];
  using CoordType = REAL_TYPE[3];

  constexpr static bool jacobianIsConstInCell() { return false; }

  REAL_TYPE const & getVertexCoord( int const a, int const b, int const c, int const i ) const 
  { return m_vertexCoords[ 4*a+2*b+c ][i]; }

  REAL_TYPE const & getVertexCoord( int const a, int const i ) const 
  { return m_vertexCoords[a][i]; }



  CoordType const & getVertexCoord( int const a, int const b, int const c ) const 
  { return m_vertexCoords[ 4*a+2*b+c ]; }

  CoordType const & getVertexCoord( int const a ) const 
  { return m_vertexCoords[a]; }




  void setVertexCoord( int const a, int const b, int const c, int const i, REAL_TYPE const & value ) 
  { m_vertexCoords[ 4*a+2*b+c ][i] = value; }

  void setVertexCoord( int const a, int const i, REAL_TYPE const & value ) 
  { m_vertexCoords[a][i] = value; }



  void setVertexCoord( int const a, CoordType const & value ) 
  { 
    m_vertexCoords[ a ][0] = value[0]; 
    m_vertexCoords[ a ][1] = value[1]; 
    m_vertexCoords[ a ][2] = value[1]; 
  }

  void setVertexCoord( int const a, int const b, int const c, CoordType const & value ) 
  { 
    setVertexCoord( 4*a+2*b+c, value ); 
  }


private:
  DataType m_vertexCoords;
};

namespace utilities
{

template< typename REAL_TYPE >
void jacobian( Cuboid<REAL_TYPE> const & ,//cell, 
               typename Cuboid<REAL_TYPE>::JacobianType::type & )//J )
{}

template< typename REAL_TYPE >
void jacobian( Cuboid<REAL_TYPE> const & cell, 
               REAL_TYPE const (&parentCoords)[3],
               typename Cuboid<REAL_TYPE>::JacobianType::type & J )
{
  constexpr static int sign[2] = { -1, 1 };

  for( int a=0; a<2; ++a )
  {
    for( int b=0; b<2; ++b )
    {
      for( int c=0; c<2; ++c )
      {
        REAL_TYPE const dNdXi[3] = { 0.125 *                         sign[a] * ( 1 + sign[b]*parentCoords[1] ) * ( 1 + sign[c]*parentCoords[2] ),
                                     0.125 * ( 1 + sign[a]*parentCoords[0] ) *                         sign[b] * ( 1 + sign[c]*parentCoords[2] ),
                                     0.125 * ( 1 + sign[a]*parentCoords[0] ) * ( 1 + sign[b]*parentCoords[1] ) *                         sign[c] };
        for( int i = 0; i < 3; ++i )
        {
          for( int j = 0; j < 3; ++j )
          {
            J[j][i] = J[j][i] + dNdXi[i] * cell.getVertexCoord( a, b, c, j );
          }
        }
      }
    }
  }
}

template< typename REAL_TYPE >
void inverseJacobian( Cuboid< REAL_TYPE > const & cell, 
                           REAL_TYPE const (&parentCoords)[3],
                           typename Cuboid< REAL_TYPE >::JacobianType::type & invJ,
                           REAL_TYPE & detJ )
{
  jacobian( cell, parentCoords, invJ );
  mathUtilities::inverse( invJ, detJ );
}


} //namespace utilities
} // namespace geometry
} // namespace shiva