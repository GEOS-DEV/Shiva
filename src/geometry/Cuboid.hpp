#pragma once

#include "types/types.hpp"

namespace shiva
{

namespace geometry
{

template< typename REAL_TYPE >
class Cuboid
{
public:
  using JacobianType = CArray2d<REAL_TYPE,3,3>;
  using DataType = REAL_TYPE[2][2][2][3];

  DataType       & getData()       { return m_VertexCoords; }
  DataType const & getData() const { return m_VertexCoords; }

private:
  constexpr static bool jacobianIsConstInCell = false;
  DataType m_VertexCoords;
};

namespace utilities
{

template< typename REAL_TYPE >
void jacobian( Cuboid<REAL_TYPE> const & ,//cell, 
               typename Cuboid<REAL_TYPE>::JacobianType::type & )//J )
{
  // constexpr static int dpsi[2] = { -1, 1 };

  // auto const & X = cell.getData();
  // for( int a=0; a<2; ++a )
  // {
  //   for( int b=0; b<2; ++b )
  //   {
  //     for( int c=0; c<2; ++c )
  //     {
  //       REAL_TYPE const dNdXi[3] = { dpsi[a] * 0.125,
  //                                    dpsi[b] * 0.125,
  //                                    dpsi[c] * 0.125 };
  //       for( int i = 0; i < 3; ++i )
  //       {
  //         for( int j = 0; j < 3; ++j )
  //         {
  //           J[j][i] = J[j][i] + dNdXi[i] * X[a][b][c][j];
  //         }
  //       }
  //     }
  //   }
  // }
}

template< typename REAL_TYPE >
void jacobian( Cuboid<REAL_TYPE> const & cell, 
               REAL_TYPE const (&parentCoords)[3],
               typename Cuboid<REAL_TYPE>::JacobianType::type & J )
{
  constexpr static int sign[2] = { -1, 1 };

  auto const & X = cell.getData();
  for( int a=0; a<2; ++a )
  {
    for( int b=0; b<2; ++b )
    {
      for( int c=0; c<2; ++c )
      {
        REAL_TYPE const dNdXi[3] = { 0.125 * sign[a] * ( 1 + sign[b]*parentCoords[1] ) * ( 1 + sign[c]*parentCoords[2] ),
                                     0.125 * sign[b] * ( 1 + sign[a]*parentCoords[0] ) * ( 1 + sign[c]*parentCoords[2] ),
                                     0.125 * sign[c] * ( 1 + sign[a]*parentCoords[0] ) * ( 1 + sign[b]*parentCoords[1] ) };
        for( int i = 0; i < 3; ++i )
        {
          for( int j = 0; j < 3; ++j )
          {
            J[j][i] = J[j][i] + dNdXi[i] * X[a][b][c][j];
          }
        }
      }
    }
  }
}

} //namespace utilities
} // namespace geometry
} // namespace shiva