#pragma once

#include "common/MathUtilities.hpp"
#include "common/ShivaMacros.hpp"
#include "types/types.hpp"
#include "types/IndexTypes.hpp"


#define USE_MULTI_INDEX

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
  using IndexType = MultiIndexRange<int, 2,2,2>;

  constexpr static bool jacobianIsConstInCell() { return false; }




  template< typename INDEX_TYPE >
  REAL_TYPE const & getVertexCoord( INDEX_TYPE const & a, int const i ) const 
  { return m_vertexCoords[ linearIndex(a) ][i]; }


  template< typename INDEX_TYPE >
  CoordType const & getVertexCoord( INDEX_TYPE const & a ) const 
  { return m_vertexCoords[ linearIndex(a) ]; }

  
  template< typename INDEX_TYPE >
  void setVertexCoord( INDEX_TYPE const & a, int const i, REAL_TYPE const & value ) 
  { m_vertexCoords[ linearIndex(a) ][i] = value; }
  



  template< typename INDEX_TYPE >
  void setVertexCoord( INDEX_TYPE const & a, CoordType const & value ) 
  { 
    m_vertexCoords[ linearIndex(a) ][0] = value[0]; 
    m_vertexCoords[ linearIndex(a) ][1] = value[1]; 
    m_vertexCoords[ linearIndex(a) ][2] = value[1]; 
}






  template< typename FUNCTION_TYPE >
  void forVertices( FUNCTION_TYPE && func ) const
  {
    IndexType index{ { 1,0,0 } };

    forRange( index={0,0,0}, [this,func]( auto const & index )
    {
      func( index, this->getVertexCoord(index) ); 
    } );
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
               REAL_TYPE const (&pointCoordsParent)[3],
               typename Cuboid<REAL_TYPE>::JacobianType::type & J )
{
  constexpr int vertexCoordsParent[2] = { -1, 1 }; // this is provided by the Basis

  cell.forVertices( [&J, pointCoordsParent ]( auto const & index, REAL_TYPE const (&vertexCoord)[3] )
  {

    // dNdXi is provided by the Basis, which will take in the generic "index" type. 
    // it will probably look like:
    // CArray1d<REAL_TYPE, 3> const dNdXi = basis.dNdXi( index, pointCoordsParent );

    int const a = index.data[0];
    int const b = index.data[1];
    int const c = index.data[2];
    REAL_TYPE const dNdXi[3] = { 0.125 *                              vertexCoordsParent[a] * ( 1 + vertexCoordsParent[b]*pointCoordsParent[1] ) * ( 1 + vertexCoordsParent[c]*pointCoordsParent[2] ),
                                 0.125 * ( 1 + vertexCoordsParent[a]*pointCoordsParent[0] ) *                              vertexCoordsParent[b] * ( 1 + vertexCoordsParent[c]*pointCoordsParent[2] ),
                                 0.125 * ( 1 + vertexCoordsParent[a]*pointCoordsParent[0] ) * ( 1 + vertexCoordsParent[b]*pointCoordsParent[1] ) *                              vertexCoordsParent[c] };

    for( int i = 0; i < 3; ++i )
    {
      for( int j = 0; j < 3; ++j )
      {
        J[j][i] = J[j][i] + dNdXi[i] * vertexCoord[j];
      }
    }
  } );
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

#undef USE_MULTI_INDEX
