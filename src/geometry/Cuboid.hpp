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

  constexpr static bool jacobianIsConstInCell() { return false; }




  template< typename INDEX_TYPE >
  REAL_TYPE const & getVertexCoord( INDEX_TYPE const & a, int const i ) const 
  { return m_vertexCoords[ linearIndex(a) ][i]; }

  REAL_TYPE const & getVertexCoord( int const a, int const b, int const c, int const i ) const 
  { return m_vertexCoords[ 4*a+2*b+c ][i]; }

  REAL_TYPE const & getVertexCoord( int const a, int const i ) const 
  { return m_vertexCoords[a][i]; }



  template< typename INDEX_TYPE >
  CoordType const & getVertexCoord( INDEX_TYPE const & a ) const 
  { return m_vertexCoords[ linearIndex(a) ]; }

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

  template< typename FUNCTION_TYPE >
  void forVertices( FUNCTION_TYPE && func ) const
  {
  #if defined(USE_MULTI_INDEX)
    MultiIndexRange<int, 2,2,2 > index{ { 1,0,0 } };

    forRange( index={0,0,0}, [this,func]( auto const & index )
    {
      func( index, this->getVertexCoord(index) ); 
    } );

  #else
    for( int a=0; a<2; ++a )
    {
      for( int b=0; b<2; ++b )
      {
        for( int c=0; c<2; ++c )
        {
          func( a, b, c, getVertexCoord(a,b,c) ); 
        }
      }
    }
  #endif
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

#if defined(USE_MULTI_INDEX)
  cell.forVertices( [&J, pointCoordsParent ]( auto const & index, REAL_TYPE const (&vertexCoord)[3] )
  {

    // dNdXi is provided by the Basis, which will take in the generic "index" type

    int const a = index.data[0];
    int const b = index.data[1];
    int const c = index.data[2];
    REAL_TYPE const dNdXi[3] = { 0.125 *                              vertexCoordsParent[a] * ( 1 + vertexCoordsParent[b]*pointCoordsParent[1] ) * ( 1 + vertexCoordsParent[c]*pointCoordsParent[2] ),
                                 0.125 * ( 1 + vertexCoordsParent[a]*pointCoordsParent[0] ) *                              vertexCoordsParent[b] * ( 1 + vertexCoordsParent[c]*pointCoordsParent[2] ),
                                 0.125 * ( 1 + vertexCoordsParent[a]*pointCoordsParent[0] ) * ( 1 + vertexCoordsParent[b]*pointCoordsParent[1] ) *                              vertexCoordsParent[c] };

#else
  cell.forVertices( [&J, pointCoordsParent ]( int const a, int const b, int const c, REAL_TYPE const (&vertexCoord)[3] )
  {
    // dNdXi is provided by the Basis
    REAL_TYPE const dNdXi[3] = { 0.125 *                              vertexCoordsParent[a] * ( 1 + vertexCoordsParent[b]*pointCoordsParent[1] ) * ( 1 + vertexCoordsParent[c]*pointCoordsParent[2] ),
                                 0.125 * ( 1 + vertexCoordsParent[a]*pointCoordsParent[0] ) *                              vertexCoordsParent[b] * ( 1 + vertexCoordsParent[c]*pointCoordsParent[2] ),
                                 0.125 * ( 1 + vertexCoordsParent[a]*pointCoordsParent[0] ) * ( 1 + vertexCoordsParent[b]*pointCoordsParent[1] ) *                              vertexCoordsParent[c] };
#endif
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
