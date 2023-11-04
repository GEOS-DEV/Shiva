
/**
 * @file Cuboid.hpp
 */

#pragma once

#include "common/MathUtilities.hpp"
#include "common/ShivaMacros.hpp"
#include "common/types.hpp"
#include "common/IndexTypes.hpp"


/**
 * namespace to encapsulate all shiva code
 */
namespace shiva
{

/**
 * namespace to encapsulate all geometry code
 */
namespace geometry
{

/**
 * @brief Class to represent a cuboid
 * @tparam REAL_TYPE The type of real numbers used for floating point data.
 *
 * The term cuboid is used here to define a 3-dimensional volume with 6
 * quadralateral sides.
 * <a href="https://en.wikipedia.org/wiki/Cuboid"> Cuboid (Wikipedia)</a>
 */
template< typename REAL_TYPE >
class Cuboid
{
public:

  /// The type used to represent the Jacobian transformation operation
  using JacobianType = CArray2d< REAL_TYPE, 3, 3 >;

  /// The type used to represent the data stored at the vertices of the cell
  using DataType = REAL_TYPE[8][3];

  /// The type used to represent the coordinates of the vertices of the cell
  using CoordType = REAL_TYPE[3];

  /// The type used to represent the index space of the cell
  using IndexType = MultiIndexRange< int, 2, 2, 2 >;

  /**
   * @brief Returns a boolean indicating whether the Jacobian is constant in
   * the cell. This is used to determine whether the Jacobian should be
   * computed once per cell or once per quadrature point.
   * @return true if the Jacobian is constant in the cell, false otherwise
   */
  constexpr static bool jacobianIsConstInCell() { return false; }



  /**
   * @brief const accessor for a component of a vertex coordinate
   * @tparam INDEX_TYPE The type of the index
   * @param[in] a The value of the vertex index
   * @param[in] i The component of the vertex coordinate
   * @return REAL_TYPE const& A const reference to the vertex coordinate component
   */
  template< typename INDEX_TYPE >
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE REAL_TYPE const & getVertexCoord( INDEX_TYPE const & a, int const i ) const
  { return m_vertexCoords[ linearIndex( a ) ][i]; }

  /**
   * @brief const accessor for the vertex coordinate
   * @tparam INDEX_TYPE The type of the index
   * @param[in] a The value of the vertex index
   * @return CoordType const& A const reference to the vertex coordinate object
   */
  template< typename INDEX_TYPE >
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE CoordType const & getVertexCoord( INDEX_TYPE const & a ) const
  { return m_vertexCoords[ linearIndex( a ) ]; }


  /**
   * @brief setter accessor for a component of a vertex coordinate
   * @tparam INDEX_TYPE The type of the index
   * @param[in] a The value of the vertex index
   * @param[in] i The component direction of the vertex coordinate
   * @param[in] value A reference to the vertex coordinate component
   */
  template< typename INDEX_TYPE >
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE void
  setVertexCoord( INDEX_TYPE const & a, int const i, REAL_TYPE const & value )
  { m_vertexCoords[ linearIndex( a ) ][i] = value; }


  /**
   * @brief setter accessor for the vertex coordinate
   * @tparam INDEX_TYPE The type of the index
   * @param[in] a The value of the vertex index
   * @param[in] value The input coordinate
   */
  template< typename INDEX_TYPE >
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE void
  setVertexCoord( INDEX_TYPE const & a, CoordType const & value )
  {
    m_vertexCoords[ linearIndex( a ) ][0] = value[0];
    m_vertexCoords[ linearIndex( a ) ][1] = value[1];
    m_vertexCoords[ linearIndex( a ) ][2] = value[1];
  }

  /**
   * @brief method to loop over the vertices of the cuboid
   * @tparam FUNCTION_TYPE The type of the function to execute
   * @param[in] func The function to execute
   */
  template< typename FUNCTION_TYPE >
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE void forVertices( FUNCTION_TYPE && func ) const
  {
    IndexType index{0, 0, 0};

    forRange( index, [this, func] ( auto const & i )
    {
      func( i, this->getVertexCoord( i ) );
    } );
  }

private:
  /// Data member that stores the vertex coordinates of the cuboid
  DataType m_vertexCoords;
};

namespace utilities
{

/**
 * @brief NoOp that would calculate the Jacobian transormation of a cuboid
 * from a parent cuboid with range from (-1,1) in each dimension. However the Jacobian is not constant in the cell, so we keep this as a
 *no-op to allow
 * for it to be called in the same way as the other geometry objects with
 * constant Jacobian.
 * @tparam REAL_TYPE The floating point type.
 */
template< typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void jacobian( Cuboid< REAL_TYPE > const &,//cell,
                                                             typename Cuboid< REAL_TYPE >::JacobianType::type & )//J )
{}

/**
 * @brief Calculates the Jacobian transormation of a cuboid from a parent
 * cuboid with range from (-1,1) in each dimension.
 * @tparam REAL_TYPE The floating point type.
 * @param[in] cell The cuboid object
 * @param[in] pointCoordsParent The parent coordinates at which to calculate the Jacobian.
 * @param[out] J The inverse Jacobian transformation.
 */
template< typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
jacobian( Cuboid< REAL_TYPE > const & cell,
          REAL_TYPE const (&pointCoordsParent)[3],
          typename Cuboid< REAL_TYPE >::JacobianType::type & J )
{

  cell.forVertices( [&J, pointCoordsParent ] ( auto const & index, REAL_TYPE const (&vertexCoord)[3] )
  {

    constexpr int vertexCoordsParent[2] = { -1, 1 }; // this is provided by the Basis
    // dNdXi is provided by the Basis, which will take in the generic "index" type.
    // it will probably look like:
    // CArray1d<REAL_TYPE, 3> const dNdXi = basis.dNdXi( index, pointCoordsParent );

    int const a = index.data[0];
    int const b = index.data[1];
    int const c = index.data[2];
    REAL_TYPE const dNdXi[3] = { 0.125 *       vertexCoordsParent[a]                          * ( 1 + vertexCoordsParent[b] * pointCoordsParent[1] ) *
                                 ( 1 + vertexCoordsParent[c] * pointCoordsParent[2] ),
                                 0.125 * ( 1 + vertexCoordsParent[a] * pointCoordsParent[0] ) *       vertexCoordsParent[b] *
                                 ( 1 + vertexCoordsParent[c] * pointCoordsParent[2] ),
                                 0.125 * ( 1 + vertexCoordsParent[a] * pointCoordsParent[0] ) * ( 1 + vertexCoordsParent[b] * pointCoordsParent[1] ) *       vertexCoordsParent[c] };

    for ( int i = 0; i < 3; ++i )
    {
      for ( int j = 0; j < 3; ++j )
      {
        J[j][i] = J[j][i] + dNdXi[i] * vertexCoord[j];
      }
    }
  } );
}

/**
 * @brief Calculates the inverse Jacobian transormation of a cuboid from a
 * parent cuboid with range from (-1,1) in each dimension.
 * @tparam REAL_TYPE The floating point type.
 * @param[in] cell The cuboid object
 * @param[in] parentCoords The parent coordinates at which to calculate the Jacobian.
 * @param[out] invJ The inverse Jacobian transformation.
 * @param[out] detJ The determinant of the Jacobian transformation.
 */
template< typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
inverseJacobian( Cuboid< REAL_TYPE > const & cell,
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
