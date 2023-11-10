
/**
 * @file LinearTransform.hpp
 */

#pragma once

#include "common/MathUtilities.hpp"
#include "common/ShivaMacros.hpp"
#include "common/types.hpp"
#include "common/IndexTypes.hpp"
#include "common/NestedSequenceUtilities.hpp"


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
 * <a href="https://en.wikipedia.org/wiki/LinearTransform"> LinearTransform
 *(Wikipedia)</a>
 */
template< typename REAL_TYPE,
          typename INTERPOLATED_SHAPE >
class LinearTransform
{
public:
  using InterpolatedShape = INTERPOLATED_SHAPE;

  /// number of vertices in the geometric object that will be transformed.
  static inline constexpr int numVertices = InterpolatedShape::numVertices;

  /// number of dimensions of the geometric object that will be transformed.
  static inline constexpr int numDims =  InterpolatedShape::numDims;

  /// Alias for the floating point type for the transform.
  using RealType = REAL_TYPE;

  /// The type used to represent the Jacobian transformation operation
  using JacobianType = CArray2d< REAL_TYPE, numDims, numDims >;

  /// The type used to represent the data stored at the vertices of the cell
  using DataType = REAL_TYPE[numVertices][numDims];

  /// The type used to represent the coordinates of the vertices of the cell
  using CoordType = REAL_TYPE[numDims];

  /// The type used to represent the index space of the cell
  using SupportIndexType = typename InterpolatedShape::BasisCombinationType::IndexType;
  // using IndexType = typename SequenceAlias< MultiIndexRangeI, decltype(InterpolatedShape::basisSupportCounts) >::type;

  /**
   * @brief Returns a boolean indicating whether the Jacobian is constant in the
   * cell. This is used to determine whether the Jacobian should be computed once
   * per cell or once per quadrature point.
   * @return true if the Jacobian is constant in the cell, false otherwise
   */
  constexpr static bool jacobianIsConstInCell() { return false; }


  /**
   * @brief Provides access to member data through reference to const.
   * @return a const reference to the member data.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE DataType const & getData() const { return m_vertexCoords; }

  /**
   * @brief Provides non-const access to member data through reference.
   * @return a mutable reference to the member data.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE DataType & getData() { return m_vertexCoords; }


  /**
   * @brief Sets the coordinates of the vertices of the cell
   * @param[in] coords The coordinates of the vertices of the cell
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE void setData( DataType const & coords )
  {
    for ( int a = 0; a < numVertices; ++a )
    {
      for ( int i = 0; i < numDims; ++i )
      {
        m_vertexCoords[a][i] = coords[a][i];
      }
    }
  }


  // /**
  //  * @brief method to loop over the vertices of the cuboid
  //  * @tparam FUNCTION_TYPE The type of the function to execute
  //  * @param[in] func The function to execute
  //  */
  // template< typename FUNCTION_TYPE >
  // constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE void forVertices( FUNCTION_TYPE && func ) const
  // {
  //   IndexType index{0, 0, 0};

  //   forRange( index, [this, func] ( auto const & i )
  //   {
  //     func( i, m_vertexCoords[linearIndex( i )] );
  //   } );
  // }

private:
  /// Data member that stores the vertex coordinates of the cuboid
  DataType m_vertexCoords;
};

namespace utilities
{

/**
 * @brief NoOp that would calculate the Jacobian transformation of a cuboid from
 * a parent cuboid with range from (-1,1) in each dimension. However the Jacobian
 * is not constant in the cell, so we keep this as a no-op to allow for it to be
 * called in the same way as the other geometry objects with constant Jacobian.
 * @tparam REAL_TYPE The floating point type.
 */
template< typename REAL_TYPE, typename INTERPOLATED_SHAPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void jacobian( LinearTransform< REAL_TYPE, INTERPOLATED_SHAPE > const &,//cell,
                                                             typename LinearTransform< REAL_TYPE, INTERPOLATED_SHAPE >::JacobianType::type & )//J
// )
{}

/**
 * @brief Calculates the Jacobian transformation of a cuboid from a parent cuboid
 * with range from (-1,1) in each dimension.
 * @tparam REAL_TYPE The floating point type.
 * @param[in] cell The cuboid object
 * @param[in] pointCoordsParent The parent coordinates at which to calculate the
 * Jacobian.
 * @param[out] J The inverse Jacobian transformation.
 */
template< typename REAL_TYPE, typename INTERPOLATED_SHAPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
jacobian( LinearTransform< REAL_TYPE, INTERPOLATED_SHAPE > const & cell,
          REAL_TYPE const (&pointCoordsParent)[3],
          typename LinearTransform< REAL_TYPE, INTERPOLATED_SHAPE >::JacobianType::type & J )
{
  using Transform = std::remove_reference_t<decltype(cell)>;
  using InterpolatedShape = typename Transform::InterpolatedShape;
  using IndexType = typename InterpolatedShape::BasisCombinationType::IndexType;
  constexpr int DIMS = Transform::numDims;

  auto const & nodeCoords = cell.getData();
  InterpolatedShape::template supportLoop(
    [&] ( auto const ... icNa ) constexpr 
    {
      IndexType index{ { decltype(icNa)::value... } };
      CArray1d< REAL_TYPE, DIMS > const dNadXi = InterpolatedShape::template gradient< decltype(icNa)::value... >( pointCoordsParent );
      auto const & nodeCoord = nodeCoords[ flattenIndex( index ) ];
      // dimensional loop from domain to codomain
      forNestedSequence< DIMS, DIMS >(
      [&] ( auto const ... icijk ) constexpr
      {
        constexpr int ijk[DIMS] = { decltype(icijk)::value... };
        J[ijk[1]][ijk[0]] = J[ijk[1]][ijk[0]] + dNadXi[ijk[0]] * nodeCoord[ijk[1]];
      } );
    } 
  );
}

/**
 * @brief Calculates the inverse Jacobian transormation of a cuboid from a
 * parent cuboid with range from (-1,1) in each dimension.
 * @tparam REAL_TYPE The floating point type.
 * @param[in] cell The cuboid object
 * @param[in] parentCoords The parent coordinates at which to calculate the
 * Jacobian.
 * @param[out] invJ The inverse Jacobian transformation.
 * @param[out] detJ The determinant of the Jacobian transformation.
 */
template< typename REAL_TYPE, typename INTERPOLATED_SHAPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
inverseJacobian( LinearTransform< REAL_TYPE, INTERPOLATED_SHAPE > const & cell,
                 REAL_TYPE const (&parentCoords)[3],
                 typename LinearTransform< REAL_TYPE, INTERPOLATED_SHAPE >::JacobianType::type & invJ,
                 REAL_TYPE & detJ )
{
  jacobian( cell, parentCoords, invJ );
  mathUtilities::inverse( invJ, detJ );
}


} //namespace utilities
} // namespace geometry
} // namespace shiva
