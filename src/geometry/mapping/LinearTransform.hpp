
/**
 * @file LinearTransform.hpp
 */

#pragma once

#include "common/MathUtilities.hpp"
#include "common/ShivaMacros.hpp"
#include "common/types.hpp"
#include "common/IndexTypes.hpp"
#include "common/NestedSequenceUtilities.hpp"
#include "common/CArray.hpp"


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
  using JacobianType = CArrayNd< REAL_TYPE, numDims, numDims >;

  /// The type used to represent the data stored at the vertices of the cell
//  using DataType = CArrayNd<REAL_TYPE, InterpolatedShape::BasisCombinationType::numSupportPoints..., numDims >;
//  using DataType = CArrayNd<REAL_TYPE, numVertices, numDims >;

//  using DataType = typename InterpolatedShape::template numSupportPointsPacker< REAL_TYPE >::template type< CArrayNd, numDims >;

  template< int ... NUM_SUPPORT_POINTS >
  using RealCArrayDims = CArrayNd< REAL_TYPE, NUM_SUPPORT_POINTS..., numDims >;
  using DataType = typename SequenceAlias< RealCArrayDims, typename InterpolatedShape::numSupportPointsSequence >::type;

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
                                                             typename LinearTransform< REAL_TYPE, INTERPOLATED_SHAPE >::JacobianType::type & )
{}

/**
 * @brief Calculates the Jacobian transformation of a cuboid from a parent cuboid
 * with range from (-1,1) in each dimension.
 * @tparam REAL_TYPE The floating point type.
 * @param[in] transform The cuboid object
 * @param[in] pointCoordsParent The parent coordinates at which to calculate the
 * Jacobian.
 * @param[out] J The inverse Jacobian transformation.
 */
template< typename REAL_TYPE,
          typename INTERPOLATED_SHAPE,
          typename COORD_TYPE = REAL_TYPE[INTERPOLATED_SHAPE::numDims] >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
jacobian( LinearTransform< REAL_TYPE, INTERPOLATED_SHAPE > const & transform,
          COORD_TYPE const & pointCoordsParent,
          typename LinearTransform< REAL_TYPE, INTERPOLATED_SHAPE >::JacobianType & J )
{
  using Transform = std::remove_reference_t<decltype(transform)>;
  using InterpolatedShape = typename Transform::InterpolatedShape;
  constexpr int DIMS = Transform::numDims;

  auto const & nodeCoords = transform.getData();
  InterpolatedShape::template supportLoop( [&] ( auto const ... icNa ) constexpr 
  {
    CArrayNd< REAL_TYPE, DIMS > const dNadXi = InterpolatedShape::template gradient< decltype(icNa)::value... >( pointCoordsParent );
    // dimensional loop from domain to codomain

#define VARIANT 1
#if VARIANT==0
    forNestedSequence< DIMS, DIMS >( [&] ( auto const ... indices ) constexpr
    {
      constexpr int ijk[DIMS] = { decltype(indices)::value... };
      J( decltype(indices)::value... ) = J( decltype(indices)::value... ) 
                                       + dNadXi(ijk[1]) * nodeCoords( decltype(icNa)::value..., ijk[0] );
    });
#elif VARIANT==1
    forNestedSequence< DIMS, DIMS >( [&] ( auto const ... indices ) constexpr
    {
      constexpr int i = IntPackPeeler< 0, decltype(indices)::value... >::value();
      constexpr int j = IntPackPeeler< 1, decltype(indices)::value... >::value();

      J(i,j) = J(i,j) + dNadXi(j) * nodeCoords( decltype(icNa)::value..., i );
    });
#else
    forNestedSequence< DIMS, DIMS >( [&] ( auto const ici, auto const icj ) constexpr
    {
      constexpr int i = decltype(ici)::value;
      constexpr int j = decltype(icj)::value;
      J(i,j) = J(i,j) + dNadXi(j) * nodeCoords( decltype(icNa)::value..., i );
    });
#endif

#undef VARIANT

  });
}

/**
 * @brief Calculates the inverse Jacobian transformation of a cuboid from a
 * parent cuboid with range from (-1,1) in each dimension.
 * @tparam REAL_TYPE The floating point type.
 * @param[in] transform The cuboid object
 * @param[in] parentCoords The parent coordinates at which to calculate the
 * Jacobian.
 * @param[out] invJ The inverse Jacobian transformation.
 * @param[out] detJ The determinant of the Jacobian transformation.
 */
template< typename REAL_TYPE,
          typename INTERPOLATED_SHAPE,
          typename COORD_TYPE = REAL_TYPE[INTERPOLATED_SHAPE::numDims] >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
inverseJacobian( LinearTransform< REAL_TYPE, INTERPOLATED_SHAPE > const & transform,
                 COORD_TYPE const & parentCoords,
                 typename LinearTransform< REAL_TYPE, INTERPOLATED_SHAPE >::JacobianType & invJ,
                 REAL_TYPE & detJ )
{
  jacobian( transform, parentCoords, invJ );
  mathUtilities::inverse( invJ, detJ );
}


} //namespace utilities
} // namespace geometry
} // namespace shiva
