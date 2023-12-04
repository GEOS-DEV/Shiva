
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
 * @tparam REAL_TYPE The type of real numbers used for floating point data.
 * @tparam INTERPOLATED_SHAPE Interpolation machinery + ideal shape being transformed from one space into another.
 *
 * The term cuboid is used here to define a 3-dimensional volume with 6
 * quadralateral sides.
 * <a href="https://en.wikipedia.org/wiki/LinearTransform"> LinearTransform
 *(Wikipedia)</a>
 */
template< typename REAL_TYPE,
          typename INTERPOLATED_SHAPE,
          typename CODOMAIN_COORD_INDEX_SPACE = INTERPOLATED_SHAPE::StandardGeom::CoordIndexSpace >
class MultiLinearTransform
{
private:
template < int ... INDEX_EXTENTS >
using rtCArrayND = CArrayND< REAL_TYPE, INDEX_EXTENTS ... >;
public:
  /// Alias for the floating point type for the transform.
  using RealType = REAL_TYPE;

  using InterpolatedShape = INTERPOLATED_SHAPE;

  using DomainCoordIndexSpace = InterpolatedShape::StandardGeom::CoordIndexSpace;
  using CodomainCoordIndexSpace = CODOMAIN_COORD_INDEX_SPACE;

  /// The type used to represent the Jacobian transformation operation
  using JacobianIndexSpace = typename Concat_t< CodomainCoordIndexSpace, DomainCoordIndexSpace >;
  using JacobianType = typename SequenceAlias< rtCArrayND, JacobianIndexSpace >;

  /// The type used to represent the data stored at the vertices of the cell
  using CoordDataIndexSpace = Concat_t< InterpolatedShape::BasisCombinationType::SupportPointsSequence, DomainCoordIndexSpace >;
  using CoordDataType = typename SequenceAlias< CArrayND, DataIndexSpace >::type;

  /// The type used to represent the index space of the cell
  using SupportIndexType = typename InterpolatedShape::BasisCombinationType::IndexType;

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
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE CoordDataType const & getData() const { return m_coordData; }

  /**
   * @brief Provides non-const access to member data through reference.
   * @return a mutable reference to the member data.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE CoordDataType & setData() { return m_coordData; }

  /**
   * @brief Sets the coordinates of the vertices of the cell
   * @param[in] coords The coordinates of the vertices of the cell
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE void setData( CoordDataType const & coords )
  {
    forNestedSequence< IndexSpace >( [&] ( auto const ... ai )
    {
      m_coordData( decltype(ai)::value ... ) = coords( decltype(ai)::value ... );
    } );
  }

private:
  /// Data member that stores the vertex coordinates of the ideal shape
  CoordDataType m_coordData;
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
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void jacobian( MultiLinearTransform< REAL_TYPE, INTERPOLATED_SHAPE > const &,//cell,
                                                             typename MultiLinearTransform< REAL_TYPE, INTERPOLATED_SHAPE >::JacobianType::type & )
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
          typename COORD_TYPE = typename INTERPOLATED_SHAPE::StandardGeom::CoordType >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
jacobian( MultiLinearTransform< REAL_TYPE, INTERPOLATED_SHAPE > const & transform,
          COORD_TYPE const & pointCoordsParent,
          typename MultiLinearTransform< REAL_TYPE, INTERPOLATED_SHAPE >::JacobianType & J )
{
  using Transform = std::remove_reference_t<decltype(transform)>;
  using InterpolatedShape = typename Transform::InterpolatedShape;
  constexpr int DIMS = Transform::StandardGeom::numDims;

  auto const & supportCoords = transform.getData();
  InterpolatedShape::BasisCombinationType::template supportLoop( [&] ( auto const ... ic_spIndices ) constexpr
  {
    CArrayNd< REAL_TYPE, DIMS > const dNadXi = InterpolatedShape::template gradient< decltype(ic_spIndices)::value... >( pointCoordsParent );
    // dimensional loop from domain to codomain

#define VARIANT 1
#if VARIANT==0
    forNestedSequence( JacobianIndexSpace, [&] ( auto const ... dimIndices ) constexpr
    {
      constexpr int ijk[DIMS] = { decltype(dimIndices)::value... };
      J( decltype(dimIndices)::value... ) = J( decltype(dimIndices)::value... )
                                       + dNadXi(ijk[1]) * supportCoords( decltype(ic_spIndices)::value..., ijk[0] );
    });
#elif VARIANT==1
    forNestedSequence( JacobianIndexSpace, [&] ( auto const ... dimIndices ) constexpr
    {
      using ij = SplitPack< decltype(dimIndices)::value_type, CodomainCoordIndexSpace::size(), DomainCoordIndexSpace::size(), decltype(dimIndices)::value... >;
      using i = typename ij::first;
      using j = typename ij::second;
      unpack< i >( [&] ( auto... i )
      {
        unpack< j > ( [&] ( auto... j )
        {
          J( decltype(dimIndices)::value... ) = J( decltype(dimIndices)::value... ) + dNadXi( j... ) * supportCoords( decltype(ic_spIndices)::value..., i... );
        } );
      } );
    } );
#else
    forNestedSequence( JacobianIndexSpace, [&] ( auto const ici, auto const icj ) constexpr
    {
      constexpr int i = decltype(ici)::value;
      constexpr int j = decltype(icj)::value;
      J(i,j) = J(i,j) + dNadXi(j) * supportCoords( decltype(ic_spIndices)::value..., i );
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
inverseJacobian( MultiLinearTransform< REAL_TYPE, INTERPOLATED_SHAPE > const & transform,
                 COORD_TYPE const & parentCoords,
                 typename MultiLinearTransform< REAL_TYPE, INTERPOLATED_SHAPE >::JacobianType & invJ,
                 REAL_TYPE & detJ )
{
  jacobian( transform, parentCoords, invJ );
  mathUtilities::inverse( invJ, detJ );
}


} //namespace utilities
} // namespace geometry
} // namespace shiva
