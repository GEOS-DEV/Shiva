#pragma once

#include "common/SequenceUtilities.hpp"
#include "common/ShivaMacros.hpp"
#include "common/types.hpp"

#include "functions/bases/BasisProduct.hpp"


#include <utility>

namespace shiva
{
namespace geometry
{

/**
 * @class InterpolatedShape
 * @brief Defines a class that provides static functions to calculate quantities
 * required from the parent element in a finite element method.
 * @tparam REAL_TYPE The floating point type to use
 * @tparam STANDARD_GEOMETRY The standard geometric form of the interpolated shape (domain)
 * @tparam FUNCTIONAL_SPACE_TYPE The functional space type
 * @tparam BASIS_TYPE Pack of basis types to apply to each direction of the
 * parent element. There should be a basis defined for each direction.
 */
template< typename REAL_TYPE, typename STANDARD_GEOMETRY, typename BASIS_COMBINATION_TYPE >
class InterpolatedShape
{
public:

  /// Alias for the floating point type
  using RealType = REAL_TYPE;

  /// The type used to represent the cell/geometry
  using StandardGeom = STANDARD_GEOMETRY;

  /// The type used to represent the product of basis functions
  using BasisCombinationType = BASIS_COMBINATION_TYPE;

  /**
   * @copydoc functions::BasisProduct::value
   */
  template< int ... BASIS_FUNCTION_INDICES, typename COORD_TYPE = StandardGeom::CoordType >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE
  RealType value( COORD_TYPE const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) = StandardGeom::numDims, "Wrong number of basis function indicies specified" );
    return ( BasisCombinationType::template value< BASIS_FUNCTION_INDICES... >( parentCoord ) );
  }

  /**
   * @copydoc functions::BasisProduct::gradient
   */
  template< int ... BASIS_FUNCTION_INDICES, typename COORD_TYPE = StandardGeom::CoordType >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE
  CArrayNd< RealType, StandardGeom::numDims > gradient( COORD_TYPE const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == StandardGeom::numDims, "Wrong number of basis function indicies specified" );
    return ( BasisCombinationType::template gradient< BASIS_FUNCTION_INDICES... >( parentCoord ) );
  }
};


} // namespace geometry
} // namespace shiva
