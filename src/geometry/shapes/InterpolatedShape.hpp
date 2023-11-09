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
 * @tparam BASE_SHAPE The cell type/geometry
 * @tparam FUNCTIONAL_SPACE_TYPE The functional space type
 * @tparam BASIS_TYPE Pack of basis types to apply to each direction of the
 * parent element. There should be a basis defined for each direction.
 */
template< typename REAL_TYPE, typename BASE_SHAPE, typename ... BASIS_TYPE >
class InterpolatedShape
{

public:

  /// The type used to represent the cell/geometry
  using BaseShape = BASE_SHAPE;
//  using FunctionalSpaceType = FUNCTIONAL_SPACE_TYPE;
//  using IndexType = typename BaseShape::IndexType;

  /// Alias for the floating point type
  using RealType = REAL_TYPE;

  /// Alias for the type that represents a coordinate
  using CoordType = typename BaseShape::CoordType;

  /// The type used to represent the product of basis functions
  using BASIS_PRODUCT_TYPE = functions::BasisProduct< REAL_TYPE, BASIS_TYPE... >;

  /// The number of dimensions on the InterpolatedShape
  static inline constexpr int numDims = sizeof...(BASIS_TYPE);

  /// The number of vertices on the InterpolatedShape
  static inline constexpr int numVertices = BaseShape::numVertices();

  static inline constexpr int numVerticesInBasis[numDims] = {BASIS_TYPE::numSupportPoints...};

  static_assert( numDims == BaseShape::numDims(), "numDims mismatch between cell and number of basis specified" );


  /**
   * @copydoc functions::BasisProduct::value
   */
  template< int ... BASIS_FUNCTION_INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE RealType
  value( CoordType const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );
    return ( BASIS_PRODUCT_TYPE::template value< BASIS_FUNCTION_INDICES... >( parentCoord ) );
  }

  /**
   * @copydoc functions::BasisProduct::gradient
   */
  template< int ... BASIS_FUNCTION_INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE CArray1d< RealType, numDims >
  gradient( CoordType const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );
    return ( BASIS_PRODUCT_TYPE::template gradient< BASIS_FUNCTION_INDICES... >( parentCoord ) );
  }
};


} // namespace geometry
} // namespace shiva
