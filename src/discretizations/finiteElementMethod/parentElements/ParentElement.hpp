#pragma once

#include "common/SequenceUtilities.hpp"
#include "common/ShivaMacros.hpp"
#include "common/types.hpp"


#include <utility>

namespace shiva
{
namespace discretizations
{
namespace finiteElementMethod
{

/**
 * @class ParentElement
 * @brief Defines a class that provides static functions to calculate quantities
 * required from the parent element in a finite element method.
 * @tparam REAL_TYPE The floating point type to use
 * @tparam SHAPE The cell type/geometry
 * @tparam FUNCTIONAL_SPACE_TYPE The functional space type
 * @tparam BASIS_TYPE Pack of basis types to apply to each direction of the
 * parent element. There should be a basis defined for each direction.
 */
template< typename REAL_TYPE, typename SHAPE, typename ... BASIS_TYPE >
class ParentElement
{

public:

  /// The type used to represent the cell/geometry
  using ShapeType = SHAPE;
//  using FunctionalSpaceType = FUNCTIONAL_SPACE_TYPE;
//  using IndexType = typename ShapeType::IndexType;

  /// Alias for the floating point type
  using RealType = REAL_TYPE;

  /// The number of dimensions on the ParentElement
  static inline constexpr int numDims = sizeof...(BASIS_TYPE);

  /// The number of degrees of freedom on the ParentElement in each
  /// dimension/basis.
  static inline constexpr int numDofs[numDims] = {BASIS_TYPE::numDofs()...};


  static_assert( numDims == ShapeType::numDims(), "numDims mismatch between cell and number of basis specified" );

  /**
   * @copydoc functions::BasisProduct::value
   */
  template< int ... BASIS_FUNCTION_INDICES, typename COORD_TYPE = RealType[numDims] >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE RealType
  value( COORD_TYPE const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );
    return ( BASIS_PRODUCT_TYPE::template value< BASIS_FUNCTION_INDICES... >( parentCoord ) );
  }


  /**
   * @copydoc functions::BasisProduct::gradient
   */
  template< int ... BASIS_FUNCTION_INDICES, typename COORD_TYPE = RealType[numDims] >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE CArrayNd< RealType, numDims >
  gradient( COORD_TYPE const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );
    return ( BASIS_PRODUCT_TYPE::template gradient< BASIS_FUNCTION_INDICES... >( parentCoord ) );
  }

  // template< typename VAR_DATA, typename VALUE_TYPE >
  // SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE VALUE_TYPE
  // value( COORD_TYPE const & parentCoord,
  //        VAR_DATA const & var )
  // {
  //   static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );

  //   return ( BASIS_PRODUCT_TYPE::template value< BASIS_FUNCTION_INDICES... >( parentCoord ) );
  // }



};


} // namespace finiteElementMethod
} // namespace discretizations
} // namespace shiva
