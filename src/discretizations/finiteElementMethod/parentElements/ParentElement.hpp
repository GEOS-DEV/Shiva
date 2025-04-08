/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2023  Lawrence Livermore National Security LLC
 * Copyright (c) 2023  TotalEnergies
 * Copyright (c) 2023- Shiva Contributors
 * All rights reserved
 *
 * See Shiva/LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

#pragma once

#include "common/NestedSequenceUtilities.hpp"
#include "common/ShivaMacros.hpp"
#include "common/types.hpp"

#include "functions/bases/BasisProduct.hpp"


//#include <utility>

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

  /// Alias for the type that represents a coordinate
  using CoordType = typename ShapeType::CoordType;

  /// The type used to represent the product of basis functions
  using BASIS_PRODUCT_TYPE = functions::BasisProduct< REAL_TYPE, BASIS_TYPE ... >;


  /// The number of dimensions on the ParentElement
  static inline constexpr int numDims = sizeof...(BASIS_TYPE);

  /// The number of degrees of freedom on the ParentElement in each
  /// dimension/basis.
  static inline constexpr int numSupportPoints[numDims] = {BASIS_TYPE::numSupportPoints ...};


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



  /**
   * @brief Evaluate the interpolation of a variable at a parent coordinate.
   * @tparam VAR_TYPE The type of the variable to interpolate.
   * @param parentCoord The parent coordinate at which to interpolate the
   * variable.
   * @param var Object containing the variable data
   * @return The interpolated value of the variable.
   */
  template< typename VAR_TYPE >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE
  value( CoordType const & parentCoord, VAR_TYPE const & var )
  {
    REAL_TYPE rval = {0};

    forNestedSequence< BASIS_TYPE::numSupportPoints... >( [&] ( auto const ... ic_indices ) constexpr
          {
            rval = rval + ( value< decltype(ic_indices)::value ... >( parentCoord ) * var( decltype(ic_indices)::value ... ) );
          } );
    return rval;
  }

  /**
   * @brief Evaluate the gradient of a variable at a parent coordinate.
   * @tparam VAR_TYPE The type of the variable that grad will apply to.
   * @param parentCoord The parent coordinate at which to interpolate the
   * variable.
   * @param var Object containing the variable data
   * @return The gradient of the variable.
   */
  template< typename VAR_TYPE >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE CArrayNd< RealType, numDims >
  gradient( CoordType const & parentCoord, VAR_TYPE const & var )
  {
    CArrayNd< RealType, numDims > rval = {0.0};
    forNestedSequence< BASIS_TYPE::numSupportPoints... >( [&] ( auto const ... ic_indices ) constexpr
          {
            CArrayNd< RealType, numDims > const grad = gradient< decltype(ic_indices)::value ... >( parentCoord );
            forSequence< numDims >( [&] ( auto const a ) constexpr
            {
              rval( a ) = rval( a ) + grad( a ) * var( decltype(ic_indices)::value ... );
            } );
          } );
    return rval;
  }


private:

};


} // namespace finiteElementMethod
} // namespace discretizations
} // namespace shiva
