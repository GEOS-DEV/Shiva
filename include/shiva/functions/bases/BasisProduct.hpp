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

#include "shiva/common/NestedSequenceUtilities.hpp"
#include "shiva/common/MultiIndex.hpp"
#include "shiva/common/CArray.hpp"

namespace shiva
{
namespace functions
{

/**
 * @class BasisProduct
 * @brief Defines a class that provides static functions to calculate quantities
 * deriving from the product of basis functions.
 * @tparam REAL_TYPE The floating point type to use
 * @tparam BASIS_TYPES Pack of basis types to apply to each direction of the
 * parent element. There should be a basis defined for each direction.
 */
template< typename REAL_TYPE, typename ... BASIS_TYPES >
struct BasisProduct
{
  /// The number of dimensions/basis in the product
  static inline constexpr int numDims = sizeof...(BASIS_TYPES);

  /// Alias for the floating point type
  using RealType = REAL_TYPE;

  /// Alias for the type that represents a coordinate
  using CoordType = REAL_TYPE[numDims];

  /// Alias for the a integer sequence holding the number of support points in each basis.
  using IndexType = typename SequenceAlias< MultiIndexRangeI,
                                            std::integer_sequence< int, BASIS_TYPES::numSupportPoints... > >::type;


  /// Alias for the a integer sequence holding the number of support points in each basis.
  using numSupportPoints = std::integer_sequence< int, BASIS_TYPES::numSupportPoints... >;


  /**
   * @brief Function to loop over the support points of the basis functions.
   * @tparam FUNC The function type to execute on each support point.
   * @param func the function to execute at each support point
   */
  template< typename FUNC >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
  supportLoop( FUNC && func )
  {
    forNestedSequence< BASIS_TYPES::numSupportPoints... >( std::forward< FUNC >( func ) );
  }



  /**
   * @brief Calculates the value of the basis function at the specified parent
   * coordinate.
   * @tparam BASIS_FUNCTION_INDICES Pack of indices of the basis function to
   * evaluate in each dimension.
   * @tparam COORD_TYPE Coordinate type supporting indexing in each dimension.
   * @param parentCoord The parent coordinate at which to evaluate the basis
   * function.
   * @return The value of the basis function at the specified parent coordinate.
   *
   * The equation for the value of a basis is:
   * \f[
   * \Phi_{i_0 i_1 ... i_{(numDims-1)}}(\boldsymbol{\xi}) =
   * \prod_{k=0}^{(numDims-1)} \phi_{i_k}(\xi_k)\text{, where } \\
   * i_j \text{is index of the basis function in the jth dimension, and
   * ranges from 0 to one less than the number of support points in that dimension.}
   * \f]
   *
   */
  template< int ... BASIS_FUNCTION_INDICES, typename COORD_TYPE = RealType[numDims] >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE RealType
  value( COORD_TYPE const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );

    // Integral-constant arguments preserve distinct coordinate indices in NVCC device code.
    return executeSequence< numDims >( [&] ( auto ... PRODUCT_TERM_INDEX ) constexpr
    {
      return ( BASIS_TYPES::template value< BASIS_FUNCTION_INDICES >( parentCoord[decltype(PRODUCT_TERM_INDEX)::value] ) * ... );
    } );
  }

  /**
   * @brief Calculates the gradient of the basis function at the specified
   * parent coordinate.
   * @tparam BASIS_FUNCTION_INDICES Pack of indices of the basis function to
   * evaluate in each dimension.
   * @tparam COORD_TYPE Coordinate type supporting indexing in each dimension.
   * @param parentCoord The parent coordinate at which to evaluate the basis
   * function gradient.
   * @return The gradient of the basis function at the specified parent
   * coordinate.
   *
   * The equation for the gradient of a basis is:
   * \f[
   * \frac{ d\left( \Phi_{i_0 i_1 ... i_{(numDims-1)}}(\boldsymbol{\xi})
   * \right) }{ d \xi_j} =
   * \nabla \phi_{i_j}(\xi_j) \prod_{ {k=0}\atop {k\neq j} }^{(numDims-1)}
   * \phi_{i_k}(\xi_k)
   * \f]
   */
  template< int ... BASIS_FUNCTION_INDICES, typename COORD_TYPE = RealType[numDims] >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE CArrayNd< RealType, numDims >
  gradient( COORD_TYPE const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );
    return executeSequence< numDims >( [&]< int ... i > () constexpr->CArrayNd< RealType, numDims >
    {
      auto gradientComponent = [&] ( auto const iGrad,
                                     auto const  ... PRODUCT_TERM_INDICES ) constexpr
      {
        // Multiply the directional gradient by the values in every other dimension.
        return ( gradientComponentHelper< BASIS_TYPES,
                                          decltype(iGrad)::value,
                                          BASIS_FUNCTION_INDICES,
                                          PRODUCT_TERM_INDICES >( parentCoord ) * ... );
      };

      return { (executeSequence< numDims >( gradientComponent, std::integral_constant< int, i >{} ) )...  };
    } );
  }


private:
  /**
   * @brief Helper function to return the gradient of a basis function, or the
   * value of the basis function depending on whether the coordinate dimension
   * (@p COORD_INDEX) matches the gradient direction (@p GRADIENT_COMPONENT).
   * This filter is illustrated in the
   * documentation for the gradient function.
   * @tparam BASIS_FUNCTION The basis function type
   * @tparam GRADIENT_COMPONENT The dimension component of the gradient.
   * @tparam BASIS_FUNCTION_INDEX The index of the basis function that is being
   * evaluated.
   * @tparam COORD_INDEX The dimension component of the coordinate.
   * @tparam COORD_TYPE Coordinate type supporting indexing in each dimension.
   * @param parentCoord The parent coordinate at which to evaluate the basis
   * function gradient.
   * @return The one-dimensional basis derivative when the dimensions match,
   * otherwise the one-dimensional basis value.
   */
  template< typename BASIS_FUNCTION, int GRADIENT_COMPONENT, int BASIS_FUNCTION_INDEX, int COORD_INDEX, typename COORD_TYPE >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE RealType
  gradientComponentHelper( COORD_TYPE const & parentCoord )
  {
    if constexpr ( GRADIENT_COMPONENT == COORD_INDEX )
    {
      return BASIS_FUNCTION::template gradient< BASIS_FUNCTION_INDEX >( parentCoord[COORD_INDEX] );
    }
    else
    {
      return ( BASIS_FUNCTION::template value< BASIS_FUNCTION_INDEX >( parentCoord[COORD_INDEX] ) );
    }
  }
};

} // namespace functions
} // namespace shiva
