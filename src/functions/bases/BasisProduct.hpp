#pragma once

#include "common/SequenceUtilities.hpp"
#include "common/MultiIndex.hpp"

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

  static inline constexpr std::integer_sequence< int, BASIS_TYPES::numSupportPoints... > basisSupportCounts{};

  using IndexType = typename SequenceAlias< MultiIndexRangeI, decltype(basisSupportCounts) >::type;

  /**
   * @brief Calculates the value of the basis function at the specified parent
   * coordinate.
   * @tparam BASIS_FUNCTION_INDICES Pack of indices of the basis function to
   * evaluate in each dimension.
   * @param parentCoord The parent coordinate at which to evaluate the basis
   * function.
   * @return The value of the basis function at the specified parent coordinate.
   *
   * The equation for the value of a basis is:
   * \f[
   * \Phi_{i_0 i_1 ... i_{(numDims-1)}}(\boldsymbol{\xi}) =
   * \prod_{k=0}^{(numDims-1)} \phi_{i_k}(\xi_k)\text{, where } \\
   * i_j \text{is index of the basis function in the jth dimension, and
   * ranges from [0...(order+1)]}
   * \f]
   *
   */
  template< int ... BASIS_FUNCTION_INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE RealType
  value( CoordType const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );

    return
#if __cplusplus >= 202002L
      // expand pack over number of dimensions
      executeSequence< numDims >( [&]< int ... PRODUCT_TERM_INDEX > () constexpr
    {
      return ( BASIS_TYPE::template value< BASIS_FUNCTION_INDICES >( parentCoord[PRODUCT_TERM_INDEX] ) * ... );
    } );
#else
      executeSequence< numDims >( [&] ( auto ... PRODUCT_TERM_INDEX ) constexpr
    {
      // fold expression to multiply the value of each BASIS_TYPE in each
      // dimension. In other words the fold expands on BASIS_TYPE...,
      // BASIS_FUNCTION_INDICES..., and PRODUCT_TERM_INDEX... together.
      return ( BASIS_TYPES::template value< BASIS_FUNCTION_INDICES >( parentCoord[decltype(PRODUCT_TERM_INDEX)::value] ) * ... );
    } );

#endif
  }

  /**
   * @brief Calculates the gradient of the basis function at the specified
   * parent coordinate.
   * @tparam BASIS_FUNCTION_INDICES Pack of indices of the basis function to
   * evaluate in each dimension.
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
  template< int ... BASIS_FUNCTION_INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE CArray1d< RealType, numDims >
  gradient( CoordType const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );

#if __cplusplus >= 202002L
    return executeSequence< numDims >( [&]< int ... i >() constexpr -> CArray1d< RealType, numDims >
    {
      auto gradientComponent = [&] ( auto const iGrad,
                                     auto const  ... PRODUCT_TERM_INDICES ) constexpr
      {
        // Ca
        return ( gradientComponentHelper< BASIS_TYPES,
                                          decltype(iGrad)::value,
                                          BASIS_FUNCTION_INDICES,
                                          PRODUCT_TERM_INDICES >( parentCoord ) * ... );
      };

      return { (executeSequence< numDims >( gradientComponent, std::integral_constant< int, i >{} ) )...  };
    } );
#else
    // Expand over the dimensions.
    return executeSequence< numDims >( [&] ( auto ... a ) constexpr -> CArray1d< RealType, numDims >
    {
      // define a lambda that calculates the gradient of the basis function in
      // a single dimension/direction.
      auto gradientComponent = [&] ( auto GRADIENT_COMPONENT, auto ... PRODUCT_TERM_INDICES ) constexpr
      {
        // fold expression calling gradientComponentHelper using expanding on
        // BASIS_TYPE, BASIS_FUNCTION_INDICES, and PRODUCT_TERM_INDICES.
        return ( gradientComponentHelper< BASIS_TYPES,
                                          decltype(GRADIENT_COMPONENT)::value,
                                          BASIS_FUNCTION_INDICES,
                                          decltype(PRODUCT_TERM_INDICES)::value >( parentCoord ) * ... );
      };

      // execute the gradientComponent lambda on each direction, expand the
      // pack on "i" corresponding to each direction of the gradient.
      return { (executeSequence< numDims >( gradientComponent, a ) )...  };
    } );
#endif
  }


private:
  /**
   * @brief Helper function to return the gradient of a basis function, or the
   * value of the basis function depending on whether or not the index of the
   * basis function (@p BASIS_FUNCTION) matches the index of the direction
   * component index (@p GRADIENT_COMPONENT). This filter is illustrated in the
   * documentation for the gradient function.
   * @tparam BASIS_FUNCTION The basis function type
   * @tparam GRADIENT_COMPONENT The dimension component of the gradient.
   * @tparam BASIS_FUNCTION_INDEX The index of the basis function that is being
   * evaluated.
   * @tparam COORD_INDEX The dimension component of the coordinate.
   * @param parentCoord The parent coordinate at which to evaluate the basis
   * function gradient.
   * @return The gradient component of the basis function at the specified
   * parent coordinate.
   */
  template< typename BASIS_FUNCTION, int GRADIENT_COMPONENT, int BASIS_FUNCTION_INDEX, int COORD_INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE RealType
  gradientComponentHelper( CoordType const & parentCoord )
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