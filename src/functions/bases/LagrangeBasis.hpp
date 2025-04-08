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

/**
 * @file LagrangeBasis.hpp
 * @brief Defines the LagrangeBasis class
 */

#pragma once

#include "common/SequenceUtilities.hpp"
#include "common/ShivaMacros.hpp"


#include <utility>

namespace shiva
{
namespace functions
{


/**
 * @class LagrangeBasis
 * @brief Defines a class to calculate quantities defined by a Lagrange
 * polynomial function of a given order.
 * @tparam REAL_TYPE The floating point type to use
 * @tparam ORDER The order of the basis function
 * @tparam SPACING_TYPE The spacing type to define the interpolation points for
 * the Lagrange polynomials.
 * @tparam USE_FOR_SEQUENCE If true, the staticFor will be used to calculate
 * values and gradient functions. If false, the executeSequence function will be
 * used.
 *
 * The equation for a Lagrange interpolating basis function is:
 *
 * \f[
 * P_j(x) = \prod_{ { {k=0} \atop {k\neq j} } }^{n-1} \frac{x-x_k}{x_j-x_k}
 * \f]
 *
 */
template< typename REAL_TYPE,
          int ORDER,
          template< typename, int > typename SPACING_TYPE >
class LagrangeBasis : public SPACING_TYPE< REAL_TYPE, ORDER + 1 >
{
public:

  /// Alias for the spacing type used to define the interpolation points
  using SpacingType = SPACING_TYPE< REAL_TYPE, ORDER + 1 >;

  /// The order of the Lagrange polynomial basis functions.
  static inline constexpr int order = ORDER;

  /// The number of support points for the Lagrange polynomial basis functions.
  static inline constexpr int numSupportPoints = ORDER + 1;

  /**
   * @brief Calculates the value of the Lagrange polynomial basis function
   * @p BF_INDEX at the input coordinate @p coord.
   * @tparam BF_INDEX The index of the basis function to calculate the value of.
   * @param coord The coordinate at which the value is calculated.
   * @return The value of the Lagrange polynomial basis function.
   *
   * The equation is the "Lagrange basis":
   *
   * \f[ P_j(x) = \prod_{ { {k=0} \atop {k\neq j} } }^{n-1}
   *\frac{x-x_k}{x_j-x_k} \f], where x = @p coord
   */
  template< int BF_INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE
  value( REAL_TYPE const & coord )
  {
#if __cplusplus >= 202002L
    return executeSequence< numSupportPoints >( [&] < int ... a > () constexpr
        {
          // return fold expression that is the product of all the polynomial
          // factor terms.
          return ( valueProductTerm< BF_INDEX, a >( coord ) * ... );
        } );
#else
    return executeSequence< numSupportPoints >( [&] ( auto const ... a ) constexpr
        {
          return ( valueProductTerm< BF_INDEX, decltype(a)::value >( coord ) * ... );
        } );
#endif
  }

  /**
   * @brief Calculates the gradient of the Lagrange polynomial basis function
   * @p BF_INDEX at the input coordinate @p coord.
   * @tparam BF_INDEX The index of the basis function to calculate the gradient
   * of.
   * @param coord The coordinate at which to calculate the gradient.
   * @return The gradient of the Lagrange polynomial basis function.
   *
   * The gradient is expressed as:
   *
   * \f[ \nabla_x P_j(x) = \sum_{ {i=0} \atop {i\neq j} }^{n-1} \left(
   *\frac{1}{x_j-x_i} \prod_{ { {k=0} \atop {k\neq i,j} } }^{n-1}
   *\frac{x-x_k}{x_j-x_k} \right) \f], where x = @p coord
   *
   */
  template< int BF_INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE
  gradient( REAL_TYPE const & coord )
  {

#if __cplusplus >= 202002L
    return executeSequence< numSupportPoints >( [&coord] < int ... a > () constexpr
        {
          auto func = [&coord] < int ... b > ( auto aa ) constexpr
          {
            constexpr int aVal = decltype(aa)::value;
            return gradientOfValueTerm< BF_INDEX, aVal >() * ( valueProductFactor< BF_INDEX, b, aVal >( coord ) * ... );
          };

          return ( executeSequence< numSupportPoints >( func, std::integral_constant< int, a >{} ) + ... );
        } );
#else
    return executeSequence< numSupportPoints >( [&coord] ( auto const ... a ) constexpr
        {
          REAL_TYPE const values[ numSupportPoints ] = { valueProductTerm< BF_INDEX, decltype(a)::value >( coord )... };
          auto func = [&values] ( auto aa, auto ... b ) constexpr
          {
            constexpr int aVal = decltype(aa)::value;
            return gradientOfValueTerm< BF_INDEX, aVal >() * ( valueProductFactor< decltype(b)::value, aVal >( values ) * ... );
          };

          return ( executeSequence< numSupportPoints >( func, a ) + ... );
        } );
#endif
  }


private:
  /**
   * @brief Calculates the value of the Lagrange polynomial basis function
   * product factor at the given @p BF_INDEX at the input coordinate
   *@p coord.
   * @tparam BF_INDEX The index of the basis function to calculate the value of.
   * @tparam TERM_INDEX The index of the term in the sequence product.
   * @param coord The coordinate to calculate the value at.
   * @return The value of the Lagrange polynomial basis function.
   *
   * In the equation for the "Lagrange basis", the "value term" is:
   * \f[ \frac{ x - x_k }{ x_j - x_k } \f]. The
   *
   */
  template< int BF_INDEX, int TERM_INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE
  valueProductTerm( REAL_TYPE const & coord )
  {
    if constexpr ( BF_INDEX == TERM_INDEX )
    {
      return 1.0;
    }
    else
    {
      constexpr REAL_TYPE coordinate_FI = SpacingType::template coordinate< TERM_INDEX >();
      constexpr REAL_TYPE coordinate_BF = SpacingType::template coordinate< BF_INDEX >();
      constexpr REAL_TYPE denom = 1.0 / ( coordinate_BF - coordinate_FI );
      return ( coord - coordinate_FI ) * denom;
    }
  }

  /**
   * @brief Applies an index filter to an array that contains the values of the
   * Lagrange polynomial basis function product terms.
   * @tparam TERM_INDEX The index of the factor to calculate the value of.
   * @tparam DERIVATIVE_INDEX The index of the derivative term that is calling
   * this function.
   * @param values The array that contains the Lagrange polynomial basis terms.
   * @return The value of the Lagrange polynomial basis function.
   *
   * In the equation for the "Lagrange basis":
   * \f[ P_j(x) = \prod_{ { {k=0} \atop {k\neq j} } }^{n-1}
   *\frac{x-x_k}{x_j-x_k} \f], the filter addresses the case where \f$ k = j
   *\f$.
   *
   */
  template< int TERM_INDEX, int DERIVATIVE_INDEX = -1 >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE
  valueProductFactor( REAL_TYPE const (&values)[numSupportPoints] )
  {
    if constexpr ( TERM_INDEX == DERIVATIVE_INDEX )
    {
      return 1.0;
    }
    else
    {
      return values[TERM_INDEX];
    }
  }

  /**
   * @brief Calculates the gradient of a term in the Lagrange polynomial basis
   * function for a @p BF_INDEX.
   * @tparam BF_INDEX The index of the basis function.
   * @tparam TERM_INDEX The index of the term in the sequence product.
   * @return The gradient of the Lagrange polynomial basis function.
   *
   */
  template< int BF_INDEX, int TERM_INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE
  gradientOfValueTerm()
  {
    if constexpr ( BF_INDEX == TERM_INDEX )
    {
      return 0.0;
    }
    else
    {
      constexpr REAL_TYPE coordinate_FI = SpacingType::template coordinate< TERM_INDEX >();
      constexpr REAL_TYPE coordinate_BF = SpacingType::template coordinate< BF_INDEX >();
      return 1.0 / ( coordinate_BF - coordinate_FI );
    }
  }

}; // class LagrangeBasis

} // namespace functions
} // namespace shiva
