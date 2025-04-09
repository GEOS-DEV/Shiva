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
 * @file Quadrature.hpp
 * @brief Contains the definition of the Quadrature classes
 */

#pragma once

#include "../spacing/Spacing.hpp"

#include <cassert>
#include <limits>


namespace shiva
{

/**
 * @brief This struct provides a static constexpr functions to compute the
 * quadrature weights on a line that correspond with the coordinates of the
 * GaussLegendreSpacing.
 * @tparam REAL_TYPE The type of real numbers used for floating point data.
 * @tparam N The number of points in the spacing/quadrature rule.
 */
template< typename REAL_TYPE, int N >
struct QuadratureGaussLegendre : public GaussLegendreSpacing< REAL_TYPE, N >
{
  /**
   * @brief Returns the weight of a point defined by the quadrature rule.
   * @param index The index of the point.
   * @return The weight of the point.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE
  weight( int const index )
  {
    static_assert( N >= 2 && N <= 4, "Unsupported number of points" );
    if constexpr ( N == 2 )
    {
      return 1.0;
    }
    else if constexpr ( N == 3 )
    {
      //assert( index >= 0 && index < 3 );
      return 0.5555555555555555555555555555555556 +
             0.3333333333333333333333333333333333 * ( index & 1 );
    }
    else if constexpr ( N == 4 )
    {
      //assert( index >= 0 && index < 4 );
      return 0.5 + ( -1 + ( ( ( index + 1 ) & 2 ) ) ) * 0.15214515486254614262693605077800059277;
    }
    return std::numeric_limits< REAL_TYPE >::max();
  }

  /**
   * @brief Returns the weight of a point defined by the quadrature rule.
   * @tparam INDEX The index of the point.
   * @return The weight of the point.
   * @note This function is only available if INDEX is a compile time constant.
   */
  template< int INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE
  weight()
  {
    static_assert( N >= 2 && N <= 4, "Unsupported number of points" );
    if constexpr ( N == 2 )
    {
      return 1.0;
    }
    else if constexpr ( N == 3 )
    {
      if constexpr ( INDEX == 0 ) return 0.5555555555555555555555555555555556; // 5.0
                                                                               // /
                                                                               // 9.0;
      if constexpr ( INDEX == 1 ) return 0.8888888888888888888888888888888889; // 8.0
                                                                               // /
                                                                               // 9.0;
      if constexpr ( INDEX == 2 ) return 0.5555555555555555555555555555555556; // 5.0
                                                                               // /
                                                                               // 9.0;
    }
    else if constexpr ( N == 4 )
    {
      if constexpr ( INDEX == 0 ) return 0.5 - 0.15214515486254614262693605077800059277; // (18.0
                                                                                         // -
                                                                                         // sqrt(30.0))
                                                                                         // /
                                                                                         // 36.0;
      if constexpr ( INDEX == 1 ) return 0.5 + 0.15214515486254614262693605077800059277; // (18.0
                                                                                         // +
                                                                                         // sqrt(30.0))
                                                                                         // /
                                                                                         // 36.0;
      if constexpr ( INDEX == 2 ) return 0.5 + 0.15214515486254614262693605077800059277; // (18.0
                                                                                         // +
                                                                                         // sqrt(30.0))
                                                                                         // /
                                                                                         // 36.0;
      if constexpr ( INDEX == 3 ) return 0.5 - 0.15214515486254614262693605077800059277; // (18.0
                                                                                         // -
                                                                                         // sqrt(30.0))
                                                                                         // /
                                                                                         // 36.0;
    }
    return std::numeric_limits< REAL_TYPE >::max();
  }
};

/**
 * @brief This struct provides a static constexpr functions to compute the
 * quadrature weights on a line that correspond with the coordinates of the
 * GaussLobattoSpacing.
 * @tparam REAL_TYPE The type of real numbers used for floating point data.
 * @tparam N The number of points in the spacing/quadrature rule.
 */
template< typename REAL_TYPE, int N >
struct QuadratureGaussLobatto : public GaussLobattoSpacing< REAL_TYPE, N >
{
  /**
   * @brief Returns the weight of a point defined by the Guass-Lobatto
   * quadrature rule.
   * @param index The index of the point.
   * @return The weight of the point.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE
  weight( int const index )
  {
    static_assert( N >= 2 && N <= 5, "Unsupported number of points" );
    if constexpr ( N == 2 )
    {
      return 1.0;
    }
    else if constexpr ( N == 3 )
    {
      //assert( index >= 0 && index < 3 );
      return 0.3333333333333333333333333333333333 + ( index & 1 );
    }
    else if constexpr ( N == 4 )
    {
      //assert( index >= 0 && index < 4 );
      return 0.1666666666666666666666666666666667 + ( ((index + 1) & 2) >> 1 ) * 0.6666666666666666666666666666666667;
    }
    else if constexpr ( N == 5 )
    {
      //assert( index >= 0 && index < 5 );
      return 0.1 + (index & 1) * 0.4444444444444444444444444444444444 + !( index - 2 ) * 0.6111111111111111111111111111111111;
    }
    return 0;//std::numeric_limits< REAL_TYPE >::max();
  }

  /**
   * @brief Returns the weight of a point defined by the Guass-Lobatto
   * quadrature rule.
   * @tparam INDEX The index of the point.
   * @return The weight of the point.
   * @note This function is only available if INDEX is a compile time constant.
   */
  template< int INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE
  weight()
  {
    static_assert( N >= 2 && N <= 9, "Unsupported number of points" );
    if constexpr ( N == 2 )
    {
      return 1.0;
    }
    else if constexpr ( N == 3 )
    {
      return 0.3333333333333333333333333333333333 + ( INDEX & 1 );
    }
    else if constexpr ( N == 4 )
    {
      if constexpr ( INDEX == 0 ) return 0.1666666666666666666666666666666667;
      if constexpr ( INDEX == 1 ) return 0.8333333333333333333333333333333333;
      if constexpr ( INDEX == 2 ) return 0.8333333333333333333333333333333333;
      if constexpr ( INDEX == 3 ) return 0.1666666666666666666666666666666667;
    }
    else if constexpr ( N == 5 )
    {
      if constexpr ( INDEX == 0 ) return 0.1;
      if constexpr ( INDEX == 1 ) return 0.5444444444444444444444444444444444;
      if constexpr ( INDEX == 2 ) return 0.7111111111111111111111111111111111;
      if constexpr ( INDEX == 3 ) return 0.5444444444444444444444444444444444;
      if constexpr ( INDEX == 4 ) return 0.1;
    }
    else if constexpr ( N == 6 )
    {
      if constexpr ( INDEX == 0 ) return 0.06666666666667;
      if constexpr ( INDEX == 1 ) return 0.37847495629785;
      if constexpr ( INDEX == 2 ) return 0.55485837703548;
      if constexpr ( INDEX == 3 ) return 0.55485837703548;
      if constexpr ( INDEX == 4 ) return 0.37847495629785;
      if constexpr ( INDEX == 5 ) return 0.06666666666667;
    }
    else if constexpr ( N == 7 )
    {
      if constexpr ( INDEX == 0 ) return 0.047619047619048;
      if constexpr ( INDEX == 1 ) return 0.27682604736157;
      if constexpr ( INDEX == 2 ) return 0.43174538120986;
      if constexpr ( INDEX == 3 ) return 0.48761904761905;
      if constexpr ( INDEX == 4 ) return 0.43174538120986;
      if constexpr ( INDEX == 5 ) return 0.27682604736157;
      if constexpr ( INDEX == 6 ) return 0.047619047619048;
    }
    else if constexpr ( N == 8 )
    {
      if constexpr ( INDEX == 0 ) return 0.035714285714286;
      if constexpr ( INDEX == 1 ) return 0.2107042271435;
      if constexpr ( INDEX == 2 ) return 0.34112269248351;
      if constexpr ( INDEX == 3 ) return 0.4124587946587;
      if constexpr ( INDEX == 4 ) return 0.4124587946587;
      if constexpr ( INDEX == 5 ) return 0.3411226924835;
      if constexpr ( INDEX == 6 ) return 0.21070422714351;
      if constexpr ( INDEX == 7 ) return 0.035714285714286;
    }
    else if constexpr ( N == 9 )
    {
      if constexpr ( INDEX == 0 ) return 0.02777777777778;
      if constexpr ( INDEX == 1 ) return 0.16549536156081;
      if constexpr ( INDEX == 2 ) return 0.27453871250016;
      if constexpr ( INDEX == 3 ) return 0.34642851097305;
      if constexpr ( INDEX == 4 ) return 0.37151927437642;
      if constexpr ( INDEX == 5 ) return 0.34642851097305;
      if constexpr ( INDEX == 6 ) return 0.27453871250016;
      if constexpr ( INDEX == 7 ) return 0.16549536156081;
      if constexpr ( INDEX == 8 ) return 0.02777777777778;
    }
    return std::numeric_limits< REAL_TYPE >::max();
  }
};

} // namespace shiva
