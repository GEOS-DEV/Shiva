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
 * @file CArrayHelper.hpp
 * @brief This file contains the helper functions for the MultiDimensionalArray and MultiDimensionalSpan.
 */

#pragma once


#include "common/ShivaErrorHandling.hpp"
#include "common/types.hpp"
#include <utility>
#include <cstdint>
#include <cinttypes>

namespace shiva
{

/**
 * @namespace shiva::CArrayHelper
 * @brief The CArrayHelper namespace contains some stride calculations and
 * linearIndex calculations for the MultiDimensionalBase class.
 */
namespace CArrayHelper
{


/**
 * @brief This recursive function helps to calculates the strides for
 *   dimensions passed in as a template parameter pack.
 * @tparam DIM This is the first dimension peeled off the parameter pack.
 * @tparam DIMS These are the remaining dimensions of the pack.
 * @return This returns the stride of the largest dimension in the pack.
 */
template< int DIM, int ... DIMS >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
strideHelper()
{
  if constexpr ( sizeof ... ( DIMS ) == 0 )
  {
    return DIM;
  }
  else
  {
    return DIM * strideHelper< DIMS ... >();
  }
}

/**
 * @brief This function calculates the stride of the largest dimension in the
 *   pack.
 * @tparam DIM This is the first dimension peeled off the parameter pack.
 * @tparam DIMS These are the remaining dimensions of the pack.
 * @return This returns the stride of the largest dimension in the pack.
 */
template< int DIM, int ... DIMS >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
stride()
{
  return strideHelper< DIMS ..., 1 >();
}

/**
 * @struct linearIndexHelper
 * @brief struct to facilitate the calculation of a linear index from a pack
 * of dimensions by peeling the dimensions pack one at a time.
 * @tparam DIM This is the first dimension peeled off the parameter pack.
 * @tparam DIMS These are the remaining dimensions of the pack.
 */
template< int DIM, int ... DIMS >
struct linearIndexHelper
{
  /**
   * @brief This recursive function calculates the linear index from a pack
   * of indices by peeling off the indices one at a time, and multiplying
   * by the stride at that "level" of the product sum.
   * @tparam INDEX This is the first index peeled off the parameter pack.
   * @tparam INDICES These are the remaining indices of the pack.
   * @return This returns the linear index.
   */
  template< int INDEX, int ... INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
  level()
  {
    static_assert( INDEX >= 0 && INDEX < DIM );
    constexpr int thisStride = strideHelper< DIMS ..., 1 >();
    if constexpr ( sizeof ... ( DIMS ) == 0 )
    {
      return INDEX * thisStride;
    }
    else
    {
      return INDEX * thisStride + linearIndexHelper< DIMS ... >::template level< INDICES ... >();
    }
  }

  /**
   * @copydoc level()
   */
  template< typename INDEX_TYPE, typename ... INDICES_TYPE >
  static constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE int
  level( INDEX_TYPE const index, INDICES_TYPE const ... indices )
  {
#if defined( SHIVA_USE_BOUNDS_CHECK )
    SHIVA_ASSERT_MSG( index >= 0 && index < DIM,
                      "Index out of bounds: 0 < index(%jd) < dim(%jd)",
                      static_cast< intmax_t >( index ),
                      static_cast< intmax_t >( DIM ) );
#endif
    constexpr int thisStride = strideHelper< DIMS ..., 1 >();
    if constexpr ( sizeof ... ( DIMS ) == 0 )
    {
      return index * thisStride;
    }
    else
    {
      return index * thisStride + linearIndexHelper< DIMS ... >::template level< INDICES_TYPE ... >( std::forward< INDICES_TYPE const >( indices )... );
    }
  }

  /**
   * @brief This function calculates the linear index from a pack of indices.
   * @tparam INDICES The indices to calculate the linear index from.
   * @return This returns the linear index.
   */
  template< int ... INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
  eval()
  {
    return level< INDICES ... >();
  }

  /**
   * @brief This function calculates the linear index from a pack of indices.
   * @tparam INDEX_TYPE The type of the index.
   * @param indices The indices to calculate the linear index from.
   * @return This returns the linear index.
   */
  template< typename ... INDEX_TYPE >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
  eval( INDEX_TYPE ... indices )
  {
    return level( std::forward< INDEX_TYPE >( indices )... );
  }
};


/**
 * @brief function to get a specific value from a pack of indices.
 * @tparam COUNT The index of the pack to return.
 * @tparam INDEX The first index peeled off the pack.
 * @tparam INDICES The remaining indices in the pack.
 * @return The value of the INDICES pack at COUNT.
 */
template< int COUNT, int INDEX, int ... INDICES >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
get()
{
  if constexpr ( COUNT == 0 )
  {
    return INDEX;
  }
  else
  {
    return get< COUNT - 1, INDICES ... >();
  }
}

/**
 * @brief function to get the product size of a pack of indices.
 * @tparam INDICES The indices to get the aggregate length of.
 */
template< int ... DIMS >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
size()
{
  return ( DIMS * ... );
}

/**
 * @brief function to peel a value from a parameter pack.
 * @tparam T The type of the first value in the pack.
 * @tparam Ts The types of the remaining values in the pack.
 */
template< typename T, typename ... Ts >
struct Peeler
{
  /// The type of the first value in the pack.
  using type = T;

  /// The type of the remaining values in the pack.
  using types = tuple< Ts ... >;
};

/**
 * @brief function to peel an integer from a parameter pack.
 * @tparam INT The first value in the pack.
 * @tparam INTS The remaining values in the pack.
 */
template< int INT, int ... INTS >
struct IntPeeler
{
  /// The type of the first value in the pack.
  static constexpr int first = INT;

  /// The type of the remaining values in the pack.
  using rest = std::integer_sequence< int, INTS... >;
};

/**
 * @brief function to apply a parameter pack to a templated type.
 * @tparam Ints The integer sequence to apply.
 * @tparam Target The target type to apply the integer sequence to.
 */
template< typename Seq, template< int... > class Target >
struct ApplyDims;

/**
 * @brief function to apply a parameter pack to a templated type.
 * @tparam Ints The integer sequence to apply.
 * @tparam Target The target type to apply the integer sequence to.
 */
template< int... Ints, template< int... > class Target >
struct ApplyDims< std::integer_sequence< int, Ints... >, Target >
{
  /// The type of the target that has the pack applied its varaidic pack.
  using type = Target< Ints... >;
};

} // namespace CArrayHelper
} // namespace shiva
