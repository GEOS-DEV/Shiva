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
 * @file IndexTypes.hpp
 * @brief functions that utilize portable index types
 */

#pragma once

#include "LinearIndex.hpp"
#include "MultiIndex.hpp"

#include <utility>

namespace shiva
{
namespace detail
{

/**
 * @brief Helper function for calculating a stride.
 * @tparam INDICES variaic pack of the indices.
 * @param ranges The ranges.
 * @return The largest stride for the index pack.
 */
template< int ... INDICES >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
strideHelper( std::integer_sequence< int, INDICES... >,
              int const * const ranges )
{
  // return result of a fold expression that multiples all entries of the ranges
  // array
  return ( ranges[INDICES] * ... *1);
}

/**
 * @brief function for calculating the strides of a multi-index.
 * @tparam INDEX The index.
 * @tparam BASE_INDEX_TYPE the base type of the index...i.e. "int"
 * @tparam RANGES variaic pack of the ranges.
 * @param index The MultiIndexRange for which the stride will be calclauted.
 * @return The stride.
 */
template< int INDEX, typename BASE_INDEX_TYPE, BASE_INDEX_TYPE... RANGES >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE BASE_INDEX_TYPE
stride( MultiIndexRange< BASE_INDEX_TYPE, RANGES... > const & index )
{
  using IndexType = MultiIndexRange< BASE_INDEX_TYPE, RANGES... >;

  // call strideHelper on an integer_sequence that ranges from 0...(NUM_INDICES
  // - 1 - INDEX).
  // This will result in a fold expression that multiples entries of the ranges
  // array for the integer_sequence.
  return detail::strideHelper( std::make_integer_sequence< int, (IndexType::NUM_INDICES - 1) - INDEX >{},
                               &(index.ranges[0]) );
}

/**
 * @brief Helper function for calculating the linear index of a multi-index.
 * @tparam T The type of the multi-index.
 * @tparam INDICES variaic pack of the indices.
 * @param index The multi-index.
 * @return The linear index.
 */
template< typename T, int... INDICES >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE typename T::BaseIndexType
linearIndexHelper( T const & index,
                   std::integer_sequence< int, INDICES... > )
{
  // return a fold expression that adds the index*stride for each component of
  // the multi-index.
  return (  ( index.data[INDICES] * stride< INDICES >( index ) ) + ... );
}


/**
 * @brief Helper function for executing an iteration over MultiIndexRange
 * object.
 * @tparam DIM The index/dimension of the current iterate
 * @tparam BASE_INDEX_TYPE The type of the base index.
 * @tparam RANGES The pack of ranges of the multi-index.
 * @tparam FUNC The type of function to be called in the iteration.
 * @param start The starting multi-index.
 * @param index The multi-index. (I think this isn't required...)
 * @param func The function to be called in the iteration.
 */
template< int DIM, typename BASE_INDEX_TYPE, BASE_INDEX_TYPE... RANGES, typename FUNC >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
forRangeHelper( MultiIndexRange< BASE_INDEX_TYPE, RANGES... > const & start,
                MultiIndexRange< BASE_INDEX_TYPE, RANGES... > & index,
                FUNC && func )
{
  using IndexType = MultiIndexRange< BASE_INDEX_TYPE, RANGES... >;
  // if we are at the last dimension, then call func on index
  if constexpr ( DIM == (IndexType::NUM_INDICES - 1) )
  {
    int & a = index.data[DIM];
    for ( a = start.data[DIM]; a < index.range( DIM ); ++a )
    {
      func( index );
    }
  }
  // otherwise, recurse until we get to the last index
  else
  {
    int & a = index.data[DIM];
    for ( a = start.data[DIM]; a < index.range( DIM ); ++a )
    {
      forRangeHelper< DIM + 1 >( start, index, func );
    }
  }
}

} // namespace detail


/**
 * @brief Returns the linear index of a multi-index.
 * @tparam BASE_INDEX_TYPE The type of the base index.
 * @tparam RANGES The ranges of the multi-index.
 * @param index The multi-index.
 * @return The linear index.
 */
template< typename BASE_INDEX_TYPE, BASE_INDEX_TYPE... RANGES >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE BASE_INDEX_TYPE
linearIndex( MultiIndexRange< BASE_INDEX_TYPE, RANGES... > const & index )
{
  using IndexType = MultiIndexRange< BASE_INDEX_TYPE, RANGES... >;
  return detail::linearIndexHelper( index, std::make_integer_sequence< int, IndexType::NUM_INDICES >{} );
}

/**
 * @brief Returns the linear index of a LinearIndex.
 * @tparam INDEX_TYPE The type of the index.
 * @param index The index.
 * @return The linear index.
 */
template< typename INDEX_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE INDEX_TYPE
linearIndex( LinearIndex< INDEX_TYPE > const & index )
{
  return index;
}


/**
 * @brief Executes an iteration over a MultiIndexRange object.
 * @tparam BASE_INDEX_TYPE The type of the base index.
 * @tparam RANGES The ranges of the multi-index.
 * @param index The multi-index.
 * @param func The function to be called in the iteration.
 */
template< typename BASE_INDEX_TYPE, BASE_INDEX_TYPE... RANGES, typename FUNC >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
forRange( MultiIndexRange< BASE_INDEX_TYPE, RANGES... > & index,
          FUNC && func )
{
  using IndexType = MultiIndexRange< BASE_INDEX_TYPE, RANGES... >;
  IndexType const start = index;
  detail::forRangeHelper< 0 >( start, index, std::forward< FUNC >( func ) );
}


} // namespace shiva
