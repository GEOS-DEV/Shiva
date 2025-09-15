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
 * @file MultiIndex.hpp
 * @brief MultiIndex.hpp provides a templated multi-index type.
 */

#pragma once

#include "common/ShivaMacros.hpp"
#include "SequenceUtilities.hpp"

#include <utility>

namespace shiva
{

/**
 * @brief MultiIndex is a templated multi-index type consisting of NUM_INDICES
 * indices of type BASE_INDEX_TYPE.
 * @tparam BASE_INDEX_TYPE The type of the base index.
 * @tparam NUM_INDICES The number of indices.
 */
template< typename BASE_INDEX_TYPE, int NUM_INDICES >
struct MultiIndex
{
  /// alias for BASE_INDEX_TYPE
  using BaseIndexType = BASE_INDEX_TYPE;

  /// contains the indices
  BASE_INDEX_TYPE data[NUM_INDICES] = {0};
};


/**
 * @brief MultiIndexRange is a templated multi-index type consisting of
 * NUM_INDICES indices of type BASE_INDEX_TYPE. It also contains an array of
 * ranges for each index s.t. a linear index may be calcuated.
 * @tparam BASE_INDEX_TYPE The type of the base index.
 * @tparam NUM_INDICES The number of indices.
 * @tparam RANGES The ranges for each index.
 */
template< typename BASE_INDEX_TYPE, BASE_INDEX_TYPE... RANGES >
struct MultiIndexRange
{
  /// alias for BASE_INDEX_TYPE
  using BaseIndexType = BASE_INDEX_TYPE;

  /// contains the number of indices in the multi-index
  static constexpr int NUM_INDICES = sizeof...(RANGES);

  /// contains the range for each index
  static constexpr int ranges[NUM_INDICES] = { RANGES ...};

  /**
   * @brief Returns the range for a given index.
   * @param i The index.
   * @return The range for the given index.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int range( int const i )
  {
    return ranges[i];
  }

  /// contains the indices
  BASE_INDEX_TYPE data[NUM_INDICES] = {0};
};

/**
 * @brief Alias for a MultiIndexRange of type int.
 * @tparam ...RANGES The ranges for each index.
 */
template< int... RANGES >
using MultiIndexRangeI = MultiIndexRange< int, RANGES... >;

} // namespace shiva
