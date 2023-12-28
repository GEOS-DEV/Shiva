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

#include "common/ShivaMacros.hpp"
#include <type_traits>
#include <utility>

#include "SequenceUtilities.hpp"

namespace shiva
{

/**
 * @namespace shiva::nestedSequenceUtilitiesImpl
 * @brief The nestedSequenceUtilitiesImpl namespace contains implementation details
 * for NestedSequenceUtilities.
 */
namespace nestedSequenceUtilitiesImpl
{

/**
 * @brief Enables the expansion of a nested sequence of for loops through
 * recursion.
 * @tparam END The current end of the first range.
 * @tparam ...REMAINING_ENDS The remaining ends of the ranges.
 */
template< int END, int... REMAINING_ENDS >
struct NestedSequenceExpansion
{

  /**
   * @brief Executes a nested static for loop through recursion.
   * @tparam FUNC The type of the function that contains the body of the for
   * loop.
   * @tparam ...INDICES Type of the indices of the for loop.
   * @param func The function that contains the body of the for
   * loop.
   * @param ...indices The indices for the loop.
   */
  template< typename FUNC, typename ... INDICES >
  static constexpr void staticForNested( FUNC && func, INDICES ... indices )
  {
    if constexpr ( sizeof...(REMAINING_ENDS) == 0 )
    {
      // Base case: execute the function with all accumulated indices
      forSequence< END >( [&] ( auto idx )
      {
        func( indices ..., idx );
      } );
    }
    else
    {
      // Recursive case: expand the next loop dimension
      forSequence< END >( [&] ( auto idx )
      {
        NestedSequenceExpansion< REMAINING_ENDS... >::staticForNested( std::forward< FUNC >( func ), indices ..., idx );
      } );
    }
  }
};

}

/**
 * @brief Executes a nested static for loop through recursion.
 * @tparam ENDS The ends of the ranges.
 * @tparam FUNC The type of the function that contains the body of the for
 * loop.
 * @param func The function that contains the body of the nested for
 * loop.
 */
template< int... ENDS, typename FUNC >
constexpr void forNestedSequence( FUNC && func )
{
  nestedSequenceUtilitiesImpl::NestedSequenceExpansion< ENDS... >::staticForNested( std::forward< FUNC >( func ) );
}

/**
 * @brief Executes a nested static for loop through recursion that takes in an
 * integer_sequence to set the ends of the ranges.
 * @tparam ENDS The ends of the ranges.
 * @tparam FUNC The type of the function that contains the body of the for
 * loop.
 * @param func The function that contains the body of the nested for
 * loop.
 */
template< int... ENDS, typename FUNC >
constexpr void forNestedSequence( std::integer_sequence< int, ENDS... >,
                                  FUNC && func )
{
  forNestedSequence< ENDS... >( std::forward< FUNC >( func ) );
}

}
