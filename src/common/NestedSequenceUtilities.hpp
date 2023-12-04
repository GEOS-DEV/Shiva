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

template < int... ENDS >
struct NestedSequenceExpansion
{};

template< int END, int... REMAINING_ENDS >
struct NestedSequenceExpansion< END, REMAINING_ENDS... >
{
  template< typename FUNC, typename... INDICES >
  static constexpr void staticForNested( FUNC && func, INDICES... indices )
  {
    if constexpr (sizeof...(REMAINING_ENDS) == 0)
    {
      // Base case: execute the function with all accumulated indices
      forSequence< END >( [&] ( auto idx )
      {
        func( indices..., idx );
      });
    }
    else
    {
      // Recursive case: expand the next loop dimension
      forSequence< END >( [&] ( auto idx )
      {
        NestedSequenceExpansion< REMAINING_ENDS... >::staticForNested( std::forward<FUNC>( func ), indices..., idx );
      });
    }
  }
};

}

template< int... ENDS, typename FUNC >
constexpr void forNestedSequence( FUNC && func )
{
  nestedSequenceUtilitiesImpl::NestedSequenceExpansion< ENDS... >::staticForNested( std::forward< FUNC >( func ) );
}

template< int... ENDS, typename FUNC >
constexpr void forNestedSequence( std::integer_sequence<int, ENDS...>,
                                  FUNC && func )
{
  forNestedSequence< ENDS... >( std::forward<FUNC>( func ) );
}

template < std::size_t Size1, std::size_t Size2, typename FUNC >
constexpr void forNestedSequenceSplit( FUNC && func )
{
  auto wrappedFunc = [&]( auto... indices )
  {
    // Make sequence from pack, Split the sequence
    using SplitResult = Split< std::integer_sequence< decltype( indices )::value ... >, Size1, Size2 >;
    using FirstSeq = typename SplitResult::first;
    using SecondSeq = typename SplitResult::second;

    callWithSequences( func, FirstSeq{}, SecondSeq{} );
  };
  // Call the wrapped function with the original nested sequence expansion
  forNestedSequence( wrappedFunc );
}


}