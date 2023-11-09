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

template< int... Is, typename FUNC >
constexpr void forNestedSequence( FUNC && func, std::integer_sequence<int, Is...> )
{
  forNestedSequence< Is... >( std::forward<FUNC>( func ) );
}

// template< typename CArray, std::size_t... I, typename FUNC >
// constexpr void forNestedSequenceHelper( const CArray arr, std::index_sequence<I...>, FUNC && func )
// {
//   forNestedSequence< arr[I]... >( std::forward< FUNC >( func ) );
// }

// // Variant to use a C-array for the bounds of each level of the nested loop
// template< typename CArray, typename FUNC >
// constexpr void forNestedSequence( const CArray arr, FUNC && func )
// {
//   forNestedSequenceHelper( arr, std::make_index_sequence< CArray::size >{}, std::forward< FUNC >( func ) );
// }

}