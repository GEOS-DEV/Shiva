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
        func( indices..., decltype(idx)::value );
      });
    } 
    else 
    {
      // Recursive case: expand the next loop dimension
      forSequence< END >( [&] ( auto idx ) 
      {
        NestedSequenceExpansion< REMAINING_ENDS... >::staticForNested( std::forward<FUNC>( func ), indices..., decltype(idx)::value );
      });
    }
  }
};

}


// template< int... ENDS, typename FUNC >
// constexpr void exectueNestedSequence( FUNC && func )
// {
//
// }

template< int... ENDS, typename FUNC >
constexpr void forNestedSequence( FUNC && func ) 
{
  nestedSequenceUtilitiesImpl::NestedSequenceExpansion< ENDS... >::staticForNested( std::forward< FUNC >( func ) );
}

}