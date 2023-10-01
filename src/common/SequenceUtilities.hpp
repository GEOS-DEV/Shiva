#pragma once

#include <type_traits>
#include <utility>

namespace shiva
{

namespace detail
{
template< typename... INTEGER_SEQUENCES >
struct SequenceExpansion
{};

template< int ... DIMENSION_INDICES >
struct SequenceExpansion< std::integer_sequence<int, DIMENSION_INDICES...> >
{
  template< typename FUNC, typename ... ARGS >
  constexpr static auto execute( FUNC && func, ARGS && ... args )
  {
    if constexpr ( !std::is_invocable_v<FUNC,int,ARGS...> )
    {
      return func.template operator()<DIMENSION_INDICES...,ARGS...>(std::forward<ARGS>(args)...);
    }
    else
    {
      return func(DIMENSION_INDICES...,std::forward<ARGS>(args)...);
    }
  }

  template< typename FUNC, typename ... ARGS >
  constexpr static auto execute( ARGS && ... args )
  {
    if constexpr ( std::is_class_v< FUNC >)
    {
      return FUNC::template execute<DIMENSION_INDICES...,ARGS...>( std::forward<ARGS>(args)... );
    }
  }

  template< typename FUNC >
  constexpr static void staticFor( FUNC && func )
  {
    if constexpr ( !std::is_invocable_v<FUNC,int> )
    {
      (func.template operator()<DIMENSION_INDICES>(),...);
    }
    else
    {
      (func(DIMENSION_INDICES),...);
    }
  }
};
}

template< int END, typename FUNC, typename ... ARGS >
constexpr auto executeSequence( FUNC&& func, ARGS && ... args )
{
  return detail::SequenceExpansion<std::make_integer_sequence<int,END> >::template execute( std::forward<FUNC>(func), std::forward<ARGS>(args)... );
}

template< int END, typename FUNC, typename ... ARGS >
constexpr auto executeSequence( ARGS && ... args )
{
  return detail::SequenceExpansion<std::make_integer_sequence<int,END> >::template execute<FUNC>( std::forward<ARGS>(args)... );
}


template< int END, typename FUNC  >
constexpr void forSequence( FUNC&& func )
{
  detail::SequenceExpansion<std::make_integer_sequence<int,END> >::template staticFor( std::forward<FUNC>(func) );
}

} // namespace shiva