#pragma once

#include "common/ShivaMacros.hpp"


#include <type_traits>
#include <utility>

namespace shiva
{


namespace detail
{
template< typename ... INTEGER_SEQUENCES >
struct SequenceExpansion
{};

template< int ... DIMENSION_INDICES >
struct SequenceExpansion< std::integer_sequence< int, DIMENSION_INDICES... > >
{
  template< typename FUNC, typename ... ARGS >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto execute( FUNC && func, ARGS && ... args )
  {
    if constexpr ( std::is_invocable_v< FUNC, std::integral_constant< int, DIMENSION_INDICES >..., ARGS ... > )
    {
      return func( std::forward< ARGS >( args )...,
                   std::integral_constant< int, DIMENSION_INDICES >{} ... );
    }
    else
    {
      return func.template operator()< DIMENSION_INDICES... >(std::forward< ARGS >( args )...);
    }
  }

  template< typename FUNC >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto staticFor( FUNC && func )
  {
    if constexpr ( std::is_invocable_v< FUNC, std::integral_constant< int, 0 > > )
    {
      return (func( std::integral_constant< int, DIMENSION_INDICES >{} ), ...);
    }
    else
    {
      return (func.template operator()< DIMENSION_INDICES >(), ...);
    }
  }
};
}

template< int END, typename FUNC, typename ... ARGS >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto executeSequence( FUNC && func,
                                ARGS && ... args )
{
  return
    detail::SequenceExpansion< std::make_integer_sequence< int, END > >::
    template execute( std::forward< FUNC >( func ),
                      std::forward< ARGS >( args )... );
}

template< int END, typename FUNC >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto forSequence( FUNC && func )
{
  return
    detail::SequenceExpansion< std::make_integer_sequence< int, END > >::
    template staticFor( std::forward< FUNC >( func ) );
}

} // namespace shiva
