#pragma once

#include "common/ShivaMacros.hpp"

#define SHIVA_USE_CAMP
#if defined(SHIVA_USE_CAMP)
#include <camp/camp.hpp>
#else

#if defined(SHIVA_USE_CUDA)
#include <cuda/std/tuple>
#else
#include <tuple>
#endif

#endif

namespace shiva
{

#if defined(SHIVA_USE_CAMP)

template< typename ... T >
using tuple = camp::tuple< T ... >;

template< typename ... T >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto make_tuple( T && ... t )
{
  return camp::make_tuple( std::forward< T >( t ) ... );
}

#else
#if defined(SHIVA_USE_CUDA)
template< typename ... T >
using tuple = cuda::std::tuple< T ... >;

template< typename ... T >
auto make_tuple( T && ... t )
{
  return cuda::std::make_tuple( std::forward< T >( t ) ... );
}
#else
template< typename ... T >
using tuple = std::tuple< T ... >;

template< typename ... T >
auto make_tuple( T && ... t )
{
  return std::make_tuple( std::forward< T >( t ) ... );
}
#endif
#endif

template< int ... T >
using int_sequence = std::integer_sequence< int, T... >;

template< int N >
using make_int_sequence = std::make_integer_sequence< int, N >;


template< typename T >
struct Scalar
{
  using type = T;
  T data;
};

template< typename T, int N >
struct CArray1d
{
  constexpr inline T operator[] ( int const i ) const { return data[i]; }
  using type = T[N];
  T data[N];
};

template< typename T, int N, int M >
struct CArray2d
{
  using type = T[N][M];
  T data[N][M];
};

}
