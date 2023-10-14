

#pragma once

#include <tuple>

namespace shiva
{

template< typename ... T >
using tuple = std::tuple< T ... >;

template< typename ... T >
auto make_tuple( T && ... t )
{
  return std::make_tuple( std::forward< T >( t ) ... );
}

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
