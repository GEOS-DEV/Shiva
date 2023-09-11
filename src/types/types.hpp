

#pragma once

namespace shiva
{

template< typename ... T >
using tuple = std::tuple< T ... >;

template< typename ... T >
auto make_tuple( T && ... t )
{
  return std::make_tuple( std::forward< T >( t ) ... );
}


template< typename T, int N >
struct CArray1d
{
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