#pragma once

namespace shiva
{

template< typename REAL_TYPE, int N >
struct QuadratureGaussLegendre : public GaussLegendreSpacing<REAL_TYPE,N>
{
  constexpr static REAL_TYPE weight( int const index ) 
  { 
    if constexpr ( N==2 )
    {
      return 1.0;
    }
    else if constexpr ( N==3 )
    {
      if constexpr ( index == 0 ) return 0.5555555555555555555555555555555556; // 5.0 / 9.0;
      if constexpr ( index == 1 ) return 0.8888888888888888888888888888888889; // 8.0 / 9.0;
      if constexpr ( index == 2 ) return 0.5555555555555555555555555555555556; // 5.0 / 9.0;
    }
    else if constexpr ( N==4 )
    {
      if constexpr ( index == 0 ) return 0.5 - 0.15214515486254614262693605077800059277; // (18.0 - sqrt(30.0)) / 36.0;
      if constexpr ( index == 1 ) return 0.5 + 0.15214515486254614262693605077800059277; // (18.0 + sqrt(30.0)) / 36.0;
      if constexpr ( index == 2 ) return 0.5 + 0.15214515486254614262693605077800059277; // (18.0 + sqrt(30.0)) / 36.0;
      if constexpr ( index == 3 ) return 0.5 - 0.15214515486254614262693605077800059277; // (18.0 - sqrt(30.0)) / 36.0;
    }
  }

  template< int INDEX >
  constexpr static REAL_TYPE weight()
  { 
    if constexpr ( N==2 )
    {
      return 1.0;
    }
    else if constexpr ( N==3 )
    {
      if constexpr ( INDEX == 0 ) return 0.5555555555555555555555555555555556; // 5.0 / 9.0;
      if constexpr ( INDEX == 1 ) return 0.8888888888888888888888888888888889; // 8.0 / 9.0;
      if constexpr ( INDEX == 2 ) return 0.5555555555555555555555555555555556; // 5.0 / 9.0;
    }
    else if constexpr ( N==4 )
    {
      if constexpr ( INDEX == 0 ) return 0.5 - 0.15214515486254614262693605077800059277; // (18.0 - sqrt(30.0)) / 36.0;
      if constexpr ( INDEX == 1 ) return 0.5 + 0.15214515486254614262693605077800059277; // (18.0 + sqrt(30.0)) / 36.0;
      if constexpr ( INDEX == 2 ) return 0.5 + 0.15214515486254614262693605077800059277; // (18.0 + sqrt(30.0)) / 36.0;
      if constexpr ( INDEX == 3 ) return 0.5 - 0.15214515486254614262693605077800059277; // (18.0 - sqrt(30.0)) / 36.0;
    }
  }
};

template< typename REAL_TYPE, int N >
struct QuadratureGaussLobatto : public GaussLobattoSpacing<REAL_TYPE, N>
{
  constexpr static REAL_TYPE weight( int const index ) 
  { 
    if constexpr ( N==2 )
    {
      return 1.0;
    }
    else if constexpr ( N==3 )
    {
      return 0.3333333333333333333333333333333333 + ( index & 1 ) * 0.6666666666666666666666666666666667;
    }
    else if constexpr ( N==4 )
    {
      if constexpr ( INDEX == 0 ) return 0.1666666666666666666666666666666667;
      if constexpr ( INDEX == 1 ) return 0.8333333333333333333333333333333333;
      if constexpr ( INDEX == 2 ) return 0.8333333333333333333333333333333333;
      if constexpr ( INDEX == 3 ) return 0.1666666666666666666666666666666667;
    }
    else if constexpr ( N==5 )
    {
      if constexpr ( INDEX == 0 ) return 0.1;
      if constexpr ( INDEX == 1 ) return 0.5444444444444444444444444444444444;
      if constexpr ( INDEX == 2 ) return 0.7111111111111111111111111111111111;
      if constexpr ( INDEX == 3 ) return 0.5444444444444444444444444444444444;
      if constexpr ( INDEX == 4 ) return 0.1;
    }

}