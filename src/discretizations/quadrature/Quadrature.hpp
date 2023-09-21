#pragma once

#include "../spacing/Spacing.hpp"

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
      if( index == 0 ) return 0.5555555555555555555555555555555556; // 5.0 / 9.0;
      if( index == 1 ) return 0.8888888888888888888888888888888889; // 8.0 / 9.0;
      if( index == 2 ) return 0.5555555555555555555555555555555556; // 5.0 / 9.0;
    }
    else if constexpr ( N==4 )
    {
      if( index == 0 ) return 0.5 - 0.15214515486254614262693605077800059277; // (18.0 - sqrt(30.0)) / 36.0;
      if( index == 1 ) return 0.5 + 0.15214515486254614262693605077800059277; // (18.0 + sqrt(30.0)) / 36.0;
      if( index == 2 ) return 0.5 + 0.15214515486254614262693605077800059277; // (18.0 + sqrt(30.0)) / 36.0;
      if( index == 3 ) return 0.5 - 0.15214515486254614262693605077800059277; // (18.0 - sqrt(30.0)) / 36.0;
    }
    return 0;
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
    return 0;
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
      return 0.3333333333333333333333333333333333 + ( index & 1 );
    }
    else if constexpr ( N==4 )
    {
      if( index == 0 ) return 0.1666666666666666666666666666666667;
      if( index == 1 ) return 0.8333333333333333333333333333333333;
      if( index == 2 ) return 0.8333333333333333333333333333333333;
      if( index == 3 ) return 0.1666666666666666666666666666666667;
    }
    else if constexpr ( N==5 )
    {
      if( index == 0 ) return 0.1;
      if( index == 1 ) return 0.5444444444444444444444444444444444;
      if( index == 2 ) return 0.7111111111111111111111111111111111;
      if( index == 3 ) return 0.5444444444444444444444444444444444;
      if( index == 4 ) return 0.1;
    }
    return 0;
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
      return 0.3333333333333333333333333333333333 + ( INDEX & 1 );
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
    else if constexpr ( N==6 )
    {
      if constexpr ( INDEX == 0 ) return 0.06666666666667;
      if constexpr ( INDEX == 1 ) return 0.37847495629785;
      if constexpr ( INDEX == 2 ) return 0.55485837703548;
      if constexpr ( INDEX == 3 ) return 0.55485837703548;
      if constexpr ( INDEX == 4 ) return 0.37847495629785;
      if constexpr ( INDEX == 5 ) return 0.06666666666667;
    }
    else if constexpr (N==7)
    {
      if constexpr ( INDEX == 0 ) return 0.047619047619048;
      if constexpr ( INDEX == 1 ) return 0.27682604736157;
      if constexpr ( INDEX == 2 ) return 0.43174538120986;
      if constexpr ( INDEX == 3 ) return 0.48761904761905;
      if constexpr ( INDEX == 4 ) return 0.43174538120986;
      if constexpr ( INDEX == 5 ) return 0.27682604736157;
      if constexpr ( INDEX == 6 ) return 0.047619047619048;
    }
    else if constexpr (N==8)
    {
      if constexpr ( INDEX == 0 ) return 0.035714285714286;
      if constexpr ( INDEX == 1 ) return 0.2107042271435;
      if constexpr ( INDEX == 2 ) return 0.34112269248351;
      if constexpr ( INDEX == 3 ) return 0.4124587946587;
      if constexpr ( INDEX == 4 ) return 0.4124587946587;
      if constexpr ( INDEX == 5 ) return 0.3411226924835;
      if constexpr ( INDEX == 6 ) return 0.21070422714351;
      if constexpr ( INDEX == 7 ) return 0.035714285714286;
    }
    else if constexpr (N==9)
    {
      if constexpr ( INDEX == 0 ) return 0.02777777777778;
      if constexpr ( INDEX == 1 ) return 0.16549536156081;
      if constexpr ( INDEX == 2 ) return 0.27453871250016;
      if constexpr ( INDEX == 3 ) return 0.34642851097305;
      if constexpr ( INDEX == 4 ) return 0.37151927437642;
      if constexpr ( INDEX == 5 ) return 0.34642851097305;
      if constexpr ( INDEX == 6 ) return 0.27453871250016;
      if constexpr ( INDEX == 7 ) return 0.16549536156081;
      if constexpr ( INDEX == 8 ) return 0.02777777777778;
    }
    return 0;
  }
};

} // namespace shiva


