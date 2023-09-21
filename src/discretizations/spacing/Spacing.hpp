#pragma once

namespace shiva
{

template< typename REAL_TYPE, int N >
struct EqualSpacing
{
  constexpr static int numPoints = N;

  constexpr static REAL_TYPE interval() { return 2.0 / (numPoints - 1); }
  
  constexpr static REAL_TYPE coordinate( int const index ) { return -1 + index * interval(); }
  
  template< int INDEX >
  constexpr static REAL_TYPE coordinate() { return -1.0 + INDEX * interval(); }
};

template< typename REAL_TYPE, int N >
struct GaussLegendreSpacing
{
  constexpr static REAL_TYPE  invSqrt3 = 0.57735026918962576450914878050195745565; //1/sqrt(3)
  constexpr static REAL_TYPE sqrt3div5 = 0.77459666924148337703585307995647992217; //sqrt(3/5)
  constexpr static REAL_TYPE       c4[2] = { 0.8611363115940525752239464888928095051,   //sqrt((15+2*sqrt(30))/35)
                                             0.3399810435848562648026657591032446872 }; //sqrt((15-2*sqrt(30))/35)

  constexpr static int numPoints = N;

  constexpr static REAL_TYPE coordinate( int const index ) 
  { 
    if constexpr ( N==2 )
    {
      return invSqrt3 * ( -1.0 + index * 2 );
    }
    else if constexpr ( N==3 )
    {
      return sqrt3div5 * ( -1.0 + index );
    }
    else if constexpr ( N==4 )
    {
      return c4[ ( ((index+1)&2)>>1) ] * ((index&2)-1) ;
    }
    return 0;
  }

  template< int INDEX >
  constexpr static REAL_TYPE coordinate()
  { 
    if constexpr ( N==2 )
    {
      return invSqrt3 * ( -1.0 + INDEX * 2.0 );
    }
    else if constexpr ( N==3 )
    {
      return sqrt3div5 * ( -1.0 + INDEX );
    }
    else if constexpr ( N==4 )
    {
      if constexpr ( INDEX == 0 ) return -c4[0];
      if constexpr ( INDEX == 1 ) return -c4[1];
      if constexpr ( INDEX == 2 ) return  c4[1];
      if constexpr ( INDEX == 3 ) return  c4[0];
    }
    return 0;

  }
};


template< typename REAL_TYPE, int N >
struct GaussLobattoSpacing
{
  constexpr static REAL_TYPE  invSqrt5 = 0.44721359549995793928183473374625524709;
  constexpr static REAL_TYPE sqrt3div7 = 0.65465367070797714379829245624685835557;

  constexpr static int numPoints = N;

  constexpr static REAL_TYPE coordinate( int const index ) 
  { 
    if constexpr ( N==2 )
    {
      return ( -1.0 + index * 2.0 );
    }
    else if constexpr ( N==3 )
    {
      return ( -1.0 + index );
    }
    else if constexpr ( N==4 )
    {
      return ((index&2)-1) * (1 - ( ((index+1)&2)>>1 ) * (1.0-invSqrt5) );
      // if ( index == 0 ) return -1.0;
      // if ( index == 1 ) return -invSqrt5;
      // if ( index == 2 ) return  invSqrt5;
      // if ( index == 3 ) return  1.0;
    }
    else if constexpr (N==5)
    {
      constexpr REAL_TYPE rval[5] = { -1.0, -sqrt3div7, 0, sqrt3div7, 1.0 };
      return rval[index];
      // if ( index == 0 ) return -1.0;
      // if ( index == 1 ) return -sqrt3div7;
      // if ( index == 2 ) return  0;
      // if ( index == 3 ) return  sqrt3div7;
      // if ( index == 4 ) return  1.0;
    }
    return 0;
  }

  template< int INDEX >
  constexpr static REAL_TYPE coordinate()
  { 
    if constexpr ( N==2 )
    {
      return ( -1.0 + INDEX * 2.0 );
    }
    else if constexpr ( N==3 )
    {
      return ( -1.0 + INDEX );
    }
    else if constexpr ( N==4 )
    {
      if constexpr ( INDEX == 0 ) return -1.0;
      if constexpr ( INDEX == 1 ) return -invSqrt5;
      if constexpr ( INDEX == 2 ) return  invSqrt5;
      if constexpr ( INDEX == 3 ) return  1.0;
    }
    else if constexpr (N==5)
    {
      if constexpr ( INDEX == 0 ) return -1.0;
      if constexpr ( INDEX == 1 ) return -sqrt3div7;
      if constexpr ( INDEX == 2 ) return  0;
      if constexpr ( INDEX == 3 ) return  sqrt3div7;
      if constexpr ( INDEX == 4 ) return  1.0;
    }
    return 0;
  }

};


}