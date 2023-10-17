#pragma once


#include "common/ShivaMacros.hpp"
namespace shiva
{

template< typename REAL_TYPE, int N >
struct EqualSpacing
{
  static inline constexpr int numPoints = N;

  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE interval() { return 2.0 / (numPoints - 1); }

  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE coordinate( int const index ) { return -1 + index * interval(); }

  template< int INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE coordinate() { return -1.0 + INDEX * interval(); }
};

template< typename REAL_TYPE, int N >
struct GaussLegendreSpacing
{
  static inline constexpr REAL_TYPE invSqrt3 = 0.57735026918962576450914878050195745565;  //1/sqrt(3)
  static inline constexpr REAL_TYPE sqrt3div5 = 0.77459666924148337703585307995647992217; //sqrt(3/5)

  static inline constexpr int numPoints = N; 

  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE coordinate( int const index )
  {
    if constexpr ( N == 2 )
    {
      return invSqrt3 * ( -1.0 + index * 2 );
    }
    else if constexpr ( N == 3 )
    {
      return sqrt3div5 * ( -1.0 + index );
    }
    else if constexpr ( N == 4 )
    {
      constexpr REAL_TYPE c4[2] = { 0.8611363115940525752239464888928095051,         //sqrt((15+2*sqrt(30))/35)
                                    0.3399810435848562648026657591032446872 };       //sqrt((15-2*sqrt(30))/35)
      return c4[ ( ((index + 1) & 2) >> 1) ] * ((index & 2) - 1);
    }
    return 0;
  }

  template< int INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE coordinate()
  {
    if constexpr ( N == 2 )
    {
      return invSqrt3 * ( -1.0 + INDEX * 2.0 );
    }
    else if constexpr ( N == 3 )
    {
      return sqrt3div5 * ( -1.0 + INDEX );
    }
    else if constexpr ( N == 4 )
    {
      constexpr REAL_TYPE c4[2] = { 0.8611363115940525752239464888928095051,         //sqrt((15+2*sqrt(30))/35)
                                    0.3399810435848562648026657591032446872 };       //sqrt((15-2*sqrt(30))/35)
      if constexpr ( INDEX == 0 ) return -c4[0];
      if constexpr ( INDEX == 1 ) return -c4[1];
      if constexpr ( INDEX == 2 ) return c4[1];
      if constexpr ( INDEX == 3 ) return c4[0];
    }
    return 0;

  }
};


template< typename REAL_TYPE, int N >
struct GaussLobattoSpacing
{
  static inline constexpr REAL_TYPE invSqrt5 = 0.44721359549995793928183473374625524709;
  static inline constexpr REAL_TYPE sqrt3div7 = 0.65465367070797714379829245624685835557;

  static inline constexpr int numPoints = N;

  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE coordinate( int const index )
  {
    if constexpr ( N == 2 )
    {
      return ( -1.0 + index * 2.0 );
    }
    else if constexpr ( N == 3 )
    {
      return ( -1.0 + index );
    }
    else if constexpr ( N == 4 )
    {
      return ((index & 2) - 1) * (1 - ( ((index + 1) & 2) >> 1 ) * (1.0 - invSqrt5) );
      // if ( index == 0 ) return -1.0;
      // if ( index == 1 ) return -invSqrt5;
      // if ( index == 2 ) return  invSqrt5;
      // if ( index == 3 ) return  1.0;
    }
    else if constexpr ( N == 5 )
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
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE coordinate()
  {
    if constexpr ( N == 2 )
    {
      return ( -1.0 + INDEX * 2.0 );
    }
    else if constexpr ( N == 3 )
    {
      return ( -1.0 + INDEX );
    }
    else if constexpr ( N == 4 )
    {
      if constexpr ( INDEX == 0 ) return -1.0;
      if constexpr ( INDEX == 1 ) return -invSqrt5;
      if constexpr ( INDEX == 2 ) return invSqrt5;
      if constexpr ( INDEX == 3 ) return 1.0;
    }
    else if constexpr ( N == 5 )
    {
      if constexpr ( INDEX == 0 ) return -1.0;
      if constexpr ( INDEX == 1 ) return -sqrt3div7;
      if constexpr ( INDEX == 2 ) return 0;
      if constexpr ( INDEX == 3 ) return sqrt3div7;
      if constexpr ( INDEX == 4 ) return 1.0;
    }
    else if constexpr ( N == 6 )
    {
      if constexpr ( INDEX == 0 ) return -1.0;
      if constexpr ( INDEX == 1 ) return -0.7650553239294648;
      if constexpr ( INDEX == 2 ) return -0.2852315164806455;
      if constexpr ( INDEX == 3 ) return 0.2852315164806455;
      if constexpr ( INDEX == 4 ) return 0.7650553239294648;
      if constexpr ( INDEX == 5 ) return 1.0;
    }
    else if constexpr ( N == 7 )
    {
      if constexpr ( INDEX == 0 ) return -1.0;
      if constexpr ( INDEX == 1 ) return -0.83022389627857;
      if constexpr ( INDEX == 2 ) return -0.4688487934707;
      if constexpr ( INDEX == 3 ) return 0.0;
      if constexpr ( INDEX == 4 ) return 0.4688487934707;
      if constexpr ( INDEX == 5 ) return 0.83022389627857;
      if constexpr ( INDEX == 6 ) return 1.0;
    }
    else if constexpr ( N == 8 )
    {
      if constexpr ( INDEX == 0 ) return -1.00000000000000;
      if constexpr ( INDEX == 1 ) return -0.87174014850961;
      if constexpr ( INDEX == 2 ) return -0.59170018143314;
      if constexpr ( INDEX == 3 ) return -0.20929921790248;
      if constexpr ( INDEX == 4 ) return 0.20929921790248;
      if constexpr ( INDEX == 5 ) return 0.59170018143314;
      if constexpr ( INDEX == 6 ) return 0.87174014850961;
      if constexpr ( INDEX == 7 ) return 1.00000000000000;
    }
    else if constexpr ( N == 9 )
    {
      if constexpr ( INDEX == 0 ) return -1.;
      if constexpr ( INDEX == 1 ) return -0.89975799541146;
      if constexpr ( INDEX == 2 ) return -0.67718627951074;
      if constexpr ( INDEX == 3 ) return -0.36311746382618;
      if constexpr ( INDEX == 4 ) return 0.0;
      if constexpr ( INDEX == 5 ) return 0.36311746382618;
      if constexpr ( INDEX == 6 ) return 0.67718627951074;
      if constexpr ( INDEX == 7 ) return 0.89975799541146;
      if constexpr ( INDEX == 8 ) return 1.;
    }
    return 0;
  }

};


} // namespace shiva
