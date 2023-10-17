#pragma once

#include "common/SequenceUtilities.hpp"
#include "common/ShivaMacros.hpp"


#include <utility>

namespace shiva
{

namespace discretizations
{
namespace finiteElementMethod
{


namespace basis
{
template< typename REAL_TYPE, int ORDER, template< typename, int > typename SPACING_TYPE, bool USE_FOR_SEQUENCE = false >
class LagrangeBasis : public SPACING_TYPE< REAL_TYPE, ORDER + 1 >
{
public:
  /// The number of support points for the basis
  static inline constexpr int order = ORDER;
  static inline constexpr int numSupportPoints = ORDER + 1;

  template< int BF_INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE 
  value( REAL_TYPE const & coord )
  {
#if __cplusplus >= 202002L
    return executeSequence< numSupportPoints >( [&]< int ... a > () constexpr
    {
      return ( valueFactor< BF_INDEX, a >( coord ) * ... );
    } );
#else
    return executeSequence< numSupportPoints >( [&] ( auto const ... a ) constexpr
    {
      return ( valueFactor< BF_INDEX, decltype(a)::value >( coord ) * ... );
    } );
#endif
  }

  template< int BF_INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE 
  gradient( REAL_TYPE const & coord )
  {

#if __cplusplus >= 202002L
    if constexpr ( USE_FOR_SEQUENCE )
    {
      REAL_TYPE rval = 0.0;
      forSequence< numSupportPoints >( [&]< int a > () constexpr
      {
        double term = gradientFactor< BF_INDEX, a >();
        forSequence< numSupportPoints >( [&]< int b > () constexpr
        {
          term *= valueFactor< BF_INDEX, b, a >( coord );
        } );
        rval += term;
      } );
      return rval;
    }
    else
    {
      return executeSequence< numSupportPoints >( [&coord]< int ... a > () constexpr
      {
        auto func = [&coord]< int ... b > ( auto aa ) constexpr
        {
          constexpr int aVal = decltype(aa)::value;
          return gradientFactor< BF_INDEX, aVal >() * ( valueFactor< BF_INDEX, b, aVal >( coord ) * ... );
        };

        return ( executeSequence< numSupportPoints >( func, std::integral_constant< int, a >{} ) + ... );
      } );
    }
#else
    if constexpr ( USE_FOR_SEQUENCE )
    {
      REAL_TYPE rval = 0.0;
      forSequence< numSupportPoints >( [&] ( auto const ica ) constexpr
      {
        constexpr int a = decltype(ica)::value;
        double term = gradientFactor< BF_INDEX, a >();
        forSequence< numSupportPoints >( [&] ( auto const icb ) constexpr
        {
          constexpr int b = decltype(icb)::value;
          term *= valueFactor< BF_INDEX, b, a >( coord );
        } );
        rval += term;
      } );
      return rval;
    }
    else
    {
      return executeSequence< numSupportPoints >( [&coord] ( auto const ... a ) constexpr
      {
        auto func = [&coord] ( auto a, auto ... b ) constexpr
        {
          constexpr int aVal = decltype(a)::value;
          return gradientFactor< BF_INDEX, aVal >() * ( valueFactor< BF_INDEX, decltype(b)::value, aVal >( coord ) * ... );
        };

        return ( executeSequence< numSupportPoints >( func, a ) + ... );
      } );
    }
#endif
  }


private:
  template< int BF_INDEX, int FACTOR_INDEX, int DERIVATIVE_INDEX = -1 >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE 
  valueFactor( REAL_TYPE const & coord )
  {
    if constexpr ( BF_INDEX == FACTOR_INDEX || FACTOR_INDEX == DERIVATIVE_INDEX )
    {
      return 1.0;
    }
    else
    {
      constexpr REAL_TYPE coordinate_FI = SPACING_TYPE< REAL_TYPE, ORDER + 1 >::template coordinate< FACTOR_INDEX >();
      constexpr REAL_TYPE coordinate_BF = SPACING_TYPE< REAL_TYPE, ORDER + 1 >::template coordinate< BF_INDEX >();
      return ( coord - coordinate_FI ) / ( coordinate_BF - coordinate_FI );
    }
  }

  template< int BF_INDEX, int FACTOR_INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE 
  gradientFactor()
  {
    if constexpr ( BF_INDEX == FACTOR_INDEX )
    {
      return 0.0;
    }
    else
    {
      constexpr REAL_TYPE coordinate_FI = SPACING_TYPE< REAL_TYPE, ORDER + 1 >::template coordinate< FACTOR_INDEX >();
      constexpr REAL_TYPE coordinate_BF = SPACING_TYPE< REAL_TYPE, ORDER + 1 >::template coordinate< BF_INDEX >();
      return 1.0 / ( coordinate_BF - coordinate_FI );
    }
  }

}; // class LagrangeBasis

} // namespace basis
} // namespace finiteElementMethod
} // namespace discretizations
} // namespace shiva
