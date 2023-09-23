#pragma once

#include <utility>

namespace shiva
{

namespace discretizations
{
namespace finiteElementMethod
{


namespace basis
{
template< typename REAL_TYPE, int ORDER, template<typename,int> typename SPACING_TYPE  >
class LagrangeBasis : public SPACING_TYPE< REAL_TYPE, ORDER+1 >
{
public:
  /// The number of support points for the basis
  constexpr static int order = ORDER;
  constexpr static int numSupportPoints = ORDER + 1;

  template< int BF_INDEX >
  constexpr static REAL_TYPE value( const REAL_TYPE coord )
  {
    return valueHelper<BF_INDEX>( coord, std::make_integer_sequence<int,numSupportPoints>{} );
  }

  template< int BF_INDEX >
  constexpr static REAL_TYPE gradient( const REAL_TYPE coord )
  {
    return gradientHelper<BF_INDEX>( coord, std::make_integer_sequence<int,numSupportPoints>{} );
  }


private:
  template< int BF_INDEX, int FACTOR_INDEX, int DERIVATIVE_INDEX=-1 >
  constexpr static REAL_TYPE valueFactor( const REAL_TYPE coord )
  {
    if constexpr ( BF_INDEX == FACTOR_INDEX || FACTOR_INDEX==DERIVATIVE_INDEX )
    {
      return 1.0;
    }
    else
    {
      return (                                                               coord - SPACING_TYPE< REAL_TYPE, ORDER+1 >::template coordinate<FACTOR_INDEX>() ) /
             ( SPACING_TYPE< REAL_TYPE, ORDER+1 >::template coordinate<BF_INDEX>() - SPACING_TYPE< REAL_TYPE, ORDER+1 >::template coordinate<FACTOR_INDEX>() );
    }
  }

  template< int BF_INDEX, int ... INDICES >
  constexpr static REAL_TYPE valueHelper( const REAL_TYPE coord,
                                         std::integer_sequence<int, INDICES...> )
  {
      return ( valueFactor<BF_INDEX,INDICES>( coord ) * ... );
  }

  template< int BF_INDEX, int FACTOR_INDEX>
  constexpr static REAL_TYPE gradientFactor()
  {
    if constexpr ( BF_INDEX == FACTOR_INDEX )
    {
      return 0.0;
    }
    else
    {
      return 1.0 / ( SPACING_TYPE< REAL_TYPE, ORDER+1 >::template coordinate<BF_INDEX>() - SPACING_TYPE< REAL_TYPE, ORDER+1 >::template coordinate<FACTOR_INDEX>() );
    }
  }

  template< int BF_INDEX, int DERIVATIVE_INDEX, int ... FACTOR_INDICES >
  constexpr static REAL_TYPE gradientChainRuleTerm( const REAL_TYPE coord,
                                                  std::integer_sequence<int, FACTOR_INDICES...> )
  {
      return gradientFactor<BF_INDEX,DERIVATIVE_INDEX>() * ( valueFactor<BF_INDEX,FACTOR_INDICES,DERIVATIVE_INDEX>( coord ) * ... );
  }

  template< int BF_INDEX, int ... DERIVATIVE_INDICES >
  constexpr static REAL_TYPE gradientHelper( const REAL_TYPE coord,
                                             std::integer_sequence<int, DERIVATIVE_INDICES...> )
  {
      return ( gradientChainRuleTerm<BF_INDEX,DERIVATIVE_INDICES>( coord, std::make_integer_sequence<int,numSupportPoints>{} ) + ... );
  }

}; // class LagrangeBasis

} // namespace basis
} // namespace finiteElementMethod
} // namespace discretizations
} // namespace shiva
