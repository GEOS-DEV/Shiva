#pragma once

#include "common/SequenceUtilities.hpp"


#include <utility>

namespace shiva
{

namespace discretizations
{
namespace finiteElementMethod
{


namespace basis
{
template< typename REAL_TYPE, int ORDER, template< typename, int > typename SPACING_TYPE >
class LagrangeBasis : public SPACING_TYPE< REAL_TYPE, ORDER+1 >
{
public:
  /// The number of support points for the basis
  constexpr static int order = ORDER;
  constexpr static int numSupportPoints = ORDER + 1;

  template< int BF_INDEX >
  constexpr static REAL_TYPE value( REAL_TYPE const & coord )
  {
    return executeSequence<numSupportPoints>( [&]< int ... a>() constexpr
    {
      return ( valueFactor< BF_INDEX, a >( coord ) * ... );
    } );
  }

  template< int BF_INDEX >
  constexpr static REAL_TYPE gradient( REAL_TYPE const & coord )
  {
#if 1
    return executeSequence<numSupportPoints>( [&coord]< int ... a >() constexpr
    {
      return 
      ( executeSequence<numSupportPoints>( [&coord, aa=std::integral_constant<int,a>{} ]< int ... b >() constexpr { return gradientFactor< BF_INDEX, aa >() * ( valueFactor< BF_INDEX, b, aa >( coord ) * ... ); } 
                                         ) + ... );
//      return ( gradientChainRuleTerm< BF_INDEX, a >( coord, std::make_integer_sequence< int, numSupportPoints >{} ) + ... );
    } );
#else
    REAL_TYPE rval = 0.0;
    forSequence<numSupportPoints>( 
    [&]( auto const a ) constexpr
    {
      double term = gradientFactor<BF_INDEX, a >();
      forSequence<numSupportPoints>( 
      [&]( auto const b ) constexpr
      {
        term *= valueFactor<BF_INDEX, b, a>( coord );
      });
      rval += term;
    } );
    return rval;
#endif
  }


private:
  template< int BF_INDEX, int FACTOR_INDEX, int DERIVATIVE_INDEX=-1 >
  constexpr static REAL_TYPE valueFactor( REAL_TYPE const & coord )
  {
    if constexpr ( BF_INDEX == FACTOR_INDEX || FACTOR_INDEX==DERIVATIVE_INDEX )
    {
      return 1.0;
    }
    else
    {
      return (                                                               coord - SPACING_TYPE< REAL_TYPE, ORDER+1 >::template coordinate< FACTOR_INDEX >() ) /
             ( SPACING_TYPE< REAL_TYPE, ORDER+1 >::template coordinate< BF_INDEX >() - SPACING_TYPE< REAL_TYPE, ORDER+1 >::template coordinate< FACTOR_INDEX >() );
    }
  }

  template< int BF_INDEX, int FACTOR_INDEX >
  constexpr static REAL_TYPE gradientFactor()
  {
    if constexpr ( BF_INDEX == FACTOR_INDEX )
    {
      return 0.0;
    }
    else
    {
      return 1.0 / ( SPACING_TYPE< REAL_TYPE, ORDER+1 >::template coordinate< BF_INDEX >() - SPACING_TYPE< REAL_TYPE, ORDER+1 >::template coordinate< FACTOR_INDEX >() );
    }
  }

  template< int BF_INDEX, int DERIVATIVE_INDEX, int ... FACTOR_INDICES >
  constexpr static REAL_TYPE gradientChainRuleTerm( REAL_TYPE const & coord,
                                                    std::integer_sequence< int, FACTOR_INDICES... > )
  {
    return gradientFactor< BF_INDEX, DERIVATIVE_INDEX >() * ( valueFactor< BF_INDEX, FACTOR_INDICES, DERIVATIVE_INDEX >( coord ) * ... );
  }

}; // class LagrangeBasis

} // namespace basis
} // namespace finiteElementMethod
} // namespace discretizations
} // namespace shiva
