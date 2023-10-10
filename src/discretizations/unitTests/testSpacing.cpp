
#include "../spacing/Spacing.hpp"
#include "common/SequenceUtilities.hpp"
#include "common/ShivaMacros.hpp"


#include <gtest/gtest.h>
#include <cmath>

using namespace shiva;

constexpr bool check( double const a, double const b, double const tolerance )
{
  return ( a - b ) * ( a - b ) < tolerance * tolerance;
}


template< typename ... T >
struct ReferenceSolution;

template< typename REAL_TYPE >
struct ReferenceSolution< EqualSpacing<REAL_TYPE,2> >
{
  static constexpr REAL_TYPE coords[2] = { -1.0, 1.0 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< EqualSpacing<REAL_TYPE,3> >
{
  static constexpr REAL_TYPE coords[3] = { -1.0, 0.0, 1.0 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< EqualSpacing<REAL_TYPE,4> >
{
  static constexpr REAL_TYPE coords[4] = { -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< EqualSpacing<REAL_TYPE,5> >
{
  static constexpr REAL_TYPE coords[5] = { -1.0, -0.5, 0.0, 0.5, 1.0 };
};



template< typename REAL_TYPE >
struct ReferenceSolution< GaussLegendreSpacing<REAL_TYPE,2> >
{
  static constexpr REAL_TYPE coords[2] = { -0.57735026918962576451, 0.57735026918962576451 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< GaussLegendreSpacing<REAL_TYPE,3> >
{
  static constexpr REAL_TYPE coords[3] = { -0.77459666924148337704, 0.0, 0.77459666924148337704 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< GaussLegendreSpacing<REAL_TYPE,4> >
{
  static constexpr REAL_TYPE coords[4] = { -0.86113631159405257522, 
                                           -0.3399810435848562648, 
                                           0.3399810435848562648, 
                                           0.86113631159405257522 };
};


template< typename REAL_TYPE >
struct ReferenceSolution< GaussLobattoSpacing<REAL_TYPE,2> >
{
  static constexpr REAL_TYPE coords[2] = { -1.0, 1.0 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< GaussLobattoSpacing<REAL_TYPE,3> >
{
  static constexpr REAL_TYPE coords[3] = { -1.0, 0.0, 1.0 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< GaussLobattoSpacing<REAL_TYPE,4> >
{
  static constexpr REAL_TYPE coords[4] = { -1.0, -0.44721359549995793928, 0.44721359549995793928, 1.0 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< GaussLobattoSpacing<REAL_TYPE,5> >
{
  static constexpr REAL_TYPE coords[5] = { -1.0, 
                                           -0.6546536707079771438, 
                                           0.0, 
                                           0.6546536707079771438, 
                                           1.0 };
};



template< template< typename, int > typename SPACING, typename REAL_TYPE, int N >
void testSpacingRT()
{
  using SpacingType = SPACING< REAL_TYPE, N >;
  using Ref = ReferenceSolution< SpacingType >;

  constexpr REAL_TYPE tolerance = 1e-13;
  for ( int a = 0; a < N; ++a )
  {
    EXPECT_NEAR( SpacingType::coordinate( a ), Ref::coords[a], abs( Ref::coords[a] ) * tolerance );
  }
}

template< template< typename, int > typename SPACING, typename REAL_TYPE, std::size_t ... I, int N = sizeof...(I) >
static constexpr void testSpacingCT( std::index_sequence< I... > )
{
  using SpacingType = SPACING< REAL_TYPE, N >;
  using Ref = ReferenceSolution< SpacingType >;

  constexpr REAL_TYPE tolerance = 1e-13;
  forSequence< N >( [&] ( auto const a ) constexpr
  {
    static_assert( check( SpacingType::template coordinate< a >(), Ref::coords[a], tolerance ) );
  } );
}


TEST( testSpacing, testEqualSpacingRT )
{
  testSpacingRT< EqualSpacing, double, 2 >( );
  testSpacingRT< EqualSpacing, double, 3 >(  );
  testSpacingRT< EqualSpacing, double, 4 >( );
  testSpacingRT< EqualSpacing, double, 5 >(  );
}

TEST( testSpacing, testEqualSpacingCT )
{
  testSpacingCT< EqualSpacing, double >( std::make_index_sequence< 2 >{} );
  testSpacingCT< EqualSpacing, double >( std::make_index_sequence< 3 >{} );
  testSpacingCT< EqualSpacing, double >( std::make_index_sequence< 4 >{} );
  testSpacingCT< EqualSpacing, double >( std::make_index_sequence< 5 >{} );
}


TEST( testSpacing, testGaussLegendreSpacingRT )
{
  testSpacingRT< GaussLegendreSpacing, double, 2 >( );
  testSpacingRT< GaussLegendreSpacing, double, 3 >( );
  testSpacingRT< GaussLegendreSpacing, double, 4 >( );
}

TEST( testSpacing, testGaussLegendreSpacingCT )
{
  testSpacingCT< GaussLegendreSpacing, double >( std::make_index_sequence< 2 >{} );
  testSpacingCT< GaussLegendreSpacing, double >( std::make_index_sequence< 3 >{} );
  testSpacingCT< GaussLegendreSpacing, double >( std::make_index_sequence< 4 >{} );
}

TEST( testSpacing, testGaussLobattoSpacingRT )
{
  testSpacingRT< GaussLobattoSpacing, double, 2 >(  );
  testSpacingRT< GaussLobattoSpacing, double, 3 >(  );
  testSpacingRT< GaussLobattoSpacing, double, 4 >(   );
  testSpacingRT< GaussLobattoSpacing, double, 5 >(   );
}

TEST( testSpacing, testGaussLobattoSpacingCT )
{
  testSpacingCT< GaussLobattoSpacing, double >( std::make_index_sequence< 2 >{} );
  testSpacingCT< GaussLobattoSpacing, double >( std::make_index_sequence< 3 >{} );
  testSpacingCT< GaussLobattoSpacing, double >( std::make_index_sequence< 4 >{} );
  testSpacingCT< GaussLobattoSpacing, double >( std::make_index_sequence< 5 >{} );
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
