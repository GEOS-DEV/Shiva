
#include "../spacing/Spacing.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace shiva;



template< template< typename, int > typename SPACING, typename REAL_TYPE, int N >
void testSpacingRT( REAL_TYPE const (&coords)[N] )
{
  using SpacingType = SPACING< REAL_TYPE, N >;
  for ( int a = 0; a < N; ++a )
  {
    EXPECT_NEAR( SpacingType::coordinate( a ), coords[a], abs( coords[a] ) * 1e-13 );
  }
}

template< template< typename, int > typename SPACING, typename REAL_TYPE, std::size_t ... I, int N = sizeof...(I) >
void testSpacingCT( REAL_TYPE const (&referenceCoords)[N],
                    std::index_sequence< I... > )
{
  using SpacingType = SPACING< REAL_TYPE, N >;
  REAL_TYPE const coords[N] = { ( SpacingType::template coordinate< I >() )... };

  for ( int a = 0; a < N; ++a )
  {
    EXPECT_NEAR( coords[a], referenceCoords[a], abs( referenceCoords[a] ) * 1e-13 );
  }

}


TEST( testSpacing, testEqualSpacingRT )
{
  testSpacingRT< EqualSpacing, double, 2 >( { -1.0, 1.0 } );
  testSpacingRT< EqualSpacing, double, 3 >( { -1.0, 0.0, 1.0 } );
  testSpacingRT< EqualSpacing, double, 4 >( { -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0 } );
  testSpacingRT< EqualSpacing, double, 5 >( { -1.0, -0.5, 0.0, 0.5, 1.0 } );
}

TEST( testSpacing, testEqualSpacingCT )
{
  testSpacingCT< EqualSpacing, double >( { -1.0, 1.0 }, std::make_index_sequence< 2 >{} );
  testSpacingCT< EqualSpacing, double >( { -1.0, 0.0, 1.0 }, std::make_index_sequence< 3 >{} );
  testSpacingCT< EqualSpacing, double >( { -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0 }, std::make_index_sequence< 4 >{} );
  testSpacingCT< EqualSpacing, double >( { -1.0, -0.5, 0.0, 0.5, 1.0 }, std::make_index_sequence< 5 >{} );
}


TEST( testSpacing, testGaussLegendreSpacingRT )
{
  testSpacingRT< GaussLegendreSpacing, double, 2 >( { -1.0 / sqrt( 3.0 ), 1.0 / sqrt( 3.0 ) } );
  testSpacingRT< GaussLegendreSpacing, double, 3 >( { -sqrt( 0.6 ), 0.0, sqrt( 0.6 ) } );
  testSpacingRT< GaussLegendreSpacing, double, 4 >( { -sqrt((15.0 + 2.0 * sqrt( 30.0 )) / 35.0 ),
                                                      -sqrt((15.0 - 2.0 * sqrt( 30.0 )) / 35.0 ),
                                                      sqrt((15.0 - 2.0 * sqrt( 30.0 )) / 35.0 ),
                                                      sqrt((15.0 + 2.0 * sqrt( 30.0 )) / 35.0 ) } );
}

TEST( testSpacing, testGaussLegendreSpacingCT )
{
  testSpacingCT< GaussLegendreSpacing, double >( { -1.0 / sqrt( 3.0 ), 1.0 / sqrt( 3.0 ) }, std::make_index_sequence< 2 >{} );
  testSpacingCT< GaussLegendreSpacing, double >( { -sqrt( 0.6 ), 0.0, sqrt( 0.6 ) }, std::make_index_sequence< 3 >{} );
  testSpacingCT< GaussLegendreSpacing, double >( { -sqrt((15.0 + 2.0 * sqrt( 30.0 )) / 35.0 ),
                                                   -sqrt((15.0 - 2.0 * sqrt( 30.0 )) / 35.0 ),
                                                   sqrt((15.0 - 2.0 * sqrt( 30.0 )) / 35.0 ),
                                                   sqrt((15.0 + 2.0 * sqrt( 30.0 )) / 35.0 ) }, std::make_index_sequence< 4 >{} );
}

TEST( testSpacing, testGaussLobattoSpacingRT )
{
  testSpacingRT< GaussLobattoSpacing, double, 2 >( { -1.0, 1.0 } );
  testSpacingRT< GaussLobattoSpacing, double, 3 >( { -1.0, 0.0, 1.0 } );
  testSpacingRT< GaussLobattoSpacing, double, 4 >( { -1.0, -sqrt( 0.2 ), sqrt( 0.2 ), 1.0 } );
  testSpacingRT< GaussLobattoSpacing, double, 5 >( { -1.0, -sqrt( 3.0 / 7.0 ), 0.0, sqrt( 3.0 / 7.0 ), 1.0 } );
}

TEST( testSpacing, testGaussLobattoSpacingCT )
{
  testSpacingCT< GaussLobattoSpacing, double >( { -1.0, 1.0 }, std::make_index_sequence< 2 >{} );
  testSpacingCT< GaussLobattoSpacing, double >( { -1.0, 0.0, 1.0 }, std::make_index_sequence< 3 >{} );
  testSpacingCT< GaussLobattoSpacing, double >( { -1.0, -sqrt( 0.2 ), sqrt( 0.2 ), 1.0 }, std::make_index_sequence< 4 >{} );
  testSpacingCT< GaussLobattoSpacing, double >( { -1.0, -sqrt( 3.0 / 7.0 ), 0.0, sqrt( 3.0 / 7.0 ), 1.0 }, std::make_index_sequence< 5 >{} );
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
