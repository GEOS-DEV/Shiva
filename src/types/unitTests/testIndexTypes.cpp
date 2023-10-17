
#include "../IndexTypes.hpp"

#include <gtest/gtest.h>

using namespace shiva;

TEST( testIndexTypes, testLinearIndexType )
{
  int i = 0;
  LinearIndex< int > a = 0;
  for ( a = 0, i = 0; a < 10; ++a, ++i )
  {
    EXPECT_EQ( a, i );
    EXPECT_EQ( linearIndex( a ), i );
  }
}



TEST( testIndexTypes, testMultiIndexManualLoop )
{
  MultiIndexRange< int, 2, 2, 2 > index{ { 1, 0, 0 } };

  int & a = index.data[0];
  int & b = index.data[1];
  int & c = index.data[2];
  for ( a = 0; a < 2; ++a )
  {
    for ( b = 0; b < 2; ++b )
    {
      for ( c = 0; c < 2; ++c )
      {
        int li = linearIndex( index );
        EXPECT_EQ( li, 4 * a + 2 * b + c );
      }
    }
  }
}


// TEST( testIndexTypes, testMultiIndexForRange )
// {
//   MultiIndexRange< int, 2, 2, 2 > index{ { 1, 0, 0 } };

//   forRange( index = {0, 0, 0}, [] SHIVA_DEVICE ( auto const & i )
//   {
//     int li = linearIndex( i );
//     EXPECT_EQ( li, 4 * i.data[0] + 2 * i.data[1] + i.data[2] );
//   } );
// }

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
