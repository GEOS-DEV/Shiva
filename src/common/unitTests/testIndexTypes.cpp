
#include "../IndexTypes.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;


void testLinearIndexTypeHelper()
{
  int * data = nullptr;
  pmpl::genericKernelWrapper( 10, data, [] SHIVA_HOST_DEVICE ( int * const data )
  {
    int i = 0;
    LinearIndex< int > a = 0;
    for ( a = 0, i = 0; a < 10; ++a, ++i )
    {
      data[i] = linearIndex( a );
    }
  } );
  for ( int i = 0; i < 10; ++i )
  {
    EXPECT_EQ( data[i], i );
  }
  pmpl::deallocateData(data);

}

TEST( testIndexTypes, testLinearIndexType )
{
  testLinearIndexTypeHelper();
}



void testMultiIndexManualLoopHelper()
{
  int * data = nullptr;
  pmpl::genericKernelWrapper( 8, data, [] SHIVA_HOST_DEVICE ( int * const data )
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
          data[4 * a + 2 * b + c] = linearIndex( index );
        }
      }
    }
  } );
    
  for ( int a = 0; a < 2; ++a )
  {
    for ( int b = 0; b < 2; ++b )
    {
      for ( int c = 0; c < 2; ++c )
      {
        EXPECT_EQ( data[4 * a + 2 * b + c], 4 * a + 2 * b + c );
      }
    }
  }
  pmpl::deallocateData(data);
}
TEST( testIndexTypes, testMultiIndexManualLoop )
{
  testMultiIndexManualLoopHelper();
}


void testMultiIndexForRangeHelper()
{
  int * data = nullptr;
  pmpl::genericKernelWrapper( 8, data, [] SHIVA_HOST_DEVICE ( int * const data )
  {
    MultiIndexRange< int, 2, 2, 2 > index{ { 0, 0, 0 } };

    forRange( index, [&] ( auto const & i )
    {
      data[4 * i.data[0] + 2 * i.data[1] + i.data[2]] = linearIndex( i );
    } );
  } );

  for ( int a = 0; a < 2; ++a )
  {
    for ( int b = 0; b < 2; ++b )
    {
      for ( int c = 0; c < 2; ++c )
      {
        EXPECT_EQ( data[4 * a + 2 * b + c], 4 * a + 2 * b + c );
      }
    }
  }
  pmpl::deallocateData(data);

}
TEST( testIndexTypes, testMultiIndexForRange )
{
  testMultiIndexForRangeHelper();
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
