/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2023  Lawrence Livermore National Security LLC
 * Copyright (c) 2023  TotalEnergies
 * Copyright (c) 2023- Shiva Contributors
 * All rights reserved
 *
 * See Shiva/LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */


#include "../IndexTypes.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;


void testLinearIndexTypeHelper()
{
  const int N = 10;
  int * data = new int[N];

  pmpl::genericKernelWrapper( N, data, [] SHIVA_HOST_DEVICE ( int * const kdata )
  {
    int i = 0;
    LinearIndex< int > a = 0;
    for ( a = 0, i = 0; a < N; ++a, ++i )
    {
      kdata[i] = linearIndex( a );
    }
  } );
  for ( int i = 0; i < N; ++i )
  {
    EXPECT_EQ( data[i], i );
  }
  delete[] data;
}

TEST( testIndexTypes, testLinearIndexType )
{
  testLinearIndexTypeHelper();
}



void testMultiIndexManualLoopHelper()
{
  int * data = new int[8];
  pmpl::genericKernelWrapper( 8, data, [] SHIVA_HOST_DEVICE ( int * const kdata )
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
          kdata[4 * a + 2 * b + c] = linearIndex( index );
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
  delete[] data;
}
TEST( testIndexTypes, testMultiIndexManualLoop )
{
  testMultiIndexManualLoopHelper();
}


void testMultiIndexForRangeHelper()
{
  int * data = new int[8];
  pmpl::genericKernelWrapper( 8, data, [] SHIVA_HOST_DEVICE ( int * const kdata )
  {
    MultiIndexRange< int, 2, 2, 2 > index{ { 0, 0, 0 } };

    forRange( index, [&] ( auto const & i )
    {
      kdata[4 * i.data[0] + 2 * i.data[1] + i.data[2]] = linearIndex( i );
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
  delete[] data;
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
