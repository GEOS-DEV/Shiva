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


#include "../NCube.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::geometry;

template< typename NCUBE >
struct NCubeSolutions;

#define NCUBE( NCUBE_NAME, N, MIN, MAX, DIVISOR, NUM_VERTICES, NUM_EDGES, NUM_FACES, NUM_CELLS, NUM_HYPERFACES, MIN_COORD, MAX_COORD, LENGTH, VOLUME ) \
        using NCUBE_NAME = NCube< double, N, MIN, MAX, DIVISOR >; \
        template<> \
        struct NCubeSolutions< NCUBE_NAME > \
        { \
          static constexpr int nDims() {return N;} \
          static constexpr int numVertices() { return NUM_VERTICES;} \
          static constexpr int numEdges() { return NUM_EDGES;} \
          static constexpr int numFaces() { return NUM_FACES;} \
          static constexpr int numCells() { return NUM_CELLS;} \
          static constexpr int numHyperFaces() { return NUM_HYPERFACES;} \
          static constexpr double minCoord() { return MIN_COORD;} \
          static constexpr double maxCoord() { return MAX_COORD;} \
          static constexpr double length() { return LENGTH;} \
          static constexpr double volume() { return VOLUME;} \
        };

//*****************************************************************************
NCUBE( NCube_1_0_1_1,
       1, 0, 1, 1,
       2, 1, 0, 0, 2, 0.0, 1.0, 1.0, 1.0 )

NCUBE( NCube_1_m1_1_2,
       1, -1, 1, 2,
       2, 1, 0, 0, 2, -0.5, 0.5, 1.0, 1.0 )

NCUBE( NCube_1_m1_1_1,
       1, -1, 1, 1,
       2, 1, 0, 0, 2, -1.0, 1.0, 2.0, 2.0 )


//*****************************************************************************
NCUBE( NCube_2_0_1_1,
       2, 0, 1, 1,
       4, 4, 1, 0, 4, 0.0, 1.0, 1.0, 1.0 )

NCUBE( NCube_2_m1_1_2,
       2, -1, 1, 2,
       4, 4, 1, 0, 4, -0.5, 0.5, 1.0, 1.0 )

NCUBE( NCube_2_m1_1_1,
       2, -1, 1, 1,
       4, 4, 1, 0, 4, -1.0, 1.0, 2.0, 4.0 )


//*****************************************************************************
NCUBE( NCube_3_0_1_1,
       3, 0, 1, 1,
       8, 12, 6, 1, 6, 0.0, 1.0, 1.0, 1.0 )

NCUBE( NCube_3_m1_1_2,
       3, -1, 1, 2,
       8, 12, 6, 1, 6, -0.5, 0.5, 1.0, 1.0 )

NCUBE( NCube_3_m1_1_1,
       3, -1, 1, 1,
       8, 12, 6, 1, 6, -1.0, 1.0, 2.0, 8.0 )


//*****************************************************************************
NCUBE( NCube_4_0_1_1,
       4, 0, 1, 1,
       16, 32, 24, 8, 8, 0.0, 1.0, 1.0, 1.0 )

NCUBE( NCube_4_m1_1_2,
       4, -1, 1, 2,
       16, 32, 24, 8, 8, -0.5, 0.5, 1.0, 1.0 )

NCUBE( NCube_4_m1_1_1,
       4, -1, 1, 1,
       16, 32, 24, 8, 8, -1.0, 1.0, 2.0, 16.0 )


template< typename NCUBE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
testNCubeHelper()
{
  static_assert( NCUBE::numDims() == NCubeSolutions< NCUBE >::nDims() );
  static_assert( NCUBE::numVertices() == NCubeSolutions< NCUBE >::numVertices() );
  static_assert( NCUBE::numEdges() == NCubeSolutions< NCUBE >::numEdges() );
  static_assert( NCUBE::numFaces() == NCubeSolutions< NCUBE >::numFaces() );
  static_assert( NCUBE::numCells() == NCubeSolutions< NCUBE >::numCells() );
  static_assert( NCUBE::numHyperFaces() == NCubeSolutions< NCUBE >::numHyperFaces() );
  static_assert( pmpl::check( NCUBE::minCoord(), NCubeSolutions< NCUBE >::minCoord(), 1.0e-12 ) );
  static_assert( pmpl::check( NCUBE::maxCoord(), NCubeSolutions< NCUBE >::maxCoord(), 1.0e-12 ) );
  static_assert( pmpl::check( NCUBE::length(), NCubeSolutions< NCUBE >::length(), 1.0e-12 ) );
  static_assert( pmpl::check( NCUBE::volume(), NCubeSolutions< NCUBE >::volume(), 1.0e-12 ) );
}

TEST( testNCube, testLines )
{
  testNCubeHelper< NCube_1_0_1_1 >( );
  testNCubeHelper< NCube_1_m1_1_1 >( );
  testNCubeHelper< NCube_1_m1_1_2 >( );
}

TEST( testNCube, testSquares )
{
  testNCubeHelper< NCube_2_0_1_1 >( );
  testNCubeHelper< NCube_2_m1_1_1 >( );
  testNCubeHelper< NCube_2_m1_1_2 >( );
}

TEST( testNCube, testCubes )
{
  testNCubeHelper< NCube_3_0_1_1 >( );
  testNCubeHelper< NCube_3_m1_1_1 >( );
  testNCubeHelper< NCube_3_m1_1_2 >( );
}

TEST( testNCube, testTesseracts )
{
  testNCubeHelper< NCube_4_0_1_1 >( );
  testNCubeHelper< NCube_4_m1_1_1 >( );
  testNCubeHelper< NCube_4_m1_1_2 >( );
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
