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


#include "../NSimplex.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::geometry;

template< typename NSIMPLEX >
struct NSimplexSolutions;

#define NSIMPLEX( NSIMPLEX_NAME, N, MIN, MAX, DIVISOR, NUM_VERTICES, NUM_EDGES, NUM_FACES, NUM_CELLS, NUM_HYPERFACES, MIN_COORD, MAX_COORD, LENGTH, VOLUME ) \
        using NSIMPLEX_NAME = NSimplex< double, N, MIN, MAX, DIVISOR >; \
        template<> \
        struct NSimplexSolutions< NSIMPLEX_NAME > \
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
NSIMPLEX( NSimplex_1_0_1_1,
          1, 0, 1, 1,
          2, 1, 0, 0, 2, 0.0, 1.0, 1.0, 1.0 )

NSIMPLEX( NSimplex_1_m1_1_1,
          1, -1, 1, 1,
          2, 1, 0, 0, 2, -1.0, 1.0, 2.0, 2.0 )

NSIMPLEX( NSimplex_1_m1_1_2,
          1, -1, 1, 2,
          2, 1, 0, 0, 2, -0.5, 0.5, 1.0, 1.0 )


//*****************************************************************************
NSIMPLEX( NSimplex_2_0_1_1,
          2, 0, 1, 1,
          3, 3, 1, 0, 3, 0.0, 1.0, 1.0, 0.5 )

NSIMPLEX( NSimplex_2_m1_1_1,
          2, -1, 1, 1,
          3, 3, 1, 0, 3, -1.0, 1.0, 2.0, 2.0 )

NSIMPLEX( NSimplex_2_m1_1_2,
          2, -1, 1, 2,
          3, 3, 1, 0, 3, -0.5, 0.5, 1.0, 0.5 )


//*****************************************************************************
NSIMPLEX( NSimplex_3_0_1_1,
          3, 0, 1, 1,
          4, 6, 4, 1, 4, 0.0, 1.0, 1.0, 1.0 / 6.0 )

NSIMPLEX( NSimplex_3_m1_1_1,
          3, -1, 1, 1,
          4, 6, 4, 1, 4, -1.0, 1.0, 2.0, 8.0 / 6.0 )

NSIMPLEX( NSimplex_3_m1_1_2,
          3, -1, 1, 2,
          4, 6, 4, 1, 4, -0.5, 0.5, 1.0, 1.0 / 6.0 )


//*****************************************************************************
NSIMPLEX( NSimplex_4_0_1_1,
          4, 0, 1, 1,
          5, 10, 10, 5, 5, 0.0, 1.0, 1.0, 1.0 / 24.0 )

NSIMPLEX( NSimplex_4_m1_1_1,
          4, -1, 1, 1,
          5, 10, 10, 5, 5, -1.0, 1.0, 2.0, 16.0 / 24.0 )

NSIMPLEX( NSimplex_4_m1_1_2,
          4, -1, 1, 2,
          5, 10, 10, 5, 5, -0.5, 0.5, 1.0, 1.0 / 24.0 )


template< typename NSIMPLEX >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
testNSimplexHelper()
{
  static_assert( NSIMPLEX::numDims() == NSimplexSolutions< NSIMPLEX >::nDims() );
  static_assert( NSIMPLEX::numVertices() == NSimplexSolutions< NSIMPLEX >::numVertices() );
  static_assert( NSIMPLEX::numEdges() == NSimplexSolutions< NSIMPLEX >::numEdges() );
  static_assert( NSIMPLEX::numFaces() == NSimplexSolutions< NSIMPLEX >::numFaces() );
  static_assert( NSIMPLEX::numCells() == NSimplexSolutions< NSIMPLEX >::numCells() );
  static_assert( NSIMPLEX::numHyperFaces() == NSimplexSolutions< NSIMPLEX >::numHyperFaces() );
  static_assert( pmpl::check( NSIMPLEX::minCoord(), NSimplexSolutions< NSIMPLEX >::minCoord(), 1.0e-12 ) );
  static_assert( pmpl::check( NSIMPLEX::maxCoord(), NSimplexSolutions< NSIMPLEX >::maxCoord(), 1.0e-12 ) );
  static_assert( pmpl::check( NSIMPLEX::length(), NSimplexSolutions< NSIMPLEX >::length(), 1.0e-12 ) );
  static_assert( pmpl::check( NSIMPLEX::volume(), NSimplexSolutions< NSIMPLEX >::volume(), 1.0e-12 ) );
}

TEST( testNCube, testLines )
{
  testNSimplexHelper< NSimplex_1_0_1_1 >( );
  testNSimplexHelper< NSimplex_1_m1_1_1 >( );
  testNSimplexHelper< NSimplex_1_m1_1_2 >( );
}

TEST( testNSimplex, testSquares )
{
  testNSimplexHelper< NSimplex_2_0_1_1 >( );
  testNSimplexHelper< NSimplex_2_m1_1_1 >( );
  testNSimplexHelper< NSimplex_2_m1_1_2 >( );
}

TEST( testNCube, testTetrahedra )
{
  testNSimplexHelper< NSimplex_3_0_1_1 >( );
  testNSimplexHelper< NSimplex_3_m1_1_1 >( );
  testNSimplexHelper< NSimplex_3_m1_1_2 >( );
}

TEST( testNCube, test5Cells )
{
  testNSimplexHelper< NSimplex_4_0_1_1 >( );
  testNSimplexHelper< NSimplex_4_m1_1_1 >( );
  testNSimplexHelper< NSimplex_4_m1_1_2 >( );
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
