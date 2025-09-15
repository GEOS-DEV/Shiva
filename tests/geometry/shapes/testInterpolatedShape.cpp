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


#include "../InterpolatedShape.hpp"
#include "functions/bases/LagrangeBasis.hpp"
#include "functions/spacing/Spacing.hpp"
#include "geometry/shapes/NCube.hpp"
#include "common/ShivaMacros.hpp"
#include "common/pmpl.hpp"



#include <gtest/gtest.h>
#include <cmath>

using namespace shiva;
using namespace shiva::geometry;
using namespace shiva::functions;

#include "testInterpolatedShapeSolutions.hpp"



template< typename TEST_INTERPOLATED_SHAPE_HELPER >
SHIVA_GLOBAL void compileTimeKernel()
{
  using InterpolatedShapeType = typename TEST_INTERPOLATED_SHAPE_HELPER::InterpolatedShapeType;
  constexpr int order = TEST_INTERPOLATED_SHAPE_HELPER::order;


  forSequence< order + 1 >( [] ( auto ica ) constexpr
  {
    constexpr int a = decltype(ica)::value;
    forSequence< order + 1 >( [] ( auto icb ) constexpr
    {
      constexpr int b = decltype(icb)::value;
      forSequence< order + 1 >( [] ( auto icc ) constexpr
      {
        constexpr int c = decltype(icc)::value;

        constexpr double coord[3] = { TEST_INTERPOLATED_SHAPE_HELPER::testCoords[0],
                                      TEST_INTERPOLATED_SHAPE_HELPER::testCoords[1],
                                      TEST_INTERPOLATED_SHAPE_HELPER::testCoords[2] };

        constexpr double value = InterpolatedShapeType::template value< a, b, c >( coord );
        constexpr CArrayNd< double, 3 > gradient = InterpolatedShapeType::template gradient< a, b, c >( coord );
        constexpr double tolerance = 1.0e-12;

        static_assert( pmpl::check( value, TEST_INTERPOLATED_SHAPE_HELPER::referenceValues[a][b][c], tolerance ) );
        static_assert( pmpl::check( gradient( 0 ), TEST_INTERPOLATED_SHAPE_HELPER::referenceGradients[a][b][c][0], tolerance ) );
        static_assert( pmpl::check( gradient( 1 ), TEST_INTERPOLATED_SHAPE_HELPER::referenceGradients[a][b][c][1], tolerance ) );
        static_assert( pmpl::check( gradient( 2 ), TEST_INTERPOLATED_SHAPE_HELPER::referenceGradients[a][b][c][2], tolerance ) );
      } );
    } );
  } );
}

template< typename TEST_INTERPOLATED_SHAPE_HELPER >
void testInterpolatedShapeAtCompileTime()
{
#if defined(SHIVA_USE_DEVICE)
  compileTimeKernel< TEST_INTERPOLATED_SHAPE_HELPER ><< < 1, 1 >> > ();
#else
  compileTimeKernel< TEST_INTERPOLATED_SHAPE_HELPER >();
#endif
}


template< typename TEST_INTERPOLATED_SHAPE_HELPER >
SHIVA_GLOBAL void runTimeKernel( double * const values,
                                 double * const gradients )
{
  using InterpolatedShapeType = typename TEST_INTERPOLATED_SHAPE_HELPER::InterpolatedShapeType;
  constexpr int order = TEST_INTERPOLATED_SHAPE_HELPER::order;
  constexpr int N = order + 1;


  double const coord[3] = { TEST_INTERPOLATED_SHAPE_HELPER::testCoords[0],
                            TEST_INTERPOLATED_SHAPE_HELPER::testCoords[1],
                            TEST_INTERPOLATED_SHAPE_HELPER::testCoords[2] };

  forSequence< N >( [&] ( auto const ica ) constexpr
  {
    constexpr int a = decltype(ica)::value;
    forSequence< N >( [&] ( auto const icb ) constexpr
    {
      constexpr int b = decltype(icb)::value;
      forSequence< N >( [&] ( auto const icc ) constexpr
      {
        constexpr int c = decltype(icc)::value;
        double const value = InterpolatedShapeType::template value< a, b, c >( coord );
        CArrayNd< double, 3 > const gradient = InterpolatedShapeType::template gradient< a, b, c >( coord );

        values[ a * N * N + b * N + c ] = value;
        gradients[ 3 * (a * N * N + b * N + c) + 0 ] = gradient( 0 );
        gradients[ 3 * (a * N * N + b * N + c) + 1 ] = gradient( 1 );
        gradients[ 3 * (a * N * N + b * N + c) + 2 ] = gradient( 2 );
      } );
    } );
  } );
}

template< typename TEST_INTERPOLATED_SHAPE_HELPER >
void testInterpolatedShapeAtRunTime()
{
  constexpr int order = TEST_INTERPOLATED_SHAPE_HELPER::order;
  constexpr int N = order + 1;

#if defined(SHIVA_USE_DEVICE)
  constexpr int bytes = N * N * N * sizeof(double);
  double * values;
  double * gradients;
  deviceMallocManaged( &values, bytes );
  deviceMallocManaged( &gradients, 3 * bytes );
  runTimeKernel< TEST_INTERPOLATED_SHAPE_HELPER ><< < 1, 1 >> > ( values, gradients );
  deviceDeviceSynchronize();
#else
  double values[N * N * N];
  double gradients[N * N * N * 3];
  runTimeKernel< TEST_INTERPOLATED_SHAPE_HELPER >( values, gradients );
#endif

  constexpr double tolerance = 1.0e-12;
  for ( int a = 0; a < N; ++a )
  {
    for ( int b = 0; b < N; ++b )
    {
      for ( int c = 0; c < N; ++c )
      {
        EXPECT_NEAR( values[ a * N * N + b * N + c ], TEST_INTERPOLATED_SHAPE_HELPER::referenceValues[a][b][c], fabs( TEST_INTERPOLATED_SHAPE_HELPER::referenceValues[a][b][c] * tolerance ) );
        EXPECT_NEAR( gradients[ 3 * (a * N * N + b * N + c) + 0 ], TEST_INTERPOLATED_SHAPE_HELPER::referenceGradients[a][b][c][0],
                     fabs( TEST_INTERPOLATED_SHAPE_HELPER::referenceGradients[a][b][c][0] * tolerance ) );
        EXPECT_NEAR( gradients[ 3 * (a * N * N + b * N + c) + 1 ], TEST_INTERPOLATED_SHAPE_HELPER::referenceGradients[a][b][c][1],
                     fabs( TEST_INTERPOLATED_SHAPE_HELPER::referenceGradients[a][b][c][1] * tolerance ) );
        EXPECT_NEAR( gradients[ 3 * (a * N * N + b * N + c) + 2 ], TEST_INTERPOLATED_SHAPE_HELPER::referenceGradients[a][b][c][2],
                     fabs( TEST_INTERPOLATED_SHAPE_HELPER::referenceGradients[a][b][c][2] * tolerance ) );
      }
    }
  }

}



TEST( testInterpolatedShape, testBasicUsage )
{

  InterpolatedShape< double,
                     Cube< double >,
                     LagrangeBasis< double, 1, GaussLobattoSpacing >,
                     LagrangeBasis< double, 1, GaussLobattoSpacing >,
                     LagrangeBasis< double, 1, GaussLobattoSpacing >
                     >::template value< 1, 1, 1 >( {0.5, 0, 1} );
}

TEST( testInterpolatedShape, testCubeLagrangeBasisGaussLobatto_O1 )
{
  using InterpolatedShapeType = InterpolatedShape< double,
                                                   Cube< double >,
                                                   LagrangeBasis< double, 1, GaussLobattoSpacing >,
                                                   LagrangeBasis< double, 1, GaussLobattoSpacing >,
                                                   LagrangeBasis< double, 1, GaussLobattoSpacing >
                                                   >;
  using TEST_INTERPOLATED_SHAPE_HELPER = TestInterpolatedShapeHelper< InterpolatedShapeType >;

  testInterpolatedShapeAtCompileTime< TEST_INTERPOLATED_SHAPE_HELPER >();
  testInterpolatedShapeAtRunTime< TEST_INTERPOLATED_SHAPE_HELPER >();
}

TEST( testInterpolatedShape, testCubeLagrangeBasisGaussLobatto_O3 )
{
  using InterpolatedShapeType = InterpolatedShape< double,
                                                   Cube< double >,
                                                   LagrangeBasis< double, 3, GaussLobattoSpacing >,
                                                   LagrangeBasis< double, 3, GaussLobattoSpacing >,
                                                   LagrangeBasis< double, 3, GaussLobattoSpacing >
                                                   >;
  using TEST_INTERPOLATED_SHAPE_HELPER = TestInterpolatedShapeHelper< InterpolatedShapeType >;

  testInterpolatedShapeAtCompileTime< TEST_INTERPOLATED_SHAPE_HELPER >();
  testInterpolatedShapeAtRunTime< TEST_INTERPOLATED_SHAPE_HELPER >();
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
