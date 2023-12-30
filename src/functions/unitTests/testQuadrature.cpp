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


#include "../quadrature/Quadrature.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace shiva;



template< template< typename, int > typename QUARATURE, typename REAL_TYPE, int N >
void testQuadratureRT( REAL_TYPE const (&referenceValues)[N][2] )
{
  using QuadratureType = QUARATURE< REAL_TYPE, N >;
  for ( int a = 0; a < N; ++a )
  {
    EXPECT_NEAR( QuadratureType::coordinate( a ), referenceValues[a][0], abs( referenceValues[a][0] ) * 1e-13 );
    EXPECT_NEAR( QuadratureType::weight( a ), referenceValues[a][1], abs( referenceValues[a][1] ) * 1e-13 );
  }
}

template< template< typename, int > typename QUARATURE, typename REAL_TYPE, std::size_t ... I, int N = sizeof...(I) >
void testQuadratureCT( REAL_TYPE const (&referenceValues)[N][2],
                       std::index_sequence< I... > )
{
  using QuadratureType = QUARATURE< REAL_TYPE, N >;
  REAL_TYPE const coords[N] = { ( QuadratureType::template coordinate< I >() )... };
  REAL_TYPE const weights[N] = { ( QuadratureType::template weight< I >() )... };

  for ( int a = 0; a < N; ++a )
  {
    EXPECT_NEAR( coords[a], referenceValues[a][0], abs( referenceValues[a][0] ) * 1e-13 );
    EXPECT_NEAR( weights[a], referenceValues[a][1], abs( referenceValues[a][1] ) * 1e-13 );
  }

}

TEST( testQuadrature, testGaussLegendreQuadrature2RT )
{
  testQuadratureRT< QuadratureGaussLegendre, double, 2 >( { { -1.0 / sqrt( 3.0 ), 1.0 },
                                                            {  1.0 / sqrt( 3.0 ), 1.0 } } );
}

TEST( testQuadrature, testGaussLegendreQuadrature3RT )
{
  testQuadratureRT< QuadratureGaussLegendre, double, 3 >( { { -sqrt( 0.6 ), 5.0 / 9.0 },
                                                            {        0.0, 8.0 / 9.0 },
                                                            {  sqrt( 0.6 ), 5.0 / 9.0 } } );
}

TEST( testQuadrature, testGaussLegendreQuadrature4RT )
{
  testQuadratureRT< QuadratureGaussLegendre, double, 4 >( { { -sqrt((15.0 + 2.0 * sqrt( 30.0 )) / 35.0 ), (18.0 - sqrt( 30.0 )) / 36.0 },
                                                            { -sqrt((15.0 - 2.0 * sqrt( 30.0 )) / 35.0 ), (18.0 + sqrt( 30.0 )) / 36.0 },
                                                            {  sqrt((15.0 - 2.0 * sqrt( 30.0 )) / 35.0 ), (18.0 + sqrt( 30.0 )) / 36.0 },
                                                            {  sqrt((15.0 + 2.0 * sqrt( 30.0 )) / 35.0 ), (18.0 - sqrt( 30.0 )) / 36.0 } } );
}

TEST( testQuadrature, testGaussLegendreQuadrature2CT )
{
  testQuadratureCT< QuadratureGaussLegendre, double >( { { -1.0 / sqrt( 3.0 ), 1.0 },
                                                         {  1.0 / sqrt( 3.0 ), 1.0 } },
                                                       std::make_index_sequence< 2 >{} );
}

TEST( testQuadrature, testGaussLegendreQuadrature3CT )
{
  testQuadratureCT< QuadratureGaussLegendre, double >( { { -sqrt( 0.6 ), 5.0 / 9.0 },
                                                         {        0.0, 8.0 / 9.0 },
                                                         {  sqrt( 0.6 ), 5.0 / 9.0 } },
                                                       std::make_index_sequence< 3 >{} );
}

TEST( testQuadrature, testGaussLegendreQuadrature4CT )
{
  testQuadratureCT< QuadratureGaussLegendre, double >( { { -sqrt((15.0 + 2.0 * sqrt( 30.0 )) / 35.0 ), (18.0 - sqrt( 30.0 )) / 36.0 },
                                                         { -sqrt((15.0 - 2.0 * sqrt( 30.0 )) / 35.0 ), (18.0 + sqrt( 30.0 )) / 36.0 },
                                                         {  sqrt((15.0 - 2.0 * sqrt( 30.0 )) / 35.0 ), (18.0 + sqrt( 30.0 )) / 36.0 },
                                                         {  sqrt((15.0 + 2.0 * sqrt( 30.0 )) / 35.0 ), (18.0 - sqrt( 30.0 )) / 36.0 } },
                                                       std::make_index_sequence< 4 >{} );
}


TEST( testQuadrature, testGaussLobattoQuadrature2RT )
{
  testQuadratureRT< QuadratureGaussLobatto, double, 2 >( { { -1.0, 1.0 },
                                                           {  1.0, 1.0 } } );
}

TEST( testQuadrature, testGaussLobattoQuadrature3RT )
{
  testQuadratureRT< QuadratureGaussLobatto, double, 3 >( { { -1.0, 1.0 / 3.0 },
                                                           {  0.0, 4.0 / 3.0 },
                                                           {  1.0, 1.0 / 3.0 } } );
}

TEST( testQuadrature, testGaussLobattoQuadrature4RT )
{
  testQuadratureRT< QuadratureGaussLobatto, double, 4 >( { {       -1.0, 1.0 / 6.0 },
                                                           { -sqrt( 0.2 ), 5.0 / 6.0 },
                                                           {  sqrt( 0.2 ), 5.0 / 6.0 },
                                                           {        1.0, 1.0 / 6.0 } } );
}

TEST( testQuadrature, testGaussLobattoQuadrature5RT )
{
  testQuadratureRT< QuadratureGaussLobatto, double, 5 >( { {           -1.0, 1.0 / 10.0 },
                                                           { -sqrt( 3.0 / 7.0 ), 49.0 / 90.0 },
                                                           {            0.0, 32.0 / 45.0 },
                                                           {  sqrt( 3.0 / 7.0 ), 49.0 / 90.0 },
                                                           {            1.0, 1.0 / 10.0 } } );
}

TEST( testQuadrature, testGaussLobattoQuadrature2CT )
{
  testQuadratureCT< QuadratureGaussLobatto, double >( { { -1.0, 1.0 },
                                                        {  1.0, 1.0 } },
                                                      std::make_index_sequence< 2 >{} );
}

TEST( testQuadrature, testGaussLobattoQuadrature3CT )
{
  testQuadratureCT< QuadratureGaussLobatto, double >( { { -1.0, 1.0 / 3.0 },
                                                        {  0.0, 4.0 / 3.0 },
                                                        {  1.0, 1.0 / 3.0 } },
                                                      std::make_index_sequence< 3 >{} );
}

TEST( testQuadrature, testGaussLobattoQuadrature4CT )
{
  testQuadratureCT< QuadratureGaussLobatto, double >( { {       -1.0, 1.0 / 6.0 },
                                                        { -sqrt( 0.2 ), 5.0 / 6.0 },
                                                        {  sqrt( 0.2 ), 5.0 / 6.0 },
                                                        {        1.0, 1.0 / 6.0 } },
                                                      std::make_index_sequence< 4 >{} );
}

TEST( testQuadrature, testGaussLobattoQuadrature5CT )
{
  testQuadratureCT< QuadratureGaussLobatto, double >( { {           -1.0, 1.0 / 10.0 },
                                                        { -sqrt( 3.0 / 7.0 ), 49.0 / 90.0 },
                                                        {            0.0, 32.0 / 45.0 },
                                                        {  sqrt( 3.0 / 7.0 ), 49.0 / 90.0 },
                                                        {            1.0, 1.0 / 10.0 } },
                                                      std::make_index_sequence< 5 >{} );
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
