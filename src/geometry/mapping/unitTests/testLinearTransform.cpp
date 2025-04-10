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


#include "geometry/shapes/NCube.hpp"
#include "../LinearTransform.hpp"
#include "../JacobianTransforms.hpp"
#include "functions/bases/LagrangeBasis.hpp"
#include "functions/spacing/Spacing.hpp"
#include "geometry/shapes/InterpolatedShape.hpp"


#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::geometry;
using namespace shiva::geometry::utilities;
using namespace shiva::functions;


constexpr SHIVA_DEVICE double Xref[8][3] =
{ { -1.31, -1.15, -1.23 },
  {  1.38, -1.22, -1.17 },
  { -1.31, 1.12, -1.31 },
  {  1.32, 1.17, -1.37 },
  { -1.23, -1.25, 1.27 },
  {  1.27, -1.34, 1.24 },
  { -1.29, 1.28, 1.41 },
  {  1.39, 1.24, 1.36 } };

double Xreference( int const a, int const i )
{
  static constexpr double X[8][3] =
  { { -1.31, -1.15, -1.23 },
    {  1.38, -1.22, -1.17 },
    { -1.31, 1.12, -1.31 },
    {  1.32, 1.17, -1.37 },
    { -1.23, -1.25, 1.27 },
    {  1.27, -1.34, 1.24 },
    { -1.29, 1.28, 1.41 },
    {  1.39, 1.24, 1.36 } };

  return X[a][i];
}

constexpr SHIVA_DEVICE double qCoords[8][3] =
{ { -0.9, -0.5, -0.6 },
  { -0.9, -0.5, 0.6 },
  { -0.9, 0.5, -0.6 },
  { -0.9, 0.5, 0.6 },
  {  0.9, -0.5, -0.6 },
  {  0.9, -0.5, 0.6 },
  {  0.9, 0.5, -0.6 },
  {  0.9, 0.5, 0.6 } };


constexpr double Jref[8][3][3] =
{ { { 1.3245, -6.3e-3, 2.925e-2 }, { -2.375e-2, 1.16365, -1.84375e-2 }, {   8.5e-3, -2.05e-2, 1.275875 } },
  { { 1.2855, -2.07e-2, 2.925e-2 }, {   -3.5e-2, 1.2406, -1.84375e-2 }, {  -1.1e-2, 4.7e-2, 1.275875 } },
  { { 1.3215, -6.3e-3, 1.725e-2 }, {   2.75e-3, 1.16365, 4.56875e-2 }, { -1.65e-2, -2.05e-2, 1.332125 } },
  { { 1.3185, -2.07e-2, 1.725e-2 }, {   -1.9e-2, 1.2406, 4.56875e-2 }, {  -2.1e-2, 4.7e-2, 1.332125 } },
  { { 1.3245, -1.17e-2, -2.925e-2 }, { -2.375e-2, 1.21135, -3.53125e-2 }, {   8.5e-3, -6.55e-2, 1.246625 } },
  { { 1.2855, 3.87e-2, -2.925e-2 }, {   -3.5e-2, 1.2694, -3.53125e-2 }, {  -1.1e-2, 2.9e-2, 1.246625 } },
  { { 1.3215, -1.17e-2, 1.275e-2 }, {   2.75e-3, 1.21135, 1.30625e-2 }, { -1.65e-2, -6.55e-2, 1.325375 } },
  { { 1.3185, 3.87e-2, 1.275e-2 }, {   -1.9e-2, 1.2694, 1.30625e-2 }, {  -2.1e-2, 2.9e-2, 1.325375 } } };


constexpr double invJref[8][3][3] =
{ { { 7.5518050317544e-1, 3.7845098812477e-3, -1.725815837519e-2 }, { 1.5337360823102e-2, 8.5966062432681e-1, 1.2071225595728e-2 }, { -4.784652399426e-3, 1.3787302411842e-2, 7.8408476886129e-1 } },
  { { 7.7812816372463e-1, 1.3651790001633e-2, -1.7641652913326e-2 }, { 2.2040308761543e-2, 8.0600721432107e-1, 1.1142219247787e-2 }, { 5.8967495163542e-3, -2.957356275738e-2, 7.8321329088924e-1 } },
  { { 7.565836884692e-1, 3.9211774440993e-3, -9.931663635598e-3 }, { -2.1546321329131e-3, 8.5883484385508e-1, -2.9427305638987e-2 }, { 9.338058291089e-3, 1.3265131820855e-2, 7.501044329807e-1 } },
  { { 7.5846203705445e-1, 1.3044334658016e-2, -1.0268881057617e-2 }, { 1.1190189368391e-2, 8.0730273673882e-1, -2.783272932447e-2 }, { 1.1561800790338e-2, -2.8277599773975e-2, 7.5150041608411e-1 } },
  { { 7.5503486460179e-1, 8.2631892175591e-3, 1.7949715158005e-2 }, { 1.4675786248095e-2, 8.2695222317457e-1, 2.3768989975822e-2 }, { -4.3770438984178e-3, 4.3393268632977e-2, 8.0329232629265e-1 } },
  { { 7.7740146085247e-1, -2.410163122351e-2, 1.7557728970103e-2 }, { 2.1611414506272e-2, 7.8659427371398e-1, 2.2788524347605e-2 }, { 6.3569117005477e-3, -1.8511061370632e-2, 8.0179064900238e-1 } },
  { { 7.5660974382485e-1, 6.9105787290405e-3, -7.346632966832e-3 }, { -1.8182541770253e-3, 8.2506893149255e-1, -8.1141414142144e-3 }, { 9.3294087518739e-3, 4.0860918277311e-2, 7.5401098126448e-1 } },
  { { 7.5799440223966e-1, -2.2947438633317e-2, -7.0656815704294e-3 }, { 1.1224373599541e-2, 7.8761135930028e-1, -7.8704397957213e-3 },
    { 1.1764501075278e-2, -1.7597001324914e-2, 7.5456369966319e-1 } } };

constexpr double detJref[8] = { 1.9654823830313,
                                2.035290793125,
                                2.0500889374688,
                                2.17609699875,
                                1.9969756307812,
                                2.0369010315,
                                2.1230873817188,
                                2.219082631125 };

template< typename REAL_TYPE >
SHIVA_HOST_DEVICE auto makeLinearTransform( REAL_TYPE const (&X)[8][3] )
{
  LinearTransform< REAL_TYPE,
                   InterpolatedShape< double,
                                      Cube< double >,
                                      LagrangeBasis< double, 1, EqualSpacing >,
                                      LagrangeBasis< double, 1, EqualSpacing >,
                                      LagrangeBasis< double, 1, EqualSpacing > > > cell;

//  typename decltype(cell)::SupportIndexType index;

//  auto & transformData = cell.getData();

  cell.setData( X );

  // forRange( index = {0, 0, 0}, [&transformData, &X] ( auto const & i )
  // {
  //   int const a = i.data[0];
  //   int const b = i.data[1];
  //   int const c = i.data[2];

  //   for ( int j = 0; j < 3; ++j )
  //   {
  //     transformData( a, b, c, j ) = X[ a + 2 * b + 4 * c ][j];
  //   }
  // } );

  return cell;
}

void testConstructionAndSettersHelper()
{
  double * data = new double[8 * 3];
  pmpl::genericKernelWrapper( 8 * 3, data, [] SHIVA_DEVICE ( double * const kernelData )
  {
    auto const cell = makeLinearTransform( Xref );
    typename decltype(cell)::SupportIndexType index{0, 0, 0};

    auto const & transformData = cell.getData();

    forRange( index, [&transformData, &kernelData] ( auto const & i )
    {
      int const a = i.data[0];
      int const b = i.data[1];
      int const c = i.data[2];

      for ( int j = 0; j < 3; ++j )
      {
        kernelData[ 3 * ( a + 2 * b + 4 * c ) + j ] = transformData( a, b, c, j );
      }
    } );
  } );


  for ( int a = 0; a < 2; ++a )
  {
    for ( int b = 0; b < 2; ++b )
    {
      for ( int c = 0; c < 2; ++c )
      {
        for ( int j = 0; j < 3; ++j )
        {
          EXPECT_DOUBLE_EQ( data[ 3 * ( a + 2 * b + 4 * c ) + j ],
                            Xreference( a + 2 * b + 4 * c, j ) );
        }
      }
    }
  }
  delete[] data;
}
TEST( testLinearTransform, testConstructionAndSetters )
{
  testConstructionAndSettersHelper();
}


void testJacobianFunctionModifyLvalueRefArgHelper()
{
  double * data = new double[9 * 8];
  pmpl::genericKernelWrapper( 9 * 8, data, [] SHIVA_DEVICE ( double * const kernelData )
  {
    auto cell = makeLinearTransform( Xref );

    for ( int q = 0; q < 8; ++q )
    {
      typename std::remove_reference_t< decltype(cell) >::JacobianType J{ 0.0 };
      jacobian( cell, qCoords[q], J );
      for ( int i = 0; i < 3; ++i )
      {
        for ( int j = 0; j < 3; ++j )
        {
          kernelData[ 9 * q + 3 * i + j ] = J( i, j );
        }
      }
    }
  } );

  for ( int q = 0; q < 8; ++q )
  {
    for ( int i = 0; i < 3; ++i )
    {
      for ( int j = 0; j < 3; ++j )
      {
        EXPECT_NEAR( data[ 9 * q + 3 * i + j ], Jref[q][i][j], abs( Jref[q][i][j] ) * 1e-13 );
      }
    }
  }
  delete[] data;
}

TEST( testLinearTransform, testJacobianFunctionModifyLvalueRefArg )
{
  testJacobianFunctionModifyLvalueRefArgHelper();
}


void testJacobianFunctionReturnByValueHelper()
{
  double * data = new double[9 * 8];
  pmpl::genericKernelWrapper( 9 * 8, data, [] SHIVA_DEVICE ( double * const kernelData )
  {
    auto cell = makeLinearTransform( Xref );
    for ( int q = 0; q < 8; ++q )
    {
      auto J = jacobian( cell, qCoords[q] );
      for ( int i = 0; i < 3; ++i )
      {
        for ( int j = 0; j < 3; ++j )
        {
          kernelData[ 9 * q + 3 * i + j ] = J( i, j );
        }
      }
    }
  } );

  for ( int q = 0; q < 8; ++q )
  {
    for ( int i = 0; i < 3; ++i )
    {
      for ( int j = 0; j < 3; ++j )
      {
        EXPECT_NEAR( data[ 9 * q + 3 * i + j ], Jref[q][i][j], abs( Jref[q][i][j] ) * 1e-13 );
      }
    }
  }
  delete[] data;
}

TEST( testLinearTransform, testJacobianFunctionReturnByValue )
{
  testJacobianFunctionReturnByValueHelper();
}


void testInvJacobianFunctionModifyLvalueRefArgHelper()
{
  double * data = new double[10 * 8];
  pmpl::genericKernelWrapper( 10 * 8, data, [] SHIVA_DEVICE ( double * const kernelData )
  {
    auto cell = makeLinearTransform( Xref );
    for ( int q = 0; q < 8; ++q )
    {
      typename std::remove_reference_t< decltype(cell) >::JacobianType invJ{0.0};
      double detJ;

      inverseJacobian( cell, qCoords[q], invJ, detJ );

      kernelData[ 10 * q ] = detJ;
      for ( int i = 0; i < 3; ++i )
      {
        for ( int j = 0; j < 3; ++j )
        {
          kernelData[ 10 * q + 3 * i + j + 1 ] = invJ( i, j );
        }
      }
    }
  } );

  for ( int q = 0; q < 8; ++q )
  {
    EXPECT_NEAR( data[ 10 * q ], detJref[q], detJref[q] * 1e-13 );
    for ( int i = 0; i < 3; ++i )
    {
      for ( int j = 0; j < 3; ++j )
      {
        EXPECT_NEAR( data[ 10 * q + 3 * i + j + 1 ], invJref[q][i][j], abs( invJref[q][i][j] ) * 1e-13 );
      }
    }
  }
  delete[] data;

}

TEST( testLinearTransform, testInvJacobianFunctionModifyLvalueRefArg )
{
  testInvJacobianFunctionModifyLvalueRefArgHelper();
}


void testInvJacobianFunctionReturnByValueHelper()
{
  double * data = new double[10 * 8];
  pmpl::genericKernelWrapper( 10 * 8, data, [] SHIVA_DEVICE ( double * const kernelData )
  {
    auto cell = makeLinearTransform( Xref );
    for ( int q = 0; q < 8; ++q )
    {
      auto [ detJ, invJ ] = inverseJacobian( cell, qCoords[q] );

      kernelData[ 10 * q ] = detJ;
      for ( int i = 0; i < 3; ++i )
      {
        for ( int j = 0; j < 3; ++j )
        {
          kernelData[ 10 * q + 3 * i + j + 1 ] = invJ( i, j );
        }
      }
    }
  } );

  for ( int q = 0; q < 8; ++q )
  {
    EXPECT_NEAR( data[ 10 * q ], detJref[q], detJref[q] * 1e-13 );
    for ( int i = 0; i < 3; ++i )
    {
      for ( int j = 0; j < 3; ++j )
      {
        EXPECT_NEAR( data[ 10 * q + 3 * i + j + 1 ], invJref[q][i][j], abs( invJref[q][i][j] ) * 1e-13 );
      }
    }
  }
  delete[] data;
}

TEST( testLinearTransform, testInvJacobianFunctionReturnByValue )
{
  testInvJacobianFunctionReturnByValueHelper();
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
