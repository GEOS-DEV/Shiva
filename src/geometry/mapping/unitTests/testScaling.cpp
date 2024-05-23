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


#include "../Scaling.hpp"
#include "../JacobianTransforms.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::geometry;
using namespace shiva::geometry::utilities;


template< typename REAL_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto makeScaling( REAL_TYPE const (&h)[3] )
{
  Scaling< REAL_TYPE > cell;
  cell.setData( h );

  return cell;
}


void testConstructionAndSettersHelper()
{
  constexpr double h[3] = { 10, 20, 30 };
  double * data = new double[3];
  pmpl::genericKernelWrapper( 3, data, [ = ] SHIVA_HOST_DEVICE ( double * const kdata )
  {
    auto cell = makeScaling( h );
    auto const & constData = cell.getData();
    kdata[0] = constData[0];
    kdata[1] = constData[1];
    kdata[2] = constData[2];
  } );
  EXPECT_EQ( data[0], h[0] );
  EXPECT_EQ( data[1], h[1] );
  EXPECT_EQ( data[2], h[2] );

  delete[] data;
}
TEST( testScaling, testConstructionAndSetters )
{
  testConstructionAndSettersHelper();
}


void testJacobianFunctionModifyLvalueRefArgHelper()
{
  constexpr double h[3] = { 10, 20, 30 };
  double * data = new double[3];
  pmpl::genericKernelWrapper( 3, data, [ = ] SHIVA_HOST_DEVICE ( double * const kdata )
  {
    auto cell = makeScaling( h );

    typename std::remove_reference_t< decltype(cell) >::JacobianType J;
    jacobian( cell, J );
    kdata[0] = J( 0 );
    kdata[1] = J( 1 );
    kdata[2] = J( 2 );
  } );
  EXPECT_EQ( data[0], ( h[0] / 2 ) );
  EXPECT_EQ( data[1], ( h[1] / 2 ) );
  EXPECT_EQ( data[2], ( h[2] / 2 ) );

  delete[] data;
}

TEST( testScaling, testJacobianFunctionModifyLvalueRefArg )
{
  testJacobianFunctionModifyLvalueRefArgHelper();
}

void testJacobianFunctionReturnByValueHelper()
{
  constexpr double h[3] = { 10, 20, 30 };
  double * data = new double[3];
  pmpl::genericKernelWrapper( 3, data, [ = ] SHIVA_HOST_DEVICE ( double * const kdata )
  {
    auto cell = makeScaling( h );

    auto J = jacobian( cell );
    kdata[0] = J( 0 );
    kdata[1] = J( 1 );
    kdata[2] = J( 2 );
  } );
  EXPECT_EQ( data[0], ( h[0] / 2 ) );
  EXPECT_EQ( data[1], ( h[1] / 2 ) );
  EXPECT_EQ( data[2], ( h[2] / 2 ) );

  delete[] data;
}
TEST( testScaling, testJacobianFunctionReturnByValue )
{
  testJacobianFunctionReturnByValueHelper();
}

void testInvJacobianFunctionModifyLvalueRefArgHelper()
{
  constexpr double h[3] = { 10, 20, 30 };
  double * data = new double[4];
  pmpl::genericKernelWrapper( 4, data, [ = ] SHIVA_HOST_DEVICE ( double * const kdata )
  {
    auto cell = makeScaling( h );

    typename std::remove_reference_t< decltype(cell) >::JacobianType invJ;
    double detJ;
    inverseJacobian( cell, invJ, detJ );
    kdata[0] = detJ;
    kdata[1] = invJ( 0 );
    kdata[2] = invJ( 1 );
    kdata[3] = invJ( 2 );
  } );
  EXPECT_EQ( data[0], 0.125 * h[0] * h[1] * h[2] );
  EXPECT_EQ( data[1], ( 2 / h[0] ) );
  EXPECT_EQ( data[2], ( 2 / h[1] ) );
  EXPECT_EQ( data[3], ( 2 / h[2] ) );

  delete[] data;
}
TEST( testScaling, testInvJacobianFunctionModifyLvalueRefArg )
{
  testInvJacobianFunctionModifyLvalueRefArgHelper();
}

void testInvJacobianFunctionReturnByValueHelper()
{
  constexpr double h[3] = { 10, 20, 30 };
  double * data = new double[4];
  pmpl::genericKernelWrapper( 4, data, [ = ] SHIVA_HOST_DEVICE ( double * const kdata )
  {
    auto cell = makeScaling( h );

    auto [ detJ, invJ ] = inverseJacobian( cell );
    kdata[0] = detJ;
    kdata[1] = invJ( 0 );
    kdata[2] = invJ( 1 );
    kdata[3] = invJ( 2 );
  } );
  EXPECT_EQ( data[0], 0.125 * h[0] * h[1] * h[2] );
  EXPECT_EQ( data[1], ( 2 / h[0] ) );
  EXPECT_EQ( data[2], ( 2 / h[1] ) );
  EXPECT_EQ( data[3], ( 2 / h[2] ) );

  delete[] data;
}
TEST( testScaling, testInvJacobianFunctionReturnByValue )
{
  testInvJacobianFunctionReturnByValueHelper();
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
