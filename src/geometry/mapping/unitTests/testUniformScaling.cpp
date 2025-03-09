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


#include "../UniformScaling.hpp"
#include "../JacobianTransforms.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::geometry;
using namespace shiva::geometry::utilities;



template< typename REAL_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto makeUniformScaling( REAL_TYPE const h )
{
  UniformScaling< REAL_TYPE > cell;
  cell.setData( h );
  static_assert( decltype(cell)::jacobianIsConstInCell() == true );
  return cell;
}


void testConstructionAndSettersHelper()
{
  constexpr double h = 3.14;
  double * data = new double[1];
  pmpl::genericKernelWrapper( 1, data, [] SHIVA_HOST_DEVICE ( double * const kdata )
  {
    auto cell = makeUniformScaling( h );
    kdata[0] = cell.getData();
  } );
  EXPECT_EQ( data[0], h );
  delete[] data;
}

TEST( testUniformScaling, testConstructionAndSetters )
{
  testConstructionAndSettersHelper();
}


void testJacobianFunctionModifyLvalueRefArgHelper()
{
  constexpr double h = 3.14;
  double * data = new double[1];
  pmpl::genericKernelWrapper( 1, data, [] SHIVA_HOST_DEVICE ( double * const kdata )
  {
    auto cell = makeUniformScaling( h );
    typename UniformScaling< double >::JacobianType J;
    jacobian( cell, J );
    kdata[0] = J( 0 );
  } );
  EXPECT_EQ( data[0], ( h / 2 ) );
  delete[] data;
}
TEST( testUniformScaling, testJacobianFunctionModifyLvalueRefArg )
{
  testJacobianFunctionModifyLvalueRefArgHelper();
}


TEST( testUniformScaling, testJacobianFunctionReturnByValue )
{
  double const h = 3.14;
  auto cell = makeUniformScaling( h );

  auto J = jacobian( cell );
  EXPECT_EQ( J( 0 ), ( h / 2 ) );
}

TEST( testUniformScaling, testInvJacobianFunctionModifyLvalueRefArg )
{
  double const h = 3.14;
  auto cell = makeUniformScaling( h );

  typename std::remove_reference_t< decltype(cell) >::JacobianType invJ;
  double detJ;
  inverseJacobian( cell, invJ, detJ );
  EXPECT_EQ( detJ, 0.125 * h * h * h );
  EXPECT_EQ( invJ( 0 ), ( 2 / h ) );
}

TEST( testUniformScaling, testInvJacobianFunctionReturnByValue )
{
  double const h = 3.14;
  auto cell = makeUniformScaling( h );

  auto [ detJ, invJ ] = inverseJacobian( cell );
  EXPECT_EQ( detJ, 0.125 * h * h * h );
  EXPECT_EQ( invJ( 0 ), ( 2 / h ) );
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
