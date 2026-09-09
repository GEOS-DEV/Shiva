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


#include "shiva/geometry/mapping/Scaling.hpp"
#include "shiva/geometry/mapping/JacobianTransforms.hpp"
#include "shiva/common/pmpl.hpp"
#include "shiva/functions/quadrature/Quadrature.hpp"

#include <algorithm>
#include <gtest/gtest.h>
#include <iterator>
#include <type_traits>

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

template< typename REAL_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE bool matchesExpectedValue( REAL_TYPE const actual, int const expected )
{
  REAL_TYPE const difference = actual - expected;
  return difference > -1e-6 && difference < 1e-6;
}

// Use void for deduction, REAL_TYPE for an explicit scalar argument, or a quadrature type.
// Initialize all entries to sentinels to verify that each overload overwrites the output.
template< typename REAL_TYPE, typename QUADRATURE = void, int ... QA >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE bool checkJacobianOverload( Scaling< REAL_TYPE > const & cell )
{
  typename Scaling< REAL_TYPE >::JacobianType J{};
  J( 0 ) = -1;
  J( 1 ) = -2;
  J( 2 ) = -3;
  if constexpr ( std::is_same_v< QUADRATURE, void > )
  {
    jacobian( cell, J );
  }
  else
  {
    jacobian< QUADRATURE, QA ... >( cell, J );
  }
  return matchesExpectedValue( J( 0 ), 5 ) && matchesExpectedValue( J( 1 ), 10 ) && matchesExpectedValue( J( 2 ), 15 );
}

// Share the same interface checks between constant evaluation and host/device execution.
template< typename REAL_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE void checkScalingInterface( bool * const checks )
{
  using Cell = Scaling< REAL_TYPE >;
  using Quadrature = QuadratureGaussLegendre< REAL_TYPE, 2 >;
  constexpr REAL_TYPE h[3] = { 10, 20, 30 };
  auto cell = makeScaling( h );
  Cell const & constCell = cell;
  static_assert( std::is_same_v< decltype(cell.getData()), typename Cell::DataType & > );
  static_assert( std::is_same_v< decltype(constCell.getData()), typename Cell::DataType const & > );

  // Deduced and explicit scalar calls, then empty, partial, and full quadrature index packs.
  checks[0] = checkJacobianOverload< REAL_TYPE >( cell );
  checks[1] = checkJacobianOverload< REAL_TYPE, REAL_TYPE >( cell );
  checks[2] = checkJacobianOverload< REAL_TYPE, Quadrature >( cell );
  checks[3] = checkJacobianOverload< REAL_TYPE, Quadrature, 0 >( cell );
  checks[4] = checkJacobianOverload< REAL_TYPE, Quadrature, 0, 1 >( cell );
  checks[5] = checkJacobianOverload< REAL_TYPE, Quadrature, 0, 0, 0 >( cell );
  checks[6] = checkJacobianOverload< REAL_TYPE, Quadrature, 1, 1, 1 >( cell );
  checks[7] = checkJacobianOverload< REAL_TYPE, Quadrature, 0, 1, 0 >( cell );

  // Both const access and Jacobian evaluation must observe mutations through the getter.
  auto & lengths = cell.getData();
  lengths[0] = 12;
  lengths[1] = 24;
  lengths[2] = 36;
  checks[8] = matchesExpectedValue( constCell.getData()[0], 12 ) &&
              matchesExpectedValue( constCell.getData()[1], 24 ) &&
              matchesExpectedValue( constCell.getData()[2], 36 );
  typename Cell::JacobianType J{};
  jacobian< Quadrature, 1, 0, 1 >( constCell, J );
  checks[9] = matchesExpectedValue( J( 0 ), 6 ) && matchesExpectedValue( J( 1 ), 12 ) && matchesExpectedValue( J( 2 ), 18 );
}

template< typename REAL_TYPE >
constexpr bool checkScalingInterfaceConstexpr()
{
  bool checks[10]{};
  checkScalingInterface< REAL_TYPE >( checks );
  return std::all_of( std::begin( checks ), std::end( checks ), [] ( bool const check ) constexpr { return check; } );
}

static_assert( checkScalingInterfaceConstexpr< float >() );
static_assert( checkScalingInterfaceConstexpr< double >() );

template< typename REAL_TYPE >
void testScalingInterfaceHelper()
{
  bool checks[10]{};
  pmpl::genericKernelWrapper( 10, checks, [ = ] SHIVA_HOST_DEVICE ( bool * const kdata )
  {
    checkScalingInterface< REAL_TYPE >( kdata );
  } );
  for( int i = 0; i < 10; ++i )
  {
    EXPECT_TRUE( checks[i] ) << "Scaling interface check " << i;
  }
}

TEST( testScaling, testScalingInterfaceFloat )
{
  testScalingInterfaceHelper< float >();
}

TEST( testScaling, testScalingInterfaceDouble )
{
  testScalingInterfaceHelper< double >();
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

#if defined(SHIVA_ENABLE_CUDA) && SHIVA_CUDA_MAJOR < 12
    auto tmp  = inverseJacobian( cell );
    auto detJ = shiva::get< 0 >( tmp );
    auto invJ = shiva::get< 1 >( tmp );
#else
    auto [detJ, invJ] = inverseJacobian( cell );
#endif
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
