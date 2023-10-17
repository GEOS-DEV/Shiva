
#include "../RectangularCuboid.hpp"
#include "../geometryUtilities.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::geometry;
using namespace shiva::geometry::utilities;


template< typename REAL_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto makeRectangularCuboid( REAL_TYPE const (&h)[3] )
{
  RectangularCuboid< REAL_TYPE > cell;
  cell.setLength( h );

  return cell;
}


void testConstructionAndSettersHelper()
{
  constexpr double h[3] = { 10, 20, 30 };
  double * data;
  pmpl::genericKernelWrapper( 3, data, [=] SHIVA_HOST_DEVICE ( double * const data )
  {
    auto cell = makeRectangularCuboid( h );
    auto const & constData = cell.getLengths();
    data[0] = constData[0];
    data[1] = constData[1];
    data[2] = constData[2];
  } );
  EXPECT_EQ( data[0], h[0] );
  EXPECT_EQ( data[1], h[1] );
  EXPECT_EQ( data[2], h[2] );

  pmpl::deallocateData( data );
}
TEST( testRectangularCuboid, testConstructionAndSetters )
{
  testConstructionAndSettersHelper();
}


void testJacobianFunctionModifyLvalueRefArgHelper()
{
  constexpr double h[3] = { 10, 20, 30 };
  double * data;
  pmpl::genericKernelWrapper( 3, data, [=] SHIVA_HOST_DEVICE ( double * const data )
  {
    auto cell = makeRectangularCuboid( h );

    typename std::remove_reference_t< decltype(cell) >::JacobianType::type J;
    jacobian( cell, J );
    data[0] = J[0];
    data[1] = J[1];
    data[2] = J[2];
  } );
  EXPECT_EQ( data[0], ( h[0] / 2 ) );
  EXPECT_EQ( data[1], ( h[1] / 2 ) );
  EXPECT_EQ( data[2], ( h[2] / 2 ) );

  pmpl::deallocateData( data );
}

TEST( testRectangularCuboid, testJacobianFunctionModifyLvalueRefArg )
{
  testJacobianFunctionModifyLvalueRefArgHelper();
}

void testJacobianFunctionReturnByValueHelper()
{
  constexpr double h[3] = { 10, 20, 30 };
  double * data;
  pmpl::genericKernelWrapper( 3, data, [=] SHIVA_HOST_DEVICE ( double * const data )
  {
    auto cell = makeRectangularCuboid( h );

    auto J = jacobian( cell );
    data[0] = J[0];
    data[1] = J[1];
    data[2] = J[2];
  } );
  EXPECT_EQ( data[0], ( h[0] / 2 ) );
  EXPECT_EQ( data[1], ( h[1] / 2 ) );
  EXPECT_EQ( data[2], ( h[2] / 2 ) );

  pmpl::deallocateData( data );
}
TEST( testRectangularCuboid, testJacobianFunctionReturnByValue )
{
  testJacobianFunctionReturnByValueHelper();
}

void testInvJacobianFunctionModifyLvalueRefArgHelper()
{
  constexpr double h[3] = { 10, 20, 30 };
  double * data;
  pmpl::genericKernelWrapper( 4, data, [=] SHIVA_HOST_DEVICE ( double * const data )
  {
    auto cell = makeRectangularCuboid( h );

    typename std::remove_reference_t< decltype(cell) >::JacobianType::type invJ;
    double detJ;
    inverseJacobian( cell, invJ, detJ );
    data[0] = detJ;
    data[1] = invJ[0];
    data[2] = invJ[1];
    data[3] = invJ[2];
  } );
  EXPECT_EQ( data[0], 0.125 * h[0] * h[1] * h[2] );
  EXPECT_EQ( data[1], ( 2 / h[0] ) );
  EXPECT_EQ( data[2], ( 2 / h[1] ) );
  EXPECT_EQ( data[3], ( 2 / h[2] ) );

  pmpl::deallocateData( data );
}
TEST( testRectangularCuboid, testInvJacobianFunctionModifyLvalueRefArg )
{
  testInvJacobianFunctionModifyLvalueRefArgHelper();
}

void testInvJacobianFunctionReturnByValueHelper()
{
  constexpr double h[3] = { 10, 20, 30 };
  double * data;
  pmpl::genericKernelWrapper( 4, data, [=] SHIVA_HOST_DEVICE ( double * const data )
  {
    auto cell = makeRectangularCuboid( h );

    auto [ detJ, invJ ] = inverseJacobian( cell );
    data[0] = detJ;
    data[1] = invJ[0];
    data[2] = invJ[1];
    data[3] = invJ[2];
  } );
  EXPECT_EQ( data[0], 0.125 * h[0] * h[1] * h[2] );
  EXPECT_EQ( data[1], ( 2 / h[0] ) );
  EXPECT_EQ( data[2], ( 2 / h[1] ) );
  EXPECT_EQ( data[3], ( 2 / h[2] ) );

  pmpl::deallocateData( data );
}
TEST( testRectangularCuboid, testInvJacobianFunctionReturnByValue )
{
  testInvJacobianFunctionReturnByValueHelper();
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
