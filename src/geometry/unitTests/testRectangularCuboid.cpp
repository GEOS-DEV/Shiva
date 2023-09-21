
#include "../RectangularCuboid.hpp"
#include "../geometryUtilities.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::geometry;
using namespace shiva::geometry::utilities;


template< typename REAL_TYPE >
auto makeRectangularCuboid( REAL_TYPE const (&h)[3] )
{
  RectangularCuboid< REAL_TYPE > cell;
  cell.setLength( h );

  return cell;
}

TEST( testRectangularCuboid, testConstructionAndSetters )
{
  double const h[3] = { 10, 20, 30 } ;
  auto cell = makeRectangularCuboid( h );

  auto const & constData = cell.getLengths();
  EXPECT_EQ( constData[0], h[0] );
  EXPECT_EQ( constData[1], h[1] );
  EXPECT_EQ( constData[2], h[2] );
}

TEST( testRectangularCuboid, testJacobianFunctionModifyLvalueRefArg )
{
  double const h[3] = { 10, 20, 30 } ;
  auto cell = makeRectangularCuboid( h );

  typename std::remove_reference_t<decltype(cell)>::JacobianType::type J ;
  jacobian( cell, J );
  EXPECT_EQ( J[0], ( h[0] / 2 ) );
  EXPECT_EQ( J[1], ( h[1] / 2 ) );
  EXPECT_EQ( J[2], ( h[2] / 2 ) );
}

TEST( testRectangularCuboid, testJacobianFunctionReturnByValue )
{
  double const h[3] = { 10, 20, 30 } ;
  auto cell = makeRectangularCuboid( h );

  auto J = jacobian( cell );
  EXPECT_EQ( J.data[0], ( h[0] / 2 ) );
  EXPECT_EQ( J.data[1], ( h[1] / 2 ) );
  EXPECT_EQ( J.data[2], ( h[2] / 2 ) );
}


TEST( testRectangularCuboid, testInvJacobianFunctionModifyLvalueRefArg )
{
  double const h[3] = { 10, 20, 30 } ;
  auto cell = makeRectangularCuboid( h );

  typename std::remove_reference_t<decltype(cell)>::JacobianType::type invJ;
  double detJ;
  inverseJacobian( cell, invJ, detJ );
    
  EXPECT_EQ( detJ, 0.125*h[0]*h[1]*h[2] );
  EXPECT_EQ( invJ[0], ( 2 / h[0] ) );
  EXPECT_EQ( invJ[1], ( 2 / h[1] ) );
  EXPECT_EQ( invJ[2], ( 2 / h[2] ) );
}

TEST( testRectangularCuboid, testInvJacobianFunctionReturnByValue )
{
  double const h[3] = { 10, 20, 30 } ;
  auto cell = makeRectangularCuboid( h );

  auto const [ detJ, invJ ] = inverseJacobian( cell );
  EXPECT_EQ( detJ, 0.125*h[0]*h[1]*h[2] );
  EXPECT_EQ( invJ.data[0], ( 2 / h[0] ) );
  EXPECT_EQ( invJ.data[1], ( 2 / h[1] ) );
  EXPECT_EQ( invJ.data[2], ( 2 / h[2] ) );
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}