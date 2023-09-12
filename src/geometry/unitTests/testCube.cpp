
#include "../Cube.hpp"
#include "../geometryUtilities.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::geometry;
using namespace shiva::geometry::utilities;


template< typename REAL_TYPE >
auto makeCube( REAL_TYPE const h )
{
  Cube< REAL_TYPE > cell;
  cell.getData() = h;
  return cell;
}

TEST( testCube, testConstructionAndSetters )
{
  double const h = 3.14;
  auto cell = makeCube( h );

  auto const & constData = cell.getData();
  EXPECT_EQ( constData, h );
  static_assert( decltype(cell)::jacobianIsConstInCell() == true );
}

TEST( testCube, testJacobianFunctionModifyLvalueRefArg )
{
  double const h = 3.14;
  auto cell = makeCube( h );

  typename Cube< double >::JacobianType::type J;
  jacobian( cell, J );
  EXPECT_EQ( J, ( h / 2 ) );
}


TEST( testCube, testJacobianFunctionReturnByValue )
{
  double const h = 3.14;
  auto cell = makeCube( h );

  auto J = jacobian( cell );
  EXPECT_EQ( J.data, ( h / 2 ) );
}

TEST( testCube, testInvJacobianFunctionModifyLvalueRefArg )
{
  double const h = 3.14;
  auto cell = makeCube( h );

  typename std::remove_reference_t<decltype(cell)>::JacobianType::type invJ;
  double detJ;
  inverseJacobian( cell, invJ, detJ );
  EXPECT_EQ( detJ, 0.125*h*h*h );
  EXPECT_EQ( invJ, ( 2 / h ) );
 }

TEST( testCube, testInvJacobianFunctionReturnByValue )
{
  double const h = 3.14;
  auto cell = makeCube( h );

  auto const [ detJ, invJ ] = inverseJacobian( cell );
  EXPECT_EQ( detJ, 0.125*h*h*h );
  EXPECT_EQ( invJ.data, ( 2 / h ) );
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}