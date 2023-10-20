
#include "../Cube.hpp"
#include "../geometryUtilities.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::geometry;
using namespace shiva::geometry::utilities;




template< typename REAL_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto makeCube( REAL_TYPE const h )
{
  Cube< REAL_TYPE > cell;
  cell.setLength( h );
  static_assert( decltype(cell)::jacobianIsConstInCell() == true );
  return cell;
}


void testConstructionAndSettersHelper()
{
  constexpr double h = 3.14;
  double * data = nullptr;
  pmpl::genericKernelWrapper( 1, data, [] SHIVA_HOST_DEVICE ( double * const data )
  {
    auto cell = makeCube( h );
    data[0] = cell.getLength();
  } );
  EXPECT_EQ( data[0], h );
  pmpl::deallocateData( data );
}

TEST( testCube, testConstructionAndSetters )
{
  testConstructionAndSettersHelper();
}


void testJacobianFunctionModifyLvalueRefArgHelper()
{
  constexpr double h = 3.14;
  double * data = nullptr;
  pmpl::genericKernelWrapper( 1, data, [] SHIVA_HOST_DEVICE ( double * const data )
  {
    auto cell = makeCube( h );
    typename Cube< double >::JacobianType::type J;
    jacobian( cell, J );
    data[0] = J;
  } );
  EXPECT_EQ( data[0], ( h / 2 ) );
  pmpl::deallocateData( data );
}
TEST( testCube, testJacobianFunctionModifyLvalueRefArg )
{
  testJacobianFunctionModifyLvalueRefArgHelper();
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

  typename std::remove_reference_t< decltype(cell) >::JacobianType::type invJ;
  double detJ;
  inverseJacobian( cell, invJ, detJ );
  EXPECT_EQ( detJ, 0.125 * h * h * h );
  EXPECT_EQ( invJ, ( 2 / h ) );
}

TEST( testCube, testInvJacobianFunctionReturnByValue )
{
  double const h = 3.14;
  auto cell = makeCube( h );

  auto [ detJ, invJ ] = inverseJacobian( cell );
  EXPECT_EQ( detJ, 0.125 * h * h * h );
  EXPECT_EQ( invJ.data, ( 2 / h ) );
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
