
#include "testGeometryHelpers.hpp"
#include "../Cube.hpp"
#include "../Cuboid.hpp"
#include "../RectangularCuboid.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::geometry;
using namespace shiva::geometry::utilities;




TEST( testCube, testConstructionAndSetters )
{
  double const h = 10;
  testConstructionAndSettersHelper< Cube< double > >( 
    [h]( auto & data, auto const & constData )
    {
      data = h;
      EXPECT_EQ( constData, h );
    } );
}

TEST( testCube, testJacobian )
{
double const h = 10;
testJacobianHelper< Cube< double > >( 
  [h]( auto & data, auto const & cell )
  {
    data = h;
    typename std::remove_reference_t<decltype(cell)>::JacobianType J;
    jacobian( cell, J );
    EXPECT_EQ( J, ( h / 2 ) );
    typename std::remove_reference_t<decltype(cell)>::JacobianType invJ;
    double detJ = inverseJacobian( cell, invJ );
    EXPECT_EQ( detJ, 0.125*h*h*h );
    EXPECT_EQ( invJ, ( 2 / h ) );
  } );
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}