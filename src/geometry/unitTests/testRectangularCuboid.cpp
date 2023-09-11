
#include "testGeometryHelpers.hpp"
#include "../RectangularCuboid.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::geometry;
using namespace shiva::geometry::utilities;




TEST( testRectangularCuboid, testConstructionAndSetters )
{
  double const h[3] = { 10, 20, 30 } ;
  testConstructionAndSettersHelper< RectangularCuboid< double > >( 
    [h]( auto & data, auto const & constData )
    {
      data[0] = h[0];
      data[1] = h[1];
      data[2] = h[2];
      EXPECT_EQ( constData[0], h[0] );
      EXPECT_EQ( constData[1], h[1] );
      EXPECT_EQ( constData[2], h[2] );
    } );
}






TEST( testRectangularCuboid, testJacobian )
{
double const h[3] = { 10, 20, 30 } ;
testJacobianHelper< RectangularCuboid< double > >( 
  [h]( auto & data, auto const & cell )
  {
    data[0] = h[0];
    data[1] = h[1];
    data[2] = h[2];
    
    typename std::remove_reference_t<decltype(cell)>::JacobianType J;
    jacobian( cell, J );
    EXPECT_EQ( J[0], ( h[0] / 2 ) );
    EXPECT_EQ( J[1], ( h[1] / 2 ) );
    EXPECT_EQ( J[2], ( h[2] / 2 ) );
    
    typename std::remove_reference_t<decltype(cell)>::JacobianType invJ;
    double detJ = inverseJacobian( cell, invJ );
    EXPECT_EQ( detJ, 0.125*h[0]*h[1]*h[2] );
    EXPECT_EQ( invJ[0], ( 2 / h[0] ) );
    EXPECT_EQ( invJ[1], ( 2 / h[1] ) );
    EXPECT_EQ( invJ[2], ( 2 / h[2] ) );
  } );
}



int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}