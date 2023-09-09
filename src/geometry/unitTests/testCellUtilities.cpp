
#include "../CellUtilities.hpp"

#include <gtest/gtest.h>

using namespace shiva;
using namespace shiva::cellUtilites;

template< typename CELLTYPE, typename FUNC >
void testJacobianHelper( FUNC && func )
{
  CELLTYPE cell;
  CELLTYPE const & cellConst = cell;

  typename CELLTYPE::DataType & data = cell.getData();

  func( data, cellConst );
}

TEST( testCellUtilities, testJacobian )
{
  {
  double const h = 10;
  testJacobianHelper< CellHexahedronUniformIJK< int, double > >( 
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

  {
  double const h[3] = { 10, 20, 30 } ;
  testJacobianHelper< CellHexahedronIJK< int, double > >( 
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

}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}