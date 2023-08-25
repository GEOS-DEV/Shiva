
#include "../CellHexahedron.hpp"

#include <gtest/gtest.h>

using namespace shiva;

template< typename CELLTYPE, typename FUNC >
void testConstructionAndSettersHelper( FUNC && setData )
{
  CELLTYPE cell;
  CELLTYPE const & cellConst = cell;

  typename CELLTYPE::DataType & data = cell.getData();
  typename CELLTYPE::DataType const & constData = cellConst.getData();

  setData( data, constData );
}

TEST( testCellHexahedron, testConstructionAndSetters )
{
  {
  double const h = 10;
  testConstructionAndSettersHelper< CellHexahedronUniformIJK< int, double > >( 
    [h]( auto & data, auto const & constData )
    {
      data = h;
      EXPECT_EQ( constData, h );
    } );
  }

  {
  double const h[3] = { 10, 20, 30 } ;
  testConstructionAndSettersHelper< CellHexahedronIJK< int, double > >( 
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

}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}