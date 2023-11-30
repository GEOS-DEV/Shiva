
#include "../CArray.hpp"

#include "../SequenceUtilities.hpp"
#include "common/pmpl.hpp"


#include <gtest/gtest.h>

using namespace shiva;



template< typename ARRAY_TYPE, int ... VALUES >
static constexpr auto initializer( std::integer_sequence< int, VALUES ... > )
{
  return ARRAY_TYPE{ (VALUES * 3.14159) ... };
}

struct TestCArrayHelper
{

  using Array1d = CArrayNd< double const, 4 >;
  static constexpr Array1d array1d{ initializer< Array1d >( std::make_integer_sequence< int, 4 >() ) };

  using Array3d = CArrayNd< double const, 2, 3, 4 >;
  static constexpr Array3d array3d{ initializer< Array3d >( std::make_integer_sequence< int, 2 * 3 * 4 >() ) };
};


TEST( testCArray, testSingleValueInitialization )
{
  CArrayNd< double, 2, 3, 4 > array{ 0.0 };
  for ( int a = 0; a < 24; ++a )
  {
    EXPECT_EQ( array.data()[a], 0.0 );
  }
}

TEST( testCArray, testViewInitialization )
{
  double data[24]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  CArrayViewNd< double, 2, 3, 4 > array{ data };
  for ( int a = 0; a < 24; ++a )
  {
    EXPECT_EQ( array.data()[a], a );
  }

  double const data3c[2][3][4]{ { {  0, 1, 2, 3 },
    {  4, 5, 6, 7 },
    {  8, 9, 10, 11 } },
    { { 12, 13, 14, 15 },
      { 16, 17, 18, 19 },
      { 20, 21, 22, 23 } } };
  CArrayViewNd< double const, 2, 3, 4 > const array3c{ &(data3c[0][0][0]) };
  for ( int a = 0; a < 24; ++a )
  {
    EXPECT_EQ( array3c.data()[a], a );
  }



  double data3[2][3][4] = { { {0.0} } };
  CArrayViewNd< double, 2, 3, 4 > array3{ &(data3[0][0][0]) };
  for ( int a = 0; a < 2; ++a )
  {
    for ( int b = 0; b < 3; ++b )
    {
      for ( int c = 0; c < 4; ++c )
      {
        array3( a, b, c ) = 100 * a + 10 * b + c;
      }
    }
  }

}

void testTraitsHelper()
{
  int * data = nullptr;
  pmpl::genericKernelWrapper( 5, data, [] SHIVA_DEVICE ( int * const kernelData )
  {
    using Array = TestCArrayHelper::Array3d;
    static_assert( Array::rank() == 3 );
    static_assert( Array::extent< 0 >() == 2 );
    static_assert( Array::extent< 1 >() == 3 );
    static_assert( Array::extent< 2 >() == 4 );
    static_assert( Array::size() == 24 );

    kernelData[0] = Array::rank();
    kernelData[1] = Array::extent< 0 >();
    kernelData[2] = Array::extent< 1 >();
    kernelData[3] = Array::extent< 2 >();
    kernelData[4] = Array::size();
  } );

  EXPECT_EQ( data[0], 3 );
  EXPECT_EQ( data[1], 2 );
  EXPECT_EQ( data[2], 3 );
  EXPECT_EQ( data[3], 4 );
  EXPECT_EQ( data[4], 24 );
  pmpl::deallocateData( data );

}
TEST( testCArray, testTraits )
{
  testTraitsHelper();
}

void testStridesHelper()
{
  pmpl::genericKernelWrapper( [] SHIVA_DEVICE () 
  {
    static_assert( CArrayHelper::stride< 2 >() == 1 );
    static_assert( CArrayHelper::stride< 3, 4 >() == 4 );
    static_assert( CArrayHelper::stride< 2, 3, 4 >() == 12 );
  } );
}

TEST( testCArray, testStrides )
{
  testStridesHelper();
}


void testLinearIndexCT()
{
  pmpl::genericKernelWrapper( [] SHIVA_DEVICE () 
  {
    using Array = TestCArrayHelper::Array3d;
    constexpr int na = Array::extent< 0 >();
    constexpr int nb = Array::extent< 1 >();
    constexpr int nc = Array::extent< 2 >();
    forSequence< na >( [] ( auto const ica )
    {
      constexpr int a = decltype(ica)::value;
      forSequence< nb >( [] ( auto const icb )
      {
        constexpr int b = decltype(icb)::value;
        forSequence< nc >( [] ( auto const icc )
        {
          constexpr int c = decltype(icc)::value;
          static_assert( CArrayHelper::linearIndexHelper< 2, 3, 4 >::template eval< a, b, c >() == a * nb * nc + b * nc + c );
          static_assert( CArrayHelper::linearIndexHelper< 2, 3, 4 >::eval( a, b, c ) == a * nb * nc + b * nc + c );
        } );
      } );
    } );
  } );
}


void testLinearIndexRT()
{
  int * data = nullptr;
  pmpl::genericKernelWrapper( TestCArrayHelper::Array3d::size(), data, [] SHIVA_DEVICE ( int * const kernelData )
  {
    int const na = TestCArrayHelper::array3d.extent< 0 >();
    int const nb = TestCArrayHelper::array3d.extent< 1 >();
    int const nc = TestCArrayHelper::array3d.extent< 2 >();

    for ( int a = 0; a < na; ++a )
    {
      for ( int b = 0; b < nb; ++b )
      {
        for ( int c = 0; c < nc; ++c )
        {
          kernelData[ a * nb * nc + b * nc + c ] = CArrayHelper::linearIndexHelper< 2, 3, 4 >::eval( a, b, c );
        }
      }
    }
  } );

  for ( int a = 0; a < TestCArrayHelper::Array3d::size(); ++a )
  {
    EXPECT_EQ( data[a], a );
  }
  pmpl::deallocateData( data );

}

TEST( testCArray, testLinearIndex )
{
  testLinearIndexCT();
  testLinearIndexRT();
}


void testParenthesesOperatorCT()
{
  using Array = TestCArrayHelper::Array3d;
  constexpr int na = Array::extent< 0 >();
  constexpr int nb = Array::extent< 1 >();
  constexpr int nc = Array::extent< 2 >();

  pmpl::genericKernelWrapper( [] SHIVA_DEVICE () 
  {
    forSequence< na >( [] ( auto const ica )
    {
      constexpr int a = decltype(ica)::value;
      forSequence< nb >( [ = ] ( auto const icb )
      {
        constexpr int b = decltype(icb)::value;
        forSequence< nc >( [ = ] ( auto const icc )
        {
          constexpr int c = decltype(icc)::value;
          static_assert( pmpl::check( TestCArrayHelper::array3d.operator()< a, b, c >(),
                                      3.14159 * CArrayHelper::linearIndexHelper< 2, 3, 4 >::eval( a, b, c ),
                                      1.0e-12 ) );
          static_assert( pmpl::check( TestCArrayHelper::array3d( a, b, c ),
                                      3.14159 * CArrayHelper::linearIndexHelper< 2, 3, 4 >::eval( a, b, c ),
                                      1.0e-12 ) );
        } );
      } );
    } );
  } );
}

void testParenthesesOperatorRT()
{
  double * data = nullptr;
  pmpl::genericKernelWrapper( TestCArrayHelper::Array3d::size(), data, [] SHIVA_DEVICE ( double * const kernelData ) 
  {
    TestCArrayHelper::Array3d const array{ initializer< TestCArrayHelper::Array3d >( std::make_integer_sequence< int, 2 * 3 * 4 >() ) };;
    int const na = array.extent< 0 >();
    int const nb = array.extent< 1 >();
    int const nc = array.extent< 2 >();

    for ( int a = 0; a < na; ++a )
    {
      for ( int b = 0; b < nb; ++b )
      {
        for ( int c = 0; c < nc; ++c )
        {
          kernelData[ a * nb * nc + b * nc + c ] = array( a, b, c );
        }
      }
    }
  } );

  for ( int a = 0; a < TestCArrayHelper::Array3d::size(); ++a )
  {
    EXPECT_EQ( data[a], 3.14159 * a );
  }
  pmpl::deallocateData( data );


}
TEST( testCArray, testParenthesesOperator )
{
  testParenthesesOperatorCT();
  testParenthesesOperatorRT();
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
