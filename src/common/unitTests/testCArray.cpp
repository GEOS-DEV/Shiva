
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
  using Array = CArray< double, 2, 3, 4 >;
  static constexpr Array array{ initializer< Array >( std::make_integer_sequence< int, 2 * 3 * 4 >() ) };
};

void testTraitsHelper()
{
  int * data = nullptr;
  pmpl::genericKernelWrapper( 5, data, [] SHIVA_DEVICE ( int * const data )
  {
    using Array = TestCArrayHelper::Array;
    static_assert( Array::rank() == 3 );
    static_assert( Array::extent<0>() == 2 );
    static_assert( Array::extent<1>() == 3 );
    static_assert( Array::extent<2>() == 4 );
    static_assert( Array::size() == 24 );

    Array array;
    data[0] = array.rank();
    data[1] = array.extent<0>();
    data[2] = array.extent<1>();
    data[3] = array.extent<2>();
    data[4] = array.size();
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
    static_assert( cArrayDetail::stride< 2 >() == 1 );
    static_assert( cArrayDetail::stride< 3, 4 >() == 4 );
    static_assert( cArrayDetail::stride< 2, 3, 4 >() == 12 );
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
    using Array = TestCArrayHelper::Array;
    constexpr int na = Array::extent<0>();
    constexpr int nb = Array::extent<1>();
    constexpr int nc = Array::extent<2>();
    forSequence< na >( [] ( auto const ica )
    {
      constexpr int a = decltype(ica)::value;
      forSequence< nb >( [] ( auto const icb )
      {
        constexpr int b = decltype(icb)::value;
        forSequence< nc >( [] ( auto const icc )
        {
          constexpr int c = decltype(icc)::value;
          static_assert( TestCArrayHelper::Array::linearIndex< a, b, c >() == a * nb * nc + b * nc + c );
          static_assert( TestCArrayHelper::Array::linearIndex( a, b, c ) == a * nb * nc + b * nc + c );
        } );
      } );
    } );
  } );
}


void testLinearIndexRT()
{
  int * data = nullptr;
  pmpl::genericKernelWrapper( TestCArrayHelper::Array::size(), data, [] SHIVA_DEVICE ( int * const data )
  {
    TestCArrayHelper::Array array;
    int const na = array.extent<0>();
    int const nb = array.extent<1>();
    int const nc = array.extent<2>();

    for ( int a = 0; a < na; ++a )
    {
      for ( int b = 0; b < nb; ++b )
      {
        for ( int c = 0; c < nc; ++c )
        {
          data[ a * nb * nc + b * nc + c ] = array.linearIndex( a, b, c );
        }
      }
    }
  } );

  for ( int a = 0; a < TestCArrayHelper::Array::size(); ++a )
  {
    EXPECT_EQ( data[a], a );
  }
  pmpl::deallocateData( data );

}

TEST( testCArray, testLinerIndex )
{
  testLinearIndexCT();
  testLinearIndexRT();
}


void testParenthesesOperatorCT()
{
  using Array = TestCArrayHelper::Array;
  constexpr int na = Array::extent<0>();
  constexpr int nb = Array::extent<1>();
  constexpr int nc = Array::extent<2>();

  pmpl::genericKernelWrapper( [] SHIVA_DEVICE ()
  {
    forSequence< na >( [] ( auto const ica )
    {
      constexpr int a = decltype(ica)::value;
      forSequence< nb >( [] ( auto const icb )
      {
        constexpr int b = decltype(icb)::value;
        forSequence< nc >( [] ( auto const icc )
        {
          constexpr int c = decltype(icc)::value;
          static_assert( pmpl::check( TestCArrayHelper::array.operator()< a, b, c >(),
                                      3.14159 * Array::linearIndex( a, b, c ),
                                      1.0e-12 ) );
          static_assert( pmpl::check( TestCArrayHelper::array( a, b, c ),
                                      3.14159 * Array::linearIndex( a, b, c ),
                                      1.0e-12 ) );
        } );
      } );
    } );
  } );
}

void testParenthesesOperatorRT()
{
  double * data = nullptr;
  pmpl::genericKernelWrapper( TestCArrayHelper::Array::size(), data, [] SHIVA_DEVICE ( double * const data )
  {
    TestCArrayHelper::Array const array{ initializer< TestCArrayHelper::Array >( std::make_integer_sequence< int, 2 * 3 * 4 >() ) };;
    int const na = array.extent<0>();
    int const nb = array.extent<1>();
    int const nc = array.extent<2>();

    for ( int a = 0; a < na; ++a )
    {
      for ( int b = 0; b < nb; ++b )
      {
        for ( int c = 0; c < nc; ++c )
        {
          data[ a * nb * nc + b * nc + c ] = array( a, b, c );
        }
      }
    }
  } );

  for ( int a = 0; a < TestCArrayHelper::Array::size(); ++a )
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
