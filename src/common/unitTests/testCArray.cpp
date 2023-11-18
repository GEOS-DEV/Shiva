
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


TEST( testCArray, testTraits )
{
  pmpl::genericKernelWrapper( [] ()
  {
    using Array = TestCArrayHelper::Array;
    static_assert( Array::rank() == 3 );
    static_assert( Array::extent<0>() == 2 );
    static_assert( Array::extent<1>() == 3 );
    static_assert( Array::extent<2>() == 4 );
    static_assert( Array::size() == 24 );

    Array array;
    EXPECT_EQ( array.rank(), 3 );
    EXPECT_EQ( array.extent<0>(), 2 );
    EXPECT_EQ( array.extent<1>(), 3 );
    EXPECT_EQ( array.extent<2>(), 4 );
    EXPECT_EQ( array.size(), 24 );
  } );
}

TEST( testCArray, testStrides )
{
  pmpl::genericKernelWrapper( [] ()
  {
    static_assert( cArrayDetail::stride< 2 >() == 1 );
    static_assert( cArrayDetail::stride< 3, 4 >() == 4 );
    static_assert( cArrayDetail::stride< 2, 3, 4 >() == 12 );
  } );
}


void testLinearIndexCT()
{
  pmpl::genericKernelWrapper( [] ()
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
  pmpl::genericKernelWrapper( [] ()
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
          EXPECT_EQ( array.linearIndex( a, b, c ), a * nb * nc + b * nc + c );
        }
      }
    }
  } );
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

  pmpl::genericKernelWrapper( [] ()
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
  pmpl::genericKernelWrapper( [] ()
  {
    TestCArrayHelper::Array const array{ initializer< TestCArrayHelper::Array >( std::make_integer_sequence< int, 2 * 3 * 4 >() ) };;
    int const na = array.extent<0>();
    int const nb = array.extent<1>();
    int const nc = array.extent<2>();

    double value = 0.0;
    for ( int a = 0; a < na; ++a )
    {
      for ( int b = 0; b < nb; ++b )
      {
        for ( int c = 0; c < nc; ++c )
        {
          EXPECT_DOUBLE_EQ( array( a, b, c ), value );
          value += 3.14159;
        }
      }
    }
  } );
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
