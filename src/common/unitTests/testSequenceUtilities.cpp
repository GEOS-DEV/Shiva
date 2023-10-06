
#include "../SequenceUtilities.hpp"

#include <gtest/gtest.h>

using namespace shiva;



TEST( testSequenceUtilities, testSequenceExpansionLambda )
{
  constexpr int h0[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  const int h1[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  constexpr int sum_of_h = 595;

  constexpr int staticSum0 = executeSequence< 10 >(
    [&] ( auto const && ... a ) constexpr
  {
    return (h0[a] + ...);
  } );
  static_assert( staticSum0 == sum_of_h );

  int staticSum1 = executeSequence< 10 >(
    [&] ( auto const && ... a ) constexpr
  {
    return (h1[a] + ...);
  } );
  EXPECT_EQ( staticSum1, sum_of_h );
}

TEST( testSequenceUtilities, testNestedSequenceExpansionLambda )
{
  constexpr int h0[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  const int h1[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  constexpr int sum_of_h = 354025;

  constexpr int staticSum0 = executeSequence< 10 >(
    [&]< int ... a > () constexpr
  {
    return
      ( executeSequence< 10 >
        (
          [ h = h0, aa = std::integral_constant< int, a >{} ] ( auto const ... b ) constexpr
    { return ( (h[aa] * h[b]) + ...); }
        ) + ...
      );
  } );
  static_assert( staticSum0 == sum_of_h );

  int const staticSum1 = executeSequence< 10 >(
    [&]< int ... a > () constexpr
  {
    return
      ( executeSequence< 10 >
        (
          [ h = h1, aa = std::integral_constant< int, a >{} ] ( auto const ... b ) constexpr
    { return ( (h[aa] * h[b]) + ...); }
        ) + ...
      );
  } );
  EXPECT_EQ( staticSum1, sum_of_h );
}

TEST( testSequenceUtilities, testForSequenceLambda )
{
  constexpr int h0[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  const int h1[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  constexpr int sum_of_h = 595;

  constexpr auto helper = [] ( auto const & h ) constexpr
  {
    int staticSum0 = 0;
    forSequence< 10 >(
      [&] ( auto const a ) constexpr
    {
      staticSum0 += h[a];
    } );
    return staticSum0;
  };

  constexpr int staticSum0 = helper( h0 );
  static_assert( staticSum0 == sum_of_h );

  int const staticSum1 = helper( h1 );
  EXPECT_EQ( staticSum1, sum_of_h );
}



#if __cplusplus >= 202002L
TEST( testSequenceUtilities, testSequenceExpansionTemplateLambda )
{
  constexpr int h0[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  const int h1[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  constexpr int sum_of_h = 595;

  constexpr int staticSum0 = executeSequence< 10 >(
    [&]< int ... a > () constexpr
  {
    return (h0[a] + ...);
  } );
  static_assert( staticSum0 == sum_of_h );

  int staticSum1 = executeSequence< 10 >(
    [&]< int ... a > () constexpr
  {
    return (h1[a] + ...);
  } );
  EXPECT_EQ( staticSum1, sum_of_h );
}


TEST( testSequenceUtilities, testNestedSequenceExpansionTemplateLambda )
{
  constexpr int h0[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  const int h1[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  constexpr int sum_of_h = 354025;

  constexpr int staticSum0 = executeSequence< 10 >(
    [&]< int ... a > () constexpr
  {
    return
      ( executeSequence< 10 >
        (
          [ h = h0, aa = std::integral_constant< int, a >{} ]< int ... b > () constexpr
    { return ( (h[aa] * h[b]) + ...); }
        ) + ...
      );
  } );
  static_assert( staticSum0 == sum_of_h );

  int const staticSum1 = executeSequence< 10 >(
    [&]< int ... a > () constexpr
  {
    return
      ( executeSequence< 10 >
        (
          [ h = h1, aa = std::integral_constant< int, a >{} ]< int ... b > () constexpr
    { return ( (h[aa] * h[b]) + ...); }
        ) + ...
      );
  } );
  EXPECT_EQ( staticSum1, sum_of_h );
}

TEST( testSequenceUtilities, testForSequenceTemplateLambda )
{
  constexpr int h0[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  const int h1[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  constexpr int sum_of_h = 595;

  constexpr auto helper = [] ( auto const & h ) constexpr
  {
    int staticSum0 = 0;
    forSequence< 10 >(
      [&]< int a > () constexpr
    {
      staticSum0 += h[a];
    } );
    return staticSum0;
  };

  constexpr int staticSum0 = helper( h0 );
  static_assert( staticSum0 == sum_of_h );

  int const staticSum1 = helper( h1 );
  EXPECT_EQ( staticSum1, sum_of_h );
}
#endif


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
