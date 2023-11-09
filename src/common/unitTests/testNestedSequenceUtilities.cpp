
#include "../NestedSequenceUtilities.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;

struct Data
{
  static constexpr int h[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  static constexpr int sum_of_h = 595;
  static constexpr int double_nested_to_8_sum = 8720;
  static constexpr int nested_sum_of_h = 354025;
};

template< typename FUNC >
SHIVA_GLOBAL void testSequenceExpansionHelper( FUNC func )
{
  func();
}

template< typename FUNC >
void kernelLaunch( FUNC && func )
{
#if defined(SHIVA_USE_DEVICE)
  testSequenceExpansionHelper << < 1, 1 >> > ( std::forward< FUNC >( func ) );
#else
  testSequenceExpansionHelper( std::forward< FUNC >( func ) );
#endif
}

void testForNestedSequenceLambdaHelper()
{
  kernelLaunch([] SHIVA_HOST_DEVICE ()
  {
    constexpr auto helper = [] ( auto const & h ) constexpr
    {
      int staticSum0 = 0;
      forNestedSequence< 10 > (
        [&] ( auto const a ) constexpr
      {
        staticSum0 += h[a];
      } );
      return staticSum0;
    };
    constexpr int staticSum0 = helper( Data::h );
    static_assert( staticSum0 == Data::sum_of_h );
  } );


  kernelLaunch([] SHIVA_HOST_DEVICE ()
  {
    constexpr auto helper = [] ( auto const & h ) constexpr
    {
      int staticSum0 = 0;
      forNestedSequence< 10, 8 > (
        [&] ( auto const a, auto const b ) constexpr
      {
        staticSum0 += h[a] + h[b];
      } );
      return staticSum0;
    };
    constexpr int staticSum0 = helper( Data::h );
    static_assert( staticSum0 == Data::double_nested_to_8_sum );
  } );
}

TEST( testNestedSequenceUtilities, testNestedForSequenceLambda )
{
  testForNestedSequenceLambdaHelper();
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
