
#include "../SequenceUtilities.hpp"

#include <gtest/gtest.h>

using namespace shiva;



TEST( testSequenceUtilities, testExecuteSequenceLambda )
{
  constexpr int h0[10] = {11,22,33,44,55,66,77,88,99,100};
  const int h1[10] = {11,22,33,44,55,66,77,88,99,100};
  constexpr int some_of_h = 595;

  constexpr int staticSum0 = executeSequence<10>( 
  [&]( auto const && ... a ) constexpr
  {
    return (h0[a] + ...);
  });
  static_assert( staticSum0 == some_of_h );

  int staticSum1 = executeSequence<10>( 
  [&]( auto const && ... a ) constexpr
  {
    return (h1[a] + ...);
  });
  EXPECT_EQ( staticSum1, some_of_h );
}

  struct FUNCTOR0
  {
    template< int ... INDICES >
    static constexpr int execute( int const (&h)[10] )
    {
      return (h[INDICES] + ...);
    }
  };

TEST( testSequenceUtilities, testExecuteSequenceFunctor )
{
  constexpr int h0[10] = {11,22,33,44,55,66,77,88,99,100};
  constexpr int sum_of_h = 595;

  constexpr int staticSum0 = executeSequence<10,FUNCTOR0>( h0 );
  static_assert( staticSum0 == sum_of_h );
}

#if __cplusplus >= 202002L
TEST( testSequenceUtilities, testExecuteSequenceTemplateLambda )
{
  constexpr int h0[10] = {11,22,33,44,55,66,77,88,99,100};
  const int h1[10] = {11,22,33,44,55,66,77,88,99,100};
  constexpr int some_of_h = 595;

  constexpr int staticSum0 = executeSequence<10>( 
  [&]<int ... a>() constexpr
  {
    return (h0[a] + ...);
  });
  static_assert( staticSum0 == some_of_h );

  int staticSum1 = executeSequence<10>( 
  [&]<int ... a>() constexpr
  {
    return (h1[a] + ...);
  });
  EXPECT_EQ( staticSum1, some_of_h );

}
#endif

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}