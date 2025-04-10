/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2023  Lawrence Livermore National Security LLC
 * Copyright (c) 2023  TotalEnergies
 * Copyright (c) 2023- Shiva Contributors
 * All rights reserved
 *
 * See Shiva/LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */


#include "../SequenceUtilities.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;

struct Data
{
  static constexpr int h[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  static constexpr int sum_of_h = 595;
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

void testSequenceExpansionLambdaHelper()
{
  kernelLaunch( [] SHIVA_HOST_DEVICE ()
  {
    constexpr int staticSum0 =
      executeSequence< 10 >( [&] ( auto const && ... a ) constexpr
    {
      return (Data::h[a] + ...);
    } );
    static_assert( staticSum0 == Data::sum_of_h );
  } );
}

TEST( testSequenceUtilities, testSequenceExpansionLambda )
{
  testSequenceExpansionLambdaHelper();
}



void testNestedSequenceExpansionLambdaHelper()
{
  kernelLaunch( [] SHIVA_HOST_DEVICE ()
  {
    constexpr int staticSum0 =
      executeSequence< 10 >( [&] ( auto const ... a ) constexpr
    {
      return
        ( executeSequence< 10 >
            ( [ h = Data::h, aa = std::integral_constant< int, a >{} ] ( auto const ... b ) constexpr
            { return ( (h[aa] * h[b]) + ...); }
            ) + ...
        );
    } );
    static_assert( staticSum0 == Data::nested_sum_of_h );
  } );
}

TEST( testSequenceUtilities, testNestedSequenceExpansionLambda )
{
  testNestedSequenceExpansionLambdaHelper();
}


void testForSequenceLambdaHelper()
{
  kernelLaunch( [] SHIVA_HOST_DEVICE ()
  {
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
    constexpr int staticSum0 = helper( Data::h );
    static_assert( staticSum0 == Data::sum_of_h );
  } );
}

TEST( testSequenceUtilities, testForSequenceLambda )
{
  testForSequenceLambdaHelper();
}



#if __cplusplus >= 202002L

void testSequenceExpansionTemplateLambdaHelper()
{
  kernelLaunch( [] SHIVA_HOST_DEVICE ()
  {
    constexpr int staticSum0 =
      executeSequence< 10 >( [&] < int ... a > () constexpr
    {
      return (Data::h[a] + ...);
    } );
    static_assert( staticSum0 == Data::sum_of_h );
  } );
}


TEST( testSequenceUtilities, testSequenceExpansionTemplateLambda )
{
  testSequenceExpansionTemplateLambdaHelper();
}



void testSequenceExpansionTemplateLambdaHelper()
{
  kernelLaunch( [] SHIVA_HOST_DEVICE ()
  {
    constexpr int staticSum0 = executeSequence< 10 >( [&] < int ... a > () constexpr
    {
      return
        ( executeSequence< 10 >
          (
            [ h = Data::h, aa = std::integral_constant< int, a >{} ] < int ... b > () constexpr
            { return ( (h[aa] * h[b]) + ...); }
          ) + ...
        );
    } );
    static_assert( staticSum0 == Data::nested_sum_of_h );
  } );
}

TEST( testSequenceUtilities, testNestedSequenceExpansionTemplateLambda )
{
  testSequenceExpansionTemplateLambdaHelper();
}


void testForSequenceTemplateLambdaHelper()
{
  kernelLaunch( [] SHIVA_HOST_DEVICE ()
  {
    constexpr auto helper = [] ( auto const & h ) constexpr
    {
      int staticSum0 = 0;
      forSequence< 10 >(
        [&] < int a > () constexpr
      {
        staticSum0 += h[a];
      } );
      return staticSum0;
    };

    constexpr int staticSum0 = helper( Data::h );
    static_assert( staticSum0 == Data::sum_of_h );
  } );
}

TEST( testSequenceUtilities, testForSequenceTemplateLambda )
{
  testForSequenceTemplateLambdaHelper();
}
#endif


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
