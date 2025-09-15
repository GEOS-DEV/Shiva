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


#include "../NestedSequenceUtilities.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;

struct NestedData
{
  static constexpr int h[10] = {11, 22, 33, 44, 55, 66, 77, 88, 99, 100};
  static constexpr int sum_of_h = 595;
  static constexpr int double_nested_to_8_sum = 8720;
  static constexpr int quad_nested_evens_sum = 735360;
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
  kernelLaunch( [] SHIVA_HOST_DEVICE ()
  {
    constexpr auto helper = [] ( auto const & h ) constexpr
    {
      int staticSum0 = 0;
      forNestedSequence< 10 >(
        [&] ( auto const a ) constexpr
      {
        staticSum0 += h[a];
      } );
      return staticSum0;
    };
    constexpr int staticSum0 = helper( NestedData::h );
    static_assert( staticSum0 == NestedData::sum_of_h );
  } );


  kernelLaunch( [] SHIVA_HOST_DEVICE ()
  {
    constexpr auto helper = [] ( auto const & h ) constexpr
    {
      int staticSum0 = 0;
      forNestedSequence< 10, 8 >(
        [&] ( auto const a, auto const b ) constexpr
      {
        staticSum0 += h[a] + h[b];
      } );
      return staticSum0;
    };
    constexpr int staticSum0 = helper( NestedData::h );
    static_assert( staticSum0 == NestedData::double_nested_to_8_sum );
  } );

  kernelLaunch( [] SHIVA_HOST_DEVICE ()
  {
    constexpr auto helper = [] ( auto const & h ) constexpr
    {
      int staticSum0 = 0;
      forNestedSequence< 10, 8, 6, 4, 2 >(
        [&] ( auto const a, auto const b, auto const c, auto const d, auto const e ) constexpr
      {
        staticSum0 += h[a] + h[b] + h[c] + h[d] + h[e];
      } );
      return staticSum0;
    };
    constexpr int staticSum0 = helper( NestedData::h );
    static_assert( staticSum0 == NestedData::quad_nested_evens_sum );
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
