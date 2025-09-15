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


#include "../bases/LagrangeBasis.hpp"
#include "../spacing/Spacing.hpp"
#include "../../common/ShivaMacros.hpp"
#include "common/pmpl.hpp"
#include "common/types.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace shiva;
using namespace shiva::functions;

template< typename ... T >
struct TestBasisHelper;

template<>
struct TestBasisHelper< LagrangeBasis< double, 5, EqualSpacing > >
{
  using BasisType = LagrangeBasis< double, 5, EqualSpacing >;
  static constexpr int order = 5;
  static constexpr double coord = 0.3;
  static constexpr double refValues[order + 1] = {-0.0076904296875, 0.0555419921875, -0.199951171875, 0.999755859375, 0.1666259765625, -0.0142822265625};
  static constexpr double refGradients[order + 1] = {-0.064208984375, 0.44474283854167, -1.42333984375, -0.88134765625, 2.0747884114583, -0.150634765625};
};

template<>
struct TestBasisHelper< LagrangeBasis< double, 5, GaussLobattoSpacing > >
{
  using BasisType = LagrangeBasis< double, 5, GaussLobattoSpacing >;
  static constexpr int order = 5;
  static constexpr double coord = 0.3;
  static constexpr double refValues[order + 1] = {-0.0039331249999999, 0.011438606947084, -0.025205197129078, 0.99880774621932, 0.026196343962672, -0.0073043749999999};
  static constexpr double refGradients[order + 1] = {-0.26265625, 0.76193547139425, -1.6595367733139, -0.16178550663066, 1.8258868085503, -0.50384375};
};



template< typename BASIS_HELPER_TYPE >
SHIVA_GLOBAL void compileTimeKernel()
{
  using BasisType = typename BASIS_HELPER_TYPE::BasisType;
  constexpr int order = BASIS_HELPER_TYPE::order;
  constexpr double coord = BASIS_HELPER_TYPE::coord;

  forSequence< order + 1 >( [&] ( auto const BF_INDEX ) constexpr
  {
    constexpr double    value = BasisType::template value< BF_INDEX >( coord );
    constexpr double gradient = BasisType::template gradient< BF_INDEX >( coord );
    constexpr double tolerance = 1.0e-12;

    static_assert( pmpl::check( value, BASIS_HELPER_TYPE::refValues[BF_INDEX], tolerance ) );
    static_assert( pmpl::check( gradient, BASIS_HELPER_TYPE::refGradients[BF_INDEX], tolerance ) );
  } );
}

template< typename BASIS_HELPER_TYPE >
void testBasisAtCompileTime()
{
#if defined(SHIVA_USE_DEVICE)
  compileTimeKernel< BASIS_HELPER_TYPE ><< < 1, 1 >> > ();
#else
  compileTimeKernel< BASIS_HELPER_TYPE >();
#endif
}


template< typename BASIS_HELPER_TYPE >
SHIVA_GLOBAL void runTimeKernel( double * const values,
                                 double * const gradients )
{
  using BasisType = typename BASIS_HELPER_TYPE::BasisType;
  constexpr int order = BASIS_HELPER_TYPE::order;

  double coord = BASIS_HELPER_TYPE::coord;

  forSequence< order + 1 >( [&] ( auto const BF_INDEX ) constexpr
  {
    values[BF_INDEX]    = BasisType::template value< BF_INDEX >( coord );
    gradients[BF_INDEX] = BasisType::template gradient< BF_INDEX >( coord );
  } );
}

template< typename BASIS_HELPER_TYPE >
void testBasisAtRunTime()
{
  constexpr int order = BASIS_HELPER_TYPE::order;
  constexpr int N = order + 1;
#if defined(SHIVA_USE_DEVICE)
  constexpr int bytes = N * sizeof(double);
  double * values;
  double * gradients;
  deviceMallocManaged( &values, bytes );
  deviceMallocManaged( &gradients, bytes );
  runTimeKernel< BASIS_HELPER_TYPE ><< < 1, 1 >> > ( values, gradients );
  deviceDeviceSynchronize();
#else
  double values[N];
  double gradients[N];
  runTimeKernel< BASIS_HELPER_TYPE >( values, gradients );
#endif

  constexpr double tolerance = 1.0e-12;
  for ( int a = 0; a < N; ++a )
  {
    EXPECT_NEAR( values[a], BASIS_HELPER_TYPE::refValues[a], fabs( BASIS_HELPER_TYPE::refValues[a] * tolerance ) );
    EXPECT_NEAR( gradients[a], BASIS_HELPER_TYPE::refGradients[a], fabs( BASIS_HELPER_TYPE::refGradients[a] * tolerance ) );
  }

#if defined(SHIVA_USE_DEVICE)
  deviceFree( values );
  deviceFree( gradients );
#endif


}

TEST( testSpacing, testLagrangeBasisEqualSpacing )
{
  using BasisHelperType = TestBasisHelper< LagrangeBasis< double, 5, EqualSpacing > >;
  testBasisAtCompileTime< BasisHelperType >();
  testBasisAtRunTime< BasisHelperType >();
}

TEST( testSpacing, testLagrangeBasisGaussLobattoSpacing )
{
  using BasisHelperType = TestBasisHelper< LagrangeBasis< double, 5, GaussLobattoSpacing > >;
  testBasisAtCompileTime< BasisHelperType >();
  testBasisAtRunTime< BasisHelperType >();
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
