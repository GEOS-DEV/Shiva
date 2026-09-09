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


#include "shiva/functions/bases/LagrangeBasis.hpp"
#include "shiva/functions/bases/BasisProduct.hpp"
#include "shiva/functions/spacing/Spacing.hpp"
#include "shiva/common/ShivaMacros.hpp"
#include "shiva/common/pmpl.hpp"
#include "shiva/common/types.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <limits>

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
#if defined(SHIVA_ENABLE_DEVICE)
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
#if defined(SHIVA_ENABLE_DEVICE)
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

#if defined(SHIVA_ENABLE_DEVICE)
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

// Closed-form values and derivatives for the constant, linear, and quadratic
// bases on [-1, 1]. Unused entries pad the lower-order rows with zeros.
template< typename REAL_TYPE, int ORDER >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE CArrayNd< REAL_TYPE, 2, 3 >
lowOrderReference( REAL_TYPE const x )
{
  constexpr REAL_TYPE zero = 0;
  constexpr REAL_TYPE one = 1;
  constexpr REAL_TYPE half = 0.5;
  if constexpr ( ORDER == 0 )
  {
    return { one, zero, zero, zero, zero, zero };
  }
  else if constexpr ( ORDER == 1 )
  {
    return { (1 - x) / 2, (1 + x) / 2, zero, -half, half, zero };
  }
  else
  {
    return { x * (x - 1) / 2, 1 - x * x, x * (x + 1) / 2,
             x - half, -2 * x, x + half };
  }
}

template< typename REAL_TYPE, int ORDER, template< typename, int > typename SPACING_TYPE >
void testLowOrderBasis()
{
  using BasisType = LagrangeBasis< REAL_TYPE, ORDER, SPACING_TYPE >;
  constexpr int numPoints = 4;
  constexpr int stride = 1 + 2 * BasisType::numSupportPoints;
  constexpr REAL_TYPE coords[numPoints] = { -1, 0, REAL_TYPE( 0.3 ), 1 };
  constexpr REAL_TYPE tolerance = 16 * std::numeric_limits< REAL_TYPE >::epsilon();
  REAL_TYPE data[numPoints * stride]{};
  for ( int point = 0; point < numPoints; ++point )
  {
    data[point * stride] = coords[point];
  }

  pmpl::genericKernelWrapper( numPoints * stride, data, [] SHIVA_HOST_DEVICE ( REAL_TYPE * const kernelData )
  {
    forSequence< numPoints >( [] ( auto const POINT ) constexpr
    {
      constexpr REAL_TYPE compileTimeCoords[numPoints] = { -1, 0, REAL_TYPE( 0.3 ), 1 };
      constexpr REAL_TYPE coord = compileTimeCoords[POINT];
      constexpr auto expected = lowOrderReference< REAL_TYPE, ORDER >( coord );
      forSequence< BasisType::numSupportPoints >( [&] ( auto const BF_INDEX ) constexpr
      {
        constexpr REAL_TYPE value = BasisType::template value< BF_INDEX >( coord );
        constexpr REAL_TYPE gradient = BasisType::template gradient< BF_INDEX >( coord );
        static_assert( pmpl::check( value, expected( 0, BF_INDEX ), tolerance ) );
        static_assert( pmpl::check( gradient, expected( 1, BF_INDEX ), tolerance ) );
      } );
    } );

    for ( int point = 0; point < numPoints; ++point )
    {
      int const offset = point * stride;
      REAL_TYPE const coord = kernelData[offset];
      forSequence< BasisType::numSupportPoints >( [&] ( auto const BF_INDEX )
      {
        kernelData[offset + 1 + BF_INDEX] = BasisType::template value< BF_INDEX >( coord );
        kernelData[offset + 1 + BasisType::numSupportPoints + BF_INDEX] = BasisType::template gradient< BF_INDEX >( coord );
      } );
    }
  } );

  for ( int point = 0; point < numPoints; ++point )
  {
    auto const expected = lowOrderReference< REAL_TYPE, ORDER >( coords[point] );
    for ( int basis = 0; basis < BasisType::numSupportPoints; ++basis )
    {
      EXPECT_NEAR( data[point * stride + 1 + basis], expected( 0, basis ), tolerance );
      EXPECT_NEAR( data[point * stride + 1 + BasisType::numSupportPoints + basis], expected( 1, basis ), tolerance );
    }
  }
}

TEST( testLagrangeBasis, lowOrderEqualSpacing )
{
  testLowOrderBasis< float, 0, EqualSpacing >();
  testLowOrderBasis< double, 0, EqualSpacing >();
  testLowOrderBasis< float, 1, EqualSpacing >();
  testLowOrderBasis< double, 1, EqualSpacing >();
  testLowOrderBasis< float, 2, EqualSpacing >();
  testLowOrderBasis< double, 2, EqualSpacing >();
}

TEST( testLagrangeBasis, lowOrderGaussLobattoSpacing )
{
  testLowOrderBasis< float, 1, GaussLobattoSpacing >();
  testLowOrderBasis< double, 1, GaussLobattoSpacing >();
  testLowOrderBasis< float, 2, GaussLobattoSpacing >();
  testLowOrderBasis< double, 2, GaussLobattoSpacing >();
}

template< typename REAL_TYPE >
void testAnisotropicBasisProduct()
{
  using Product = BasisProduct< REAL_TYPE,
                                LagrangeBasis< REAL_TYPE, 1, EqualSpacing >,
                                LagrangeBasis< REAL_TYPE, 2, GaussLobattoSpacing > >;
  constexpr REAL_TYPE tolerance = std::numeric_limits< REAL_TYPE >::epsilon();
  REAL_TYPE data[3] = { REAL_TYPE( 0.5 ), REAL_TYPE( -0.25 ), 0 };
  pmpl::genericKernelWrapper( 3, data, [] SHIVA_HOST_DEVICE ( REAL_TYPE * const kernelData )
  {
    // Phi_01(x,y) = (1-x)(1-y*y)/2.
    // Distinct magnitudes ensure reusing a coordinate changes the basis value.
    constexpr REAL_TYPE coord[2] = { REAL_TYPE( 0.5 ), REAL_TYPE( -0.25 ) };
    constexpr auto value = Product::template value< 0, 1 >( coord );
    constexpr auto gradient = Product::template gradient< 0, 1 >( coord );
    static_assert( pmpl::check( value, REAL_TYPE( 0.234375 ), tolerance ) );
    static_assert( pmpl::check( gradient( 0 ), REAL_TYPE( -0.46875 ), tolerance ) );
    static_assert( pmpl::check( gradient( 1 ), REAL_TYPE( 0.125 ), tolerance ) );

    REAL_TYPE const runtimeCoord[2] = { kernelData[0], kernelData[1] };
    auto const runtimeGradient = Product::template gradient< 0, 1 >( runtimeCoord );
    kernelData[0] = Product::template value< 0, 1 >( runtimeCoord );
    kernelData[1] = runtimeGradient( 0 );
    kernelData[2] = runtimeGradient( 1 );
  } );

  EXPECT_NEAR( data[0], REAL_TYPE( 0.234375 ), tolerance );
  EXPECT_NEAR( data[1], REAL_TYPE( -0.46875 ), tolerance );
  EXPECT_NEAR( data[2], REAL_TYPE( 0.125 ), tolerance );
}

TEST( testLagrangeBasis, anisotropicBasisProduct )
{
  testAnisotropicBasisProduct< float >();
  testAnisotropicBasisProduct< double >();
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
