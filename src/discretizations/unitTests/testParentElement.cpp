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


#include "../finiteElementMethod/parentElements/ParentElement.hpp"
#include "functions/bases/LagrangeBasis.hpp"
#include "functions/spacing/Spacing.hpp"
#include "geometry/shapes/NCube.hpp"
#include "common/ShivaMacros.hpp"
#include "common/pmpl.hpp"



#include <gtest/gtest.h>
#include <cmath>

using namespace shiva;
using namespace shiva::geometry;
using namespace shiva::functions;
using namespace shiva::discretizations::finiteElementMethod;


template< typename ARRAY_TYPE, int ... INDICES >
static constexpr auto initializer( std::integer_sequence< int, INDICES ... >,
                                   double const (&init)[ sizeof ... ( INDICES ) ] )
{
  return ARRAY_TYPE{ init[INDICES] ... };
}


template< typename ... T >
struct TestParentElementHelper;

template<>
struct TestParentElementHelper< ParentElement< double,
                                               Cube< double >,
                                               LagrangeBasis< double, 1, GaussLobattoSpacing >,
                                               LagrangeBasis< double, 1, GaussLobattoSpacing >,
                                               LagrangeBasis< double, 1, GaussLobattoSpacing >
                                               > >
{
  using ParentElementType = ParentElement< double,
                                           Cube< double >,
                                           LagrangeBasis< double, 1, GaussLobattoSpacing >,
                                           LagrangeBasis< double, 1, GaussLobattoSpacing >,
                                           LagrangeBasis< double, 1, GaussLobattoSpacing >
                                           >;

  static constexpr int order = 1;
  static constexpr double testCoords[3] = { 0.31415, -0.161803, 0.69314 };
//  static constexpr double fieldValues0[2][2][2] = {{{0.75321698514839,0.043928500403754},{-0.82755314609174,-0.24417409139023}},{{-0.97671085406151,0.85453232668432},{0.087513533902136,-0.041336659055151}}};
  static constexpr CArrayNd< double, 2, 2, 2 > fieldValues{ 0.75321698514839, 0.043928500403754, -0.82755314609174, -0.24417409139023, -0.97671085406151, 0.85453232668432, 0.087513533902136,
                                                            -0.041336659055151};
  static constexpr double referenceValue = 0.19546114484888;
  static constexpr double referenceGradient[3] = {0.18762801054421, -0.27892876405183, 0.30302193042302};

};


template<>
struct TestParentElementHelper< ParentElement< double,
                                               Cube< double >,
                                               LagrangeBasis< double, 3, GaussLobattoSpacing >,
                                               LagrangeBasis< double, 3, GaussLobattoSpacing >,
                                               LagrangeBasis< double, 3, GaussLobattoSpacing >
                                               > >
{

  using ParentElementType = ParentElement< double,
                                           Cube< double >,
                                           LagrangeBasis< double, 3, GaussLobattoSpacing >,
                                           LagrangeBasis< double, 3, GaussLobattoSpacing >,
                                           LagrangeBasis< double, 3, GaussLobattoSpacing >
                                           >;

  static constexpr int order = 3;
  static constexpr double testCoords[3] = { 0.31415, -0.161803, 0.69314};
  // static constexpr double fieldValues0[4][4][4] = {{{0.75321698514839,0.043928500403754,-0.82755314609174,-0.24417409139023},{-0.97671085406151,0.85453232668432,0.087513533902136,-0.041336659055151},{-0.50930154443859,0.51979200312318,0.96998599508861,-0.56590975687332},{-0.081965626149824,0.76945833740991,0.1677085719803,-0.47205366499994}},
  //                                                 {{0.83912032472826,-0.15232955420577,0.97458062333262,0.17588531238907},{-0.83439021321014,0.58042926096295,0.39231786572069,0.50373246771709},{-0.19690154632734,0.26548251523734,0.64878944174577,-0.78892331571369},{-0.057476701102634,-0.38000100540515,0.56360464169372,-0.56726393971425}},
  //                                                 {{-0.57679390444802,0.75875590367842,-0.18369900352008,0.35735232771115},{0.71207507000379,-0.98436408586698,0.30360304171591,-0.3522210520638},{0.31575877583795,0.53395391736692,0.21200301642118,-0.94894741267982},{0.23239011510346,0.23202839452223,0.69368646162036,-0.21906790427127}},
  //                                                 {{0.87616166398538,-0.24685540701046,0.80172883854487,-0.75819705120112},{-0.54601148928802,0.97267584128641,0.62933378989992,-0.70069407632118},{0.81158931783762,-0.51941755373757,-0.59755364127609,-0.90505968109323},{0.39469344992863,-0.7569937332484,0.24030904537321,0.64377524701047}}};

  static constexpr CArrayNd< double, 4, 4, 4 > fieldValues{0.75321698514839, 0.043928500403754, -0.82755314609174, -0.24417409139023, -0.97671085406151, 0.85453232668432, 0.087513533902136,
                                                           -0.041336659055151, -0.50930154443859, 0.51979200312318, 0.96998599508861, -0.56590975687332, -0.081965626149824, 0.76945833740991,
                                                           0.1677085719803, -0.47205366499994,
                                                           0.83912032472826, -0.15232955420577, 0.97458062333262, 0.17588531238907, -0.83439021321014, 0.58042926096295, 0.39231786572069,
                                                           0.50373246771709, -0.19690154632734, 0.26548251523734, 0.64878944174577, -0.78892331571369, -0.057476701102634, -0.38000100540515,
                                                           0.56360464169372, -0.56726393971425,
                                                           -0.57679390444802, 0.75875590367842, -0.18369900352008, 0.35735232771115, 0.71207507000379, -0.98436408586698, 0.30360304171591,
                                                           -0.3522210520638, 0.31575877583795, 0.53395391736692, 0.21200301642118, -0.94894741267982, 0.23239011510346, 0.23202839452223,
                                                           0.69368646162036, -0.21906790427127,
                                                           0.87616166398538, -0.24685540701046, 0.80172883854487, -0.75819705120112, -0.54601148928802, 0.97267584128641, 0.62933378989992,
                                                           -0.70069407632118, 0.81158931783762, -0.51941755373757, -0.59755364127609, -0.90505968109323, 0.39469344992863, -0.7569937332484,
                                                           0.24030904537321, 0.64377524701047};
  static constexpr double referenceValue = 0.25230798990974;
  static constexpr double referenceGradient[3] = {-0.23132556090577, -0.71640960094542, -1.1388626399789};
};



template< typename TEST_PARENT_ELEMENT_HELPER >
SHIVA_GLOBAL void compileTimeKernel()
{
  using ParentElementType = typename TEST_PARENT_ELEMENT_HELPER::ParentElementType;

  constexpr double coord[3] = { TEST_PARENT_ELEMENT_HELPER::testCoords[0],
                                TEST_PARENT_ELEMENT_HELPER::testCoords[1],
                                TEST_PARENT_ELEMENT_HELPER::testCoords[2] };

  constexpr double value = ParentElementType::value( coord, TEST_PARENT_ELEMENT_HELPER::fieldValues );


  constexpr CArrayNd< double, 3 > gradient = ParentElementType::gradient( coord, TEST_PARENT_ELEMENT_HELPER::fieldValues );
  constexpr double tolerance = 1.0e-12;

  static_assert( pmpl::check( value, TEST_PARENT_ELEMENT_HELPER::referenceValue, tolerance ) );
  static_assert( pmpl::check( gradient( 0 ), TEST_PARENT_ELEMENT_HELPER::referenceGradient[0], tolerance ) );
  static_assert( pmpl::check( gradient( 1 ), TEST_PARENT_ELEMENT_HELPER::referenceGradient[1], tolerance ) );
  static_assert( pmpl::check( gradient( 2 ), TEST_PARENT_ELEMENT_HELPER::referenceGradient[2], tolerance ) );
}

template< typename TEST_PARENT_ELEMENT_HELPER >
void testParentElementAtCompileTime()
{
#if defined(SHIVA_USE_DEVICE)
  compileTimeKernel< TEST_PARENT_ELEMENT_HELPER ><< < 1, 1 >> > ();
#else
  compileTimeKernel< TEST_PARENT_ELEMENT_HELPER >();
#endif
}


template< typename TEST_PARENT_ELEMENT_HELPER, typename FV_TYPE >
SHIVA_GLOBAL void runTimeKernel( double * const value,
                                 double * const gradient,
                                 FV_TYPE const fieldValues )
{
  using ParentElementType = typename TEST_PARENT_ELEMENT_HELPER::ParentElementType;

  double const coord[3] = { TEST_PARENT_ELEMENT_HELPER::testCoords[0],
                            TEST_PARENT_ELEMENT_HELPER::testCoords[1],
                            TEST_PARENT_ELEMENT_HELPER::testCoords[2] };

  *value = ParentElementType::value( coord, fieldValues );
  CArrayNd< double, 3 > const temp = ParentElementType::gradient( coord, fieldValues );
  gradient[0] = temp( 0 );
  gradient[1] = temp( 1 );
  gradient[2] = temp( 2 );
}

template< typename TEST_PARENT_ELEMENT_HELPER >
void testParentElementAtRunTime()
{

  constexpr int N = TEST_PARENT_ELEMENT_HELPER::order + 1;
  CArrayNd< double, N, N, N > fieldValues;
  for ( int i = 0; i < TEST_PARENT_ELEMENT_HELPER::fieldValues.size(); ++i )
  {
    fieldValues.data()[i] = TEST_PARENT_ELEMENT_HELPER::fieldValues.data()[i];
  }

#if defined(SHIVA_USE_DEVICE)
  constexpr int bytes = sizeof(double);
  double * value;
  double * gradient;
  deviceMallocManaged( &value, bytes );
  deviceMallocManaged( &gradient, 3 * bytes );
  runTimeKernel< TEST_PARENT_ELEMENT_HELPER ><< < 1, 1 >> > ( value, gradient, fieldValues );
  deviceDeviceSynchronize();
#else
  double value[1];
  double gradient[3];
  runTimeKernel< TEST_PARENT_ELEMENT_HELPER >( value, gradient, fieldValues );
#endif

  constexpr double tolerance = 1.0e-12;
  EXPECT_NEAR( value[0], TEST_PARENT_ELEMENT_HELPER::referenceValue, fabs( TEST_PARENT_ELEMENT_HELPER::referenceValue * tolerance ) );
  EXPECT_NEAR( gradient[ 0 ], TEST_PARENT_ELEMENT_HELPER::referenceGradient[0],
               fabs( TEST_PARENT_ELEMENT_HELPER::referenceGradient[0] * tolerance ) );
  EXPECT_NEAR( gradient[ 1 ], TEST_PARENT_ELEMENT_HELPER::referenceGradient[1],
               fabs( TEST_PARENT_ELEMENT_HELPER::referenceGradient[1] * tolerance ) );
  EXPECT_NEAR( gradient[ 2 ], TEST_PARENT_ELEMENT_HELPER::referenceGradient[2],
               fabs( TEST_PARENT_ELEMENT_HELPER::referenceGradient[2] * tolerance ) );

}



TEST( testParentElement, testBasicUsage )
{

  ParentElement< double,
                 Cube< double >,
                 LagrangeBasis< double, 1, GaussLobattoSpacing >,
                 LagrangeBasis< double, 1, GaussLobattoSpacing >,
                 LagrangeBasis< double, 1, GaussLobattoSpacing >
                 >::template value< 1, 1, 1 >( {0.5, 0, 1} );
}

TEST( testParentElement, testCubeLagrangeBasisGaussLobatto_O1 )
{
  using ParentElementType = ParentElement< double,
                                           Cube< double >,
                                           LagrangeBasis< double, 1, GaussLobattoSpacing >,
                                           LagrangeBasis< double, 1, GaussLobattoSpacing >,
                                           LagrangeBasis< double, 1, GaussLobattoSpacing >
                                           >;
  using TEST_PARENT_ELEMENT_HELPER = TestParentElementHelper< ParentElementType >;

  testParentElementAtCompileTime< TEST_PARENT_ELEMENT_HELPER >();
  testParentElementAtRunTime< TEST_PARENT_ELEMENT_HELPER >();
}

TEST( testParentElement, testCubeLagrangeBasisGaussLobatto_O3 )
{
  using ParentElementType = ParentElement< double,
                                           Cube< double >,
                                           LagrangeBasis< double, 3, GaussLobattoSpacing >,
                                           LagrangeBasis< double, 3, GaussLobattoSpacing >,
                                           LagrangeBasis< double, 3, GaussLobattoSpacing >
                                           >;
  using TEST_PARENT_ELEMENT_HELPER = TestParentElementHelper< ParentElementType >;

  testParentElementAtCompileTime< TEST_PARENT_ELEMENT_HELPER >();
  testParentElementAtRunTime< TEST_PARENT_ELEMENT_HELPER >();
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
