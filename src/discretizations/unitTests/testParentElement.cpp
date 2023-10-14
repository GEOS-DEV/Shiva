
#include "../finiteElementMethod/parentElements/ParentElement.hpp"
#include "../finiteElementMethod/bases/LagrangeBasis.hpp"
#include "../spacing/Spacing.hpp"
#include "../../geometry/Cube.hpp"
#include "../../common/ShivaMacros.hpp"



#include <gtest/gtest.h>
#include <cmath>

using namespace shiva;
using namespace shiva::discretizations::finiteElementMethod;
using namespace shiva::discretizations::finiteElementMethod::basis;
using namespace shiva::geometry;

#include "testParentElementSolutions.hpp"

constexpr bool check( double const a, double const b, double const tolerance )
{
  return ( a - b ) * ( a - b ) < tolerance * tolerance;
}


template< typename TEST_PARENT_ELEMENT_HELPER >
SHIVA_GLOBAL void compileTimeKernel()
{
  using ParentElementType = typename TEST_PARENT_ELEMENT_HELPER::ParentElementType;
  constexpr int order = TEST_PARENT_ELEMENT_HELPER::order;

  constexpr double coord[3] = { TEST_PARENT_ELEMENT_HELPER::testParentCoords[0],
                                TEST_PARENT_ELEMENT_HELPER::testParentCoords[1],
                                TEST_PARENT_ELEMENT_HELPER::testParentCoords[2] };

  forSequence< order + 1 >( [&] ( auto const a ) constexpr
  {
    forSequence< order + 1 >( [&] ( auto const b ) constexpr
    {
      forSequence< order + 1 >( [&] ( auto const c ) constexpr
      {
        constexpr double value = ParentElementType::template value< a, b, c >( coord );
        constexpr CArray1d< double, 3 > gradient = ParentElementType::template gradient< a, b, c >( coord );
        constexpr double tolerance = 1.0e-12;
        
        static_assert( check( value, TEST_PARENT_ELEMENT_HELPER::referenceValues[a][b][c], tolerance ) );
        static_assert( check( gradient[0], TEST_PARENT_ELEMENT_HELPER::referenceGradients[a][b][c][0], tolerance ) );
        static_assert( check( gradient[1], TEST_PARENT_ELEMENT_HELPER::referenceGradients[a][b][c][1], tolerance ) );
        static_assert( check( gradient[2], TEST_PARENT_ELEMENT_HELPER::referenceGradients[a][b][c][2], tolerance ) );
      } );
    } );
  } );
}
template< typename TEST_PARENT_ELEMENT_HELPER >
void testParentElementAtCompileTime()
{
#if defined(SHIVA_USE_DEVICE)
  compileTimeKernel<TEST_PARENT_ELEMENT_HELPER><<<1,1>>>();
#else
  compileTimeKernel<TEST_PARENT_ELEMENT_HELPER>();
#endif
}


template< typename TEST_PARENT_ELEMENT_HELPER >
SHIVA_GLOBAL void runTimeKernel()
{
  using ParentElementType = typename TEST_PARENT_ELEMENT_HELPER::ParentElementType;
  constexpr int order = TEST_PARENT_ELEMENT_HELPER::order;


  double const coord[3] = { TEST_PARENT_ELEMENT_HELPER::testParentCoords[0],
                            TEST_PARENT_ELEMENT_HELPER::testParentCoords[1],
                            TEST_PARENT_ELEMENT_HELPER::testParentCoords[2] };

  forSequence< order + 1 >( [&] ( auto const a ) constexpr
  {
    forSequence< order + 1 >( [&] ( auto const b ) constexpr
    {
      forSequence< order + 1 >( [&] ( auto const c ) constexpr
      {
        double const value = ParentElementType::template value< a, b, c >( coord );
        CArray1d< double, 3 > const gradient = ParentElementType::template gradient< a, b, c >( coord );
        constexpr double tolerance = 1.0e-12;

        EXPECT_NEAR( value, TEST_PARENT_ELEMENT_HELPER::referenceValues[a][b][c], fabs( value * tolerance ) );
        EXPECT_NEAR( gradient[0], TEST_PARENT_ELEMENT_HELPER::referenceGradients[a][b][c][0], fabs( gradient[0] * tolerance ) );
        EXPECT_NEAR( gradient[1], TEST_PARENT_ELEMENT_HELPER::referenceGradients[a][b][c][1], fabs( gradient[1] * tolerance ) );
        EXPECT_NEAR( gradient[2], TEST_PARENT_ELEMENT_HELPER::referenceGradients[a][b][c][2], fabs( gradient[2] * tolerance ) );
      } );
    } );
  } );
}

template< typename TEST_PARENT_ELEMENT_HELPER >
void testParentElementAtRunTime()
{
#if defined(SHIVA_USE_DEVICE)
  runTimeKernel<TEST_PARENT_ELEMENT_HELPER><<<1,1>>>();
#else
  runTimeKernel<TEST_PARENT_ELEMENT_HELPER>();
#endif
}



TEST( testParentElement, testBasicUsage )
{

  ParentElement< double,
                 Cube,
                 LagrangeBasis< double, 1, GaussLobattoSpacing >,
                 LagrangeBasis< double, 1, GaussLobattoSpacing >,
                 LagrangeBasis< double, 1, GaussLobattoSpacing >
                 >::template value< 1, 1, 1 >( {0.5, 0, 1} );
}

TEST( testParentElement, testCubeLagrangeBasisGaussLobatto_O1 )
{
  using ParentElementType = ParentElement< double,
                                           Cube,
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
                                           Cube,
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
