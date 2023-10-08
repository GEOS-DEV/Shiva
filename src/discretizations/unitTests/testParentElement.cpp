
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
  return ( a - b )*( a - b ) < tolerance*tolerance;
}


template< typename TestParentElementHelperType >
constexpr void compileTimeCheck()
{
  using ParentElementType = typename TestParentElementHelperType::ParentElementType;
  constexpr int order = TestParentElementHelperType::order;

  forSequence< order + 2 >( [&] ( auto const i ) constexpr
  {
    forSequence< order + 2 >( [&] ( auto const j ) constexpr
    {
      forSequence< order + 2 >( [&] ( auto const k ) constexpr
      {
        constexpr double coord[3] = { TestParentElementHelperType::testParentCoords[i], 
                                      TestParentElementHelperType::testParentCoords[j], 
                                      TestParentElementHelperType::testParentCoords[k] };

        forSequence< order + 1 >( [&] ( auto const a ) constexpr
        {
          forSequence< order + 1 >( [&] ( auto const b ) constexpr
          {
            forSequence< order + 1 >( [&] ( auto const c ) constexpr
            {
              constexpr double value = ParentElementType::template value< a, b, c >( coord );
              constexpr CArray1d< double, 3 > gradient = ParentElementType::template gradient< a, b, c >( coord );
              constexpr double tolerance = 1.0e-12;

              static_assert( check( value,           TestParentElementHelperType::referenceValues[i][j][k][a][b][c], tolerance ) );
              static_assert( check( gradient[0],  TestParentElementHelperType::referenceGradients[i][j][k][a][b][c][0], tolerance ) );
              static_assert( check( gradient[1],  TestParentElementHelperType::referenceGradients[i][j][k][a][b][c][1], tolerance ) );
              static_assert( check( gradient[2],  TestParentElementHelperType::referenceGradients[i][j][k][a][b][c][2], tolerance ) );
            } );
          } );
        } );

      } );
    } );
  } );
}


template< typename TestParentElementHelperType >
constexpr void runTimeCheck()
{
  using ParentElementType = typename TestParentElementHelperType::ParentElementType;
  constexpr int order = TestParentElementHelperType::order;

  for ( int i = 0; i < order + 2; ++i )
  {
    for ( int j = 0; j < order + 2; ++j )
    {
      for ( int k = 0; k < order + 2; ++k )
      {
       double const coord[3] = { TestParentElementHelperType::testParentCoords[i], 
                                 TestParentElementHelperType::testParentCoords[j], 
                                 TestParentElementHelperType::testParentCoords[k] };

        forSequence< order + 1 >( [&] ( auto const a ) constexpr
        {
          forSequence< order + 1 >( [&] ( auto const b ) constexpr
          {
            forSequence< order + 1 >( [&] ( auto const c ) constexpr
            {
              double const value = ParentElementType::template value< a, b, c >( coord );
              CArray1d< double, 3 > const gradient = ParentElementType::template gradient< a, b, c >( coord );
              constexpr double tolerance = 1.0e-12;

              EXPECT_NEAR( value,             TestParentElementHelperType::referenceValues[i][j][k][a][b][c], fabs(value*tolerance) );
              EXPECT_NEAR( gradient[0], TestParentElementHelperType::referenceGradients[i][j][k][a][b][c][0], fabs(gradient[0]*tolerance) );
              EXPECT_NEAR( gradient[1], TestParentElementHelperType::referenceGradients[i][j][k][a][b][c][1], fabs(gradient[1]*tolerance) );
              EXPECT_NEAR( gradient[2], TestParentElementHelperType::referenceGradients[i][j][k][a][b][c][2], fabs(gradient[2]*tolerance) );
            });
          });
        });

      }
    }
  }
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
  using TestParentElementHelperType = TestParentElementHelper< ParentElementType >;

  compileTimeCheck<TestParentElementHelperType>();
  runTimeCheck<TestParentElementHelperType>();
}

TEST( testParentElement, testCubeLagrangeBasisGaussLobatto_O3 )
{
  using ParentElementType = ParentElement< double,
                                           Cube,
                                           LagrangeBasis< double, 3, GaussLobattoSpacing >,
                                           LagrangeBasis< double, 3, GaussLobattoSpacing >,
                                           LagrangeBasis< double, 3, GaussLobattoSpacing >
                                           >;
  using TestParentElementHelperType = TestParentElementHelper< ParentElementType >;

  compileTimeCheck<TestParentElementHelperType>();
  runTimeCheck<TestParentElementHelperType>();
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
