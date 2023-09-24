
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


template <typename... INTEGER_SEQUENCES>
struct recursiveSequenceExpansionHelper
{};
template < typename PARENT_ELEMENT_TYPE, int... INDICES, typename... INTEGER_SEQUENCES>
struct recursiveSequenceExpansionHelper<PARENT_ELEMENT_TYPE, std::integer_sequence<int, INDICES...>, INTEGER_SEQUENCES...> 
{
    template< int ... PERMUTATION, typename REFERENCE_VALUE_TYPE, typename REFERENCE_GRADIENT_TYPE >
    static void value( double const (&coord)[3],
                       REFERENCE_VALUE_TYPE const & referenceValues,
                       REFERENCE_GRADIENT_TYPE const & referenceGradients )
    {
        if constexpr (sizeof...(INTEGER_SEQUENCES)==0)
        {
          double values[sizeof...(INDICES)] = { PARENT_ELEMENT_TYPE::template value<PERMUTATION...,INDICES>(coord)... };

          CArray1d<double,3> gradients[sizeof...(INDICES)] = { PARENT_ELEMENT_TYPE::template gradient<PERMUTATION...,INDICES>(coord)... };

          for( size_t a=0; a<sizeof...(INDICES); ++a )
          {
            EXPECT_NEAR( values[a], referenceValues[a], abs(referenceValues[a]) * 1e-13 );
            for( int i=0; i<3; ++i )
            {
              EXPECT_NEAR( gradients[a].data[i], referenceGradients[a][i], abs(referenceGradients[a][i]) * 1e-13 );
            }
          }
        }
        else
        {
           (recursiveSequenceExpansionHelper< PARENT_ELEMENT_TYPE, 
                                              INTEGER_SEQUENCES... >::template value< PERMUTATION..., 
                                                                                      INDICES>( coord, 
                                                                                                referenceValues[INDICES],
                                                                                                referenceGradients[INDICES] ), ... );
        }
    }
};

template< typename PARENT_ELEMENT_TYPE, typename ... INTEGER_SEQUENCES, typename REFERENCE_VALUE_TYPE, typename REFERENCE_GRADIENT_TYPE >
void recursiveSequenceExpansion( double const (&coord)[3],
                                 REFERENCE_VALUE_TYPE const & referenceValues,
                                 REFERENCE_GRADIENT_TYPE const & referenceGradients )
{
    recursiveSequenceExpansionHelper< PARENT_ELEMENT_TYPE, INTEGER_SEQUENCES...>::value( coord, referenceValues, referenceGradients );
}


template< typename REAL_TYPE, int ORDER, typename PARENT_ELEMENT_TYPE >
void testParentElementCT( REAL_TYPE const (&testParentCoords)[ORDER+2],
                          REAL_TYPE const (&referenceValues)[ORDER+2][ORDER+2][ORDER+2][ORDER+1][ORDER+1][ORDER+1],
                          REAL_TYPE const (&referenceGradient)[ORDER+2][ORDER+2][ORDER+2][ORDER+1][ORDER+1][ORDER+1][3] )
{

    for( int i=0; i<ORDER+2; ++i )
    {
      for( int j=0; j<ORDER+2; ++j )
      {
        for( int k=0; k<ORDER+2; ++k )
        {
          REAL_TYPE const coord[3] = { testParentCoords[i], testParentCoords[j], testParentCoords[k] };
          recursiveSequenceExpansion< PARENT_ELEMENT_TYPE,
                                      std::make_integer_sequence< int, ORDER+1 >,
                                      std::make_integer_sequence< int, ORDER+1 >,
                                      std::make_integer_sequence< int, ORDER+1 > >( coord,
                                                                                    referenceValues[i][j][k],
                                                                                    referenceGradient[i][j][k] );
        }
      }
    }
}


TEST( testParentElement, testBasicUsage )
{

  ParentElement< double, 
                 Cube, 
                 LagrangeBasis<double,1,GaussLobattoSpacing>, 
                 LagrangeBasis<double,1,GaussLobattoSpacing>, 
                 LagrangeBasis<double,1,GaussLobattoSpacing>
               >::template value<1,1,1>({0.5,0,1});
}

TEST( testParentElement, testCubeLagrangeBasisGaussLobattoL_order1 )
{
  using ParentElementType = ParentElement< double, 
                 Cube, 
                 LagrangeBasis<double,1,GaussLobattoSpacing>, 
                 LagrangeBasis<double,1,GaussLobattoSpacing>, 
                 LagrangeBasis<double,1,GaussLobattoSpacing>
               >;

  double const testParentCoords[3] = { -1.0, 0.0, 1.0 };
  double const referenceValues[3][3][3][2][2][2] = {{{{{{1.,0},{0,0}},{{0,0},{0,0}}},{{{0.5,0.5},{0,0}},{{0,0},{0,0}}},{{{0,1.},{0,0}},{{0,0},{0,0}}}},{{{{0.5,0},{0.5,0}},{{0,0},{0,0}}},{{{0.25,0.25},{0.25,0.25}},{{0,0},{0,0}}},{{{0,0.5},{0,0.5}},{{0,0},{0,0}}}},{{{{0,0},{1.,0}},{{0,0},{0,0}}},{{{0,0},{0.5,0.5}},{{0,0},{0,0}}},{{{0,0},{0,1.}},{{0,0},{0,0}}}}},{{{{{0.5,0},{0,0}},{{0.5,0},{0,0}}},{{{0.25,0.25},{0,0}},{{0.25,0.25},{0,0}}},{{{0,0.5},{0,0}},{{0,0.5},{0,0}}}},{{{{0.25,0},{0.25,0}},{{0.25,0},{0.25,0}}},{{{0.125,0.125},{0.125,0.125}},{{0.125,0.125},{0.125,0.125}}},{{{0,0.25},{0,0.25}},{{0,0.25},{0,0.25}}}},{{{{0,0},{0.5,0}},{{0,0},{0.5,0}}},{{{0,0},{0.25,0.25}},{{0,0},{0.25,0.25}}},{{{0,0},{0,0.5}},{{0,0},{0,0.5}}}}},{{{{{0,0},{0,0}},{{1.,0},{0,0}}},{{{0,0},{0,0}},{{0.5,0.5},{0,0}}},{{{0,0},{0,0}},{{0,1.},{0,0}}}},{{{{0,0},{0,0}},{{0.5,0},{0.5,0}}},{{{0,0},{0,0}},{{0.25,0.25},{0.25,0.25}}},{{{0,0},{0,0}},{{0,0.5},{0,0.5}}}},{{{{0,0},{0,0}},{{0,0},{1.,0}}},{{{0,0},{0,0}},{{0,0},{0.5,0.5}}},{{{0,0},{0,0}},{{0,0},{0,1.}}}}}};
  double const referenceGradients[3][3][3][2][2][2][3] = {{{{{{{-0.5,-0.5,-0.5},{0,0,0.5}},{{0,0.5,0},{0,0,0}}},{{{0.5,0,0},{0,0,0}},{{0,0,0},{0,0,0}}}},{{{{-0.25,-0.25,-0.5},{-0.25,-0.25,0.5}},{{0,0.25,0},{0,0.25,0}}},{{{0.25,0,0},{0.25,0,0}},{{0,0,0},{0,0,0}}}},{{{{0,0,-0.5},{-0.5,-0.5,0.5}},{{0,0,0},{0,0.5,0}}},{{{0,0,0},{0.5,0,0}},{{0,0,0},{0,0,0}}}}},{{{{{-0.25,-0.5,-0.25},{0,0,0.25}},{{-0.25,0.5,-0.25},{0,0,0.25}}},{{{0.25,0,0},{0,0,0}},{{0.25,0,0},{0,0,0}}}},{{{{-0.125,-0.25,-0.25},{-0.125,-0.25,0.25}},{{-0.125,0.25,-0.25},{-0.125,0.25,0.25}}},{{{0.125,0,0},{0.125,0,0}},{{0.125,0,0},{0.125,0,0}}}},{{{{0,0,-0.25},{-0.25,-0.5,0.25}},{{0,0,-0.25},{-0.25,0.5,0.25}}},{{{0,0,0},{0.25,0,0}},{{0,0,0},{0.25,0,0}}}}},{{{{{0,-0.5,0},{0,0,0}},{{-0.5,0.5,-0.5},{0,0,0.5}}},{{{0,0,0},{0,0,0}},{{0.5,0,0},{0,0,0}}}},{{{{0,-0.25,0},{0,-0.25,0}},{{-0.25,0.25,-0.5},{-0.25,0.25,0.5}}},{{{0,0,0},{0,0,0}},{{0.25,0,0},{0.25,0,0}}}},{{{{0,0,0},{0,-0.5,0}},{{0,0,-0.5},{-0.5,0.5,0.5}}},{{{0,0,0},{0,0,0}},{{0,0,0},{0.5,0,0}}}}}},{{{{{{-0.5,-0.25,-0.25},{0,0,0.25}},{{0,0.25,0},{0,0,0}}},{{{0.5,-0.25,-0.25},{0,0,0.25}},{{0,0.25,0},{0,0,0}}}},{{{{-0.25,-0.125,-0.25},{-0.25,-0.125,0.25}},{{0,0.125,0},{0,0.125,0}}},{{{0.25,-0.125,-0.25},{0.25,-0.125,0.25}},{{0,0.125,0},{0,0.125,0}}}},{{{{0,0,-0.25},{-0.5,-0.25,0.25}},{{0,0,0},{0,0.25,0}}},{{{0,0,-0.25},{0.5,-0.25,0.25}},{{0,0,0},{0,0.25,0}}}}},{{{{{-0.25,-0.25,-0.125},{0,0,0.125}},{{-0.25,0.25,-0.125},{0,0,0.125}}},{{{0.25,-0.25,-0.125},{0,0,0.125}},{{0.25,0.25,-0.125},{0,0,0.125}}}},{{{{-0.125,-0.125,-0.125},{-0.125,-0.125,0.125}},{{-0.125,0.125,-0.125},{-0.125,0.125,0.125}}},{{{0.125,-0.125,-0.125},{0.125,-0.125,0.125}},{{0.125,0.125,-0.125},{0.125,0.125,0.125}}}},{{{{0,0,-0.125},{-0.25,-0.25,0.125}},{{0,0,-0.125},{-0.25,0.25,0.125}}},{{{0,0,-0.125},{0.25,-0.25,0.125}},{{0,0,-0.125},{0.25,0.25,0.125}}}}},{{{{{0,-0.25,0},{0,0,0}},{{-0.5,0.25,-0.25},{0,0,0.25}}},{{{0,-0.25,0},{0,0,0}},{{0.5,0.25,-0.25},{0,0,0.25}}}},{{{{0,-0.125,0},{0,-0.125,0}},{{-0.25,0.125,-0.25},{-0.25,0.125,0.25}}},{{{0,-0.125,0},{0,-0.125,0}},{{0.25,0.125,-0.25},{0.25,0.125,0.25}}}},{{{{0,0,0},{0,-0.25,0}},{{0,0,-0.25},{-0.5,0.25,0.25}}},{{{0,0,0},{0,-0.25,0}},{{0,0,-0.25},{0.5,0.25,0.25}}}}}},{{{{{{-0.5,0,0},{0,0,0}},{{0,0,0},{0,0,0}}},{{{0.5,-0.5,-0.5},{0,0,0.5}},{{0,0.5,0},{0,0,0}}}},{{{{-0.25,0,0},{-0.25,0,0}},{{0,0,0},{0,0,0}}},{{{0.25,-0.25,-0.5},{0.25,-0.25,0.5}},{{0,0.25,0},{0,0.25,0}}}},{{{{0,0,0},{-0.5,0,0}},{{0,0,0},{0,0,0}}},{{{0,0,-0.5},{0.5,-0.5,0.5}},{{0,0,0},{0,0.5,0}}}}},{{{{{-0.25,0,0},{0,0,0}},{{-0.25,0,0},{0,0,0}}},{{{0.25,-0.5,-0.25},{0,0,0.25}},{{0.25,0.5,-0.25},{0,0,0.25}}}},{{{{-0.125,0,0},{-0.125,0,0}},{{-0.125,0,0},{-0.125,0,0}}},{{{0.125,-0.25,-0.25},{0.125,-0.25,0.25}},{{0.125,0.25,-0.25},{0.125,0.25,0.25}}}},{{{{0,0,0},{-0.25,0,0}},{{0,0,0},{-0.25,0,0}}},{{{0,0,-0.25},{0.25,-0.5,0.25}},{{0,0,-0.25},{0.25,0.5,0.25}}}}},{{{{{0,0,0},{0,0,0}},{{-0.5,0,0},{0,0,0}}},{{{0,-0.5,0},{0,0,0}},{{0.5,0.5,-0.5},{0,0,0.5}}}},{{{{0,0,0},{0,0,0}},{{-0.25,0,0},{-0.25,0,0}}},{{{0,-0.25,0},{0,-0.25,0}},{{0.25,0.25,-0.5},{0.25,0.25,0.5}}}},{{{{0,0,0},{0,0,0}},{{0,0,0},{-0.5,0,0}}},{{{0,0,0},{0,-0.5,0}},{{0,0,-0.5},{0.5,0.5,0.5}}}}}}};

  testParentElementCT< double, 1, ParentElementType>( testParentCoords, referenceValues, referenceGradients );
}



int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}