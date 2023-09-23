
#include "../finiteElementMethod/parentElements/ParentElement.hpp"
#include "../finiteElementMethod/bases/LagrangeBasis.hpp"
#include "../spacing/Spacing.hpp"
#include "../../geometry/Cube.hpp"
#include "../../common/ShivaMacros.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <cxxabi.h>

using namespace shiva;
using namespace shiva::discretizations::finiteElementMethod;
using namespace shiva::discretizations::finiteElementMethod::basis;
using namespace shiva::geometry;


template< typename PARENT_ELEMENT_TYPE, int ... INDICES >
void inline someFunctionYouWantToCall( double const (&coord)[3] )
{
    double value = PARENT_ELEMENT_TYPE::template value<INDICES...>( coord );
    printf( "value( ");
    ((printf("%d ",INDICES),...));
    printf( " ) = %f\n", value );
    // printf( "\n" );
}




template <typename... INTEGER_SEQUENCES>
struct recursiveSequenceExpansionHelper
{};
template < typename PARENT_ELEMENT_TYPE, int... INDICES, typename... INTEGER_SEQUENCES>
struct recursiveSequenceExpansionHelper<PARENT_ELEMENT_TYPE, std::integer_sequence<int, INDICES...>, INTEGER_SEQUENCES...> 
{
    template< int ... PERMUTATION, typename REFERENCE_VALUE_TYPE >
    static void value( double const (&coord)[3],
                       REFERENCE_VALUE_TYPE const & referenceValues )
    {
        if constexpr (sizeof...(INTEGER_SEQUENCES)==0)
        {
          double values[sizeof...(INDICES)] = { PARENT_ELEMENT_TYPE::template value<PERMUTATION...,INDICES>(coord)... };
          EXPECT_NEAR( values[0], referenceValues[0], abs(referenceValues[0]) * 1e-13 );
          EXPECT_NEAR( values[1], referenceValues[1], abs(referenceValues[1]) * 1e-13 );

//          printf( "sizeof...(INDICES) = %lu \n", sizeof...(INDICES) );
          // printf( " values = %f, %f \n", value[0], value[1] );
          // printf( "rvalues = %f, %f \n", referenceValues[0], referenceValues[1] );
          // int status = -4; // some arbitrary value to eliminate the compiler warning
          // char * const demangledName = abi::__cxa_demangle( typeid(REFERENCE_VALUE_TYPE).name(), nullptr, nullptr, &status );
          // std::cout<<demangledName<<std::endl;
//           (someFunctionYouWantToCall< PARENT_ELEMENT_TYPE, PERMUTATION...,INDICES>( coord ),...);
        }
        else
        {
           (recursiveSequenceExpansionHelper<PARENT_ELEMENT_TYPE, INTEGER_SEQUENCES... >::template value< PERMUTATION..., INDICES>( coord, referenceValues[INDICES] ),...);
        }
    }
};

template< typename PARENT_ELEMENT_TYPE, typename ... INTEGER_SEQUENCES, typename REFERENCE_VALUE_TYPE >
void recursiveSequenceExpansion( double const (&coord)[3],
                                 REFERENCE_VALUE_TYPE const & referenceValues )
{
    recursiveSequenceExpansionHelper< PARENT_ELEMENT_TYPE, INTEGER_SEQUENCES...>::value( coord, referenceValues );
}


template< typename REAL_TYPE, int ORDER, typename PARENT_ELEMENT_TYPE >
void testParentElementCT( REAL_TYPE const (&testParentCoords)[ORDER+2],
                          REAL_TYPE const (&referenceValues)[ORDER+2][ORDER+2][ORDER+2][ORDER+1][ORDER+1][ORDER+1]//,
//                          REAL_TYPE const (&referenceGradient)[ORDER+1] 
                        )
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
                                                                                    referenceValues[i][j][k] );
        }
      }
    }
}

TEST( testSpacing, testLagrangeBasis )
{
  using ParentElementType = ParentElement< double, 
                 Cube, 
                 LagrangeBasis<double,1,GaussLobattoSpacing>, 
                 LagrangeBasis<double,1,GaussLobattoSpacing>, 
                 LagrangeBasis<double,1,GaussLobattoSpacing>
               >;

  double const testParentCoords[3] = { -1.0, 0.0, 1.0 };
  double const referenceValues[3][3][3][2][2][2] = {{{{{{1.,0},{0,0}},{{0,0},{0,0}}},{{{0.5,0.5},{0,0}},{{0,0},{0,0}}},{{{0,1.},{0,0}},{{0,0},{0,0}}}},{{{{0.5,0},{0.5,0}},{{0,0},{0,0}}},{{{0.25,0.25},{0.25,0.25}},{{0,0},{0,0}}},{{{0,0.5},{0,0.5}},{{0,0},{0,0}}}},{{{{0,0},{1.,0}},{{0,0},{0,0}}},{{{0,0},{0.5,0.5}},{{0,0},{0,0}}},{{{0,0},{0,1.}},{{0,0},{0,0}}}}},{{{{{0.5,0},{0,0}},{{0.5,0},{0,0}}},{{{0.25,0.25},{0,0}},{{0.25,0.25},{0,0}}},{{{0,0.5},{0,0}},{{0,0.5},{0,0}}}},{{{{0.25,0},{0.25,0}},{{0.25,0},{0.25,0}}},{{{0.125,0.125},{0.125,0.125}},{{0.125,0.125},{0.125,0.125}}},{{{0,0.25},{0,0.25}},{{0,0.25},{0,0.25}}}},{{{{0,0},{0.5,0}},{{0,0},{0.5,0}}},{{{0,0},{0.25,0.25}},{{0,0},{0.25,0.25}}},{{{0,0},{0,0.5}},{{0,0},{0,0.5}}}}},{{{{{0,0},{0,0}},{{1.,0},{0,0}}},{{{0,0},{0,0}},{{0.5,0.5},{0,0}}},{{{0,0},{0,0}},{{0,1.},{0,0}}}},{{{{0,0},{0,0}},{{0.5,0},{0.5,0}}},{{{0,0},{0,0}},{{0.25,0.25},{0.25,0.25}}},{{{0,0},{0,0}},{{0,0.5},{0,0.5}}}},{{{{0,0},{0,0}},{{0,0},{1.,0}}},{{{0,0},{0,0}},{{0,0},{0.5,0.5}}},{{{0,0},{0,0}},{{0,0},{0,1.}}}}}};
  // recursiveSequenceExpansion< ParentElementType,
  //                             std::integer_sequence< int, 0, 1 >,
  //                             std::integer_sequence< int, 0, 1 >,
  //                             std::integer_sequence< int, 0, 1 > >( testParentCoords,
  //                                                                   referenceValues[0][0][0] );


  testParentElementCT< double, 1, ParentElementType>( testParentCoords, referenceValues );
}



int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}