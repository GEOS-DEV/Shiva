
#include "../finiteElementMethod/parentElements/ParentElement.hpp"
#include "../finiteElementMethod/bases/LagrangeBasis.hpp"
#include "../spacing/Spacing.hpp"
#include "../../common/ShivaMacros.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace shiva;
using namespace shiva::finiteElement::basis;



template< typename REAL_TYPE, int ORDER, typename PARENT_ELEMENT_TYPE, int ... BF_INDEX >
void testParentElementCT( typename PARENT_ELEMENT_TYPE::CoordType const coord,
                          REAL_TYPE const (&referenceValue)[ORDER+1],
                          REAL_TYPE const (&referenceGradient)[ORDER+1],
                          std::integer_sequence<int, BF_INDEX...> )
{

    double values[6] = { PARENT_ELEMENT_TYPE::template value<BF_INDEX>(coord)... };
    double gradients[6] = { PARENT_ELEMENT_TYPE::template gradient<BF_INDEX>(coord)... };

  for( int a=0; a<ORDER+1 ; ++a )
  {
    EXPECT_NEAR( values[a], referenceValue[a], abs(referenceValue[a]) * 1e-13 );
    EXPECT_NEAR( gradients[a], referenceGradient[a], abs(referenceGradient[a]) * 1e-13 );
  }

}


TEST( testSpacing, testLagrangeBasis )
{
  testParentElementCT< double, 5, GaussLobattoSpacing >( {0.3,0.3,0.3}, 
                                                {-0.0039331249999999,0.011438606947084,-0.025205197129078,0.99880774621932,0.026196343962672,-0.0073043749999999}, 
                                                {-0.26265625,0.76193547139425,-1.6595367733139,-0.16178550663066,1.8258868085503,-0.50384375},
                                                std::make_integer_sequence<int,6>{} );

}



int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}