
#include "../finiteElementMethod/bases/LagrangeBasis.hpp"
#include "../spacing/Spacing.hpp"
#include "../../common/ShivaMacros.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace shiva;
using namespace shiva::discretizations::finiteElementMethod::basis;



template< typename REAL_TYPE, int ORDER, template<typename,int> typename SPACING, int ... BF_INDEX >
void testBasisCT( REAL_TYPE const coord,
                  REAL_TYPE const (&referenceValue)[ORDER+1],
                  REAL_TYPE const (&referenceGradient)[ORDER+1],
                  std::integer_sequence<int, BF_INDEX...> )
{

  using BasisType = LagrangeBasis<REAL_TYPE,ORDER,SPACING>;

    double values[6] = { BasisType::template value<BF_INDEX>(coord)... };
    double gradients[6] = { BasisType::template gradient<BF_INDEX>(coord)... };

  for( int a=0; a<ORDER+1 ; ++a )
  {
    EXPECT_NEAR( values[a], referenceValue[a], abs(referenceValue[a]) * 1e-13 );
    EXPECT_NEAR( gradients[a], referenceGradient[a], abs(referenceGradient[a]) * 1e-13 );
  }

}


TEST( testSpacing, testLagrangeBasis )
{
  testBasisCT< double, 5, EqualSpacing >( 0.3, 
                                          {-0.0076904296875,0.0555419921875,-0.199951171875,0.999755859375,0.1666259765625,-0.0142822265625}, 
                                          {-0.064208984375,0.44474283854167,-1.42333984375,-0.88134765625,2.0747884114583,-0.150634765625},
                                          std::make_integer_sequence<int,6>{} );

  testBasisCT< double, 5, GaussLobattoSpacing >( 0.3, 
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