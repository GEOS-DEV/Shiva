
#include "../finiteElementMethod/bases/LagrangeBasis.hpp"
#include "../spacing/Spacing.hpp"
#include "../../common/ShivaMacros.hpp"
#include "types/types.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace shiva;
using namespace shiva::discretizations::finiteElementMethod::basis;

constexpr bool check( double const a, double const b, double const tolerance )
{
  return ( ( a / b - 1.0 ) < tolerance ) && ( ( a / b - 1.0 ) > -tolerance );
}


template< typename ... T >
struct TestBasisHelper;

template< bool USE_FOR_SEQUENCE >
struct TestBasisHelper< LagrangeBasis< double, 5, EqualSpacing, USE_FOR_SEQUENCE > >
{
  using BasisType = LagrangeBasis< double, 5, EqualSpacing, USE_FOR_SEQUENCE >;
  static constexpr int order = 5;
  static constexpr double coord = 0.3;
  static constexpr double refValues[order + 1] = {-0.0076904296875, 0.0555419921875, -0.199951171875, 0.999755859375, 0.1666259765625, -0.0142822265625};
  static constexpr double refGradients[order + 1] = {-0.064208984375, 0.44474283854167, -1.42333984375, -0.88134765625, 2.0747884114583, -0.150634765625};
};

template< bool USE_FOR_SEQUENCE >
struct TestBasisHelper< LagrangeBasis< double, 5, GaussLobattoSpacing, USE_FOR_SEQUENCE > >
{
  using BasisType = LagrangeBasis< double, 5, GaussLobattoSpacing, USE_FOR_SEQUENCE >;
  static constexpr int order = 5;
  static constexpr double coord = 0.3;
  static constexpr double refValues[order + 1] = {-0.0039331249999999, 0.011438606947084, -0.025205197129078, 0.99880774621932, 0.026196343962672, -0.0073043749999999};
  static constexpr double refGradients[order + 1] = {-0.26265625, 0.76193547139425, -1.6595367733139, -0.16178550663066, 1.8258868085503, -0.50384375};
};



template< typename BASIS_HELPER_TYPE >
constexpr void compileTimeCheck()
{
  using BasisType = typename BASIS_HELPER_TYPE::BasisType;
  constexpr int order = BASIS_HELPER_TYPE::order;

  constexpr double coord = BASIS_HELPER_TYPE::coord;

  executeSequence< order + 1 >( [&] ( auto ... BF_INDEX ) constexpr
  {
    constexpr double    values[order + 1] = { BasisType::template value< BF_INDEX >( coord )... };
    constexpr double gradients[order + 1] = { BasisType::template gradient< BF_INDEX >( coord )... };
    constexpr double tolerance = 1.0e-12;

    static_assert( ( check( values[BF_INDEX], BASIS_HELPER_TYPE::refValues[BF_INDEX], tolerance ), ... ) );
    static_assert( ( check( gradients[BF_INDEX], BASIS_HELPER_TYPE::refGradients[BF_INDEX], tolerance ), ... ) );
  } );
}

template< typename BASIS_HELPER_TYPE >
constexpr void runTimeCheck()
{
  using BasisType = typename BASIS_HELPER_TYPE::BasisType;
  constexpr int order = BASIS_HELPER_TYPE::order;

  double coord = BASIS_HELPER_TYPE::coord;

  executeSequence< order + 1 >( [&] ( auto ... BF_INDEX ) constexpr
  {
    double    values[order + 1] = { BasisType::template value< BF_INDEX >( coord )... };
    double gradients[order + 1] = { BasisType::template gradient< BF_INDEX >( coord )... };
    constexpr double tolerance = 1.0e-10;

    for ( int a = 0; a < order + 1; ++a )
    {
      EXPECT_NEAR( values[a], BASIS_HELPER_TYPE::refValues[a], fabs( BASIS_HELPER_TYPE::refValues[a] * tolerance ) );
      EXPECT_NEAR( gradients[a], BASIS_HELPER_TYPE::refGradients[a], fabs( BASIS_HELPER_TYPE::refGradients[a] * tolerance ) );
    }
  } );
}

TEST( testSpacing, testLagrangeBasisEqualSpacing )
{
  using BasisHelperType = TestBasisHelper< LagrangeBasis< double, 5, EqualSpacing, false > >;
  compileTimeCheck< BasisHelperType >();
  runTimeCheck< BasisHelperType >();
}

TEST( testSpacing, testLagrangeBasisEqualSpacingUseForSequence )
{
  using BasisHelperType = TestBasisHelper< LagrangeBasis< double, 5, EqualSpacing, true > >;
  compileTimeCheck< BasisHelperType >();
  runTimeCheck< BasisHelperType >();
}


TEST( testSpacing, testLagrangeBasisGaussLobattoSpacing )
{
  using BasisHelperType = TestBasisHelper< LagrangeBasis< double, 5, GaussLobattoSpacing, false > >;
  compileTimeCheck< BasisHelperType >();
  runTimeCheck< BasisHelperType >();
}

TEST( testSpacing, testLagrangeBasisGaussLobattoSpacingUseForSequence )
{
  using BasisHelperType = TestBasisHelper< LagrangeBasis< double, 5, GaussLobattoSpacing, false > >;
  compileTimeCheck< BasisHelperType >();
  runTimeCheck< BasisHelperType >();
}


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
