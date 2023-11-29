
#include "../MultiDimensionalSlice.hpp"

#include "../SequenceUtilities.hpp"
#include "common/pmpl.hpp"


#include <gtest/gtest.h>

using namespace shiva;



template< int ... VALUES >
static constexpr auto initializer( std::integer_sequence< int, VALUES ... > )
{
  return std::array<double,sizeof...(VALUES)>{ (VALUES * 3.14159) ... };
}

struct TestMultiDimensionalHelper
{
  static constexpr std::array<double,4> array1d = initializer( std::make_integer_sequence< int, 4 >() );

  static constexpr std::array<double,24> array3d = initializer( std::make_integer_sequence< int, 2 * 3 * 4 >() );
};

TEST( testMultiDimensionalArray, testConstruction )
{
  constexpr MultiDimensionalSlice< double const, 4 > view0c{ TestMultiDimensionalHelper::array1d.data() };
  static_assert( view0c.rank() == 1 );

  double data[4]{ 0.0, 1.0, 2.0, 3.0 };
  MultiDimensionalSlice< double const, 4 > const view1c{ data };
  for( int i = 0; i < 4; ++i )
  {
    EXPECT_EQ( view1c[i], i );
  }

  MultiDimensionalSlice< double, 4 > const view1{ data };
  view1[0] = 3.14159;


}



TEST( testMultiDimensionalArray, testDataAccessor )
{

  double data[4]{ 0.0, 1.0, 2.0, 3.0 };
  double const cdata[4]{ 0.0, 1.0, 2.0, 3.0 };

  MultiDimensionalSlice< double const, 4 > const view0c{ cdata };
  MultiDimensionalSlice< double, 4 > const       view0{ data };
}

void testTraitsHelper()
{
  int * data = nullptr;
  pmpl::genericKernelWrapper( 5, data, [] SHIVA_DEVICE ( int * const kernelData )
  {
    using Array = MultiDimensionalSlice< double, 2, 3, 4 >;
    
    static_assert( Array::rank() == 3 );
    static_assert( Array::extent<0>() == 2 );
    static_assert( Array::extent<1>() == 3 );
    static_assert( Array::extent<2>() == 4 );
    static_assert( Array::size() == 24 );

    double junk[24];
    Array array( junk );
    kernelData[0] = array.rank();
    kernelData[1] = array.extent<0>();
    kernelData[2] = array.extent<1>();
    kernelData[3] = array.extent<2>();
    kernelData[4] = array.size();
  } );

  EXPECT_EQ( data[0], 3 );
  EXPECT_EQ( data[1], 2 );
  EXPECT_EQ( data[2], 3 );
  EXPECT_EQ( data[3], 4 );
  EXPECT_EQ( data[4], 24 );
  pmpl::deallocateData( data );

}
TEST( testMultiDimensionalArray, testTraits )
{
  testTraitsHelper();
}

// void testStridesHelper()
// {
//   pmpl::genericKernelWrapper( [] SHIVA_DEVICE () constexpr
//   {
//     static_assert( MultiDimensionalArrayHelper::stride< 2 >() == 1 );
//     static_assert( MultiDimensionalArrayHelper::stride< 3, 4 >() == 4 );
//     static_assert( MultiDimensionalArrayHelper::stride< 2, 3, 4 >() == 12 );
//   } );
// }

// TEST( testMultiDimensionalArray, testStrides )
// {
//   testStridesHelper();
// }


// void testLinearIndexCT()
// {
//   pmpl::genericKernelWrapper( [] SHIVA_DEVICE () constexpr
//   {
//     using Array = MultiDimensionalSlice< double, 2, 3, 4 >;
//     constexpr int na = Array::extent<0>();
//     constexpr int nb = Array::extent<1>();
//     constexpr int nc = Array::extent<2>();
//     forSequence< na >( [] ( auto const ica )
//     {
//       constexpr int a = decltype(ica)::value;
//       forSequence< nb >( [] ( auto const icb )
//       {
//         constexpr int b = decltype(icb)::value;
//         forSequence< nc >( [] ( auto const icc )
//         {
//           constexpr int c = decltype(icc)::value;
//           static_assert( MultiDimensionalArrayHelper::linearIndexHelper<2,3,4>::template eval< a, b, c >() == a * nb * nc + b * nc + c );
//           static_assert( MultiDimensionalArrayHelper::linearIndexHelper<2,3,4>::eval( a, b, c ) == a * nb * nc + b * nc + c );
//         } );
//       } );
//     } );
//   } );
// }


// void testLinearIndexRT()
// {
//   int * data = nullptr;
//   pmpl::genericKernelWrapper( TestMultiDimensionalHelper::Array3d::size(), data, [] SHIVA_DEVICE ( int * const kernelData )
//   {
//     TestMultiDimensionalHelper::Array3d array;
//     int const na = array.extent<0>();
//     int const nb = array.extent<1>();
//     int const nc = array.extent<2>();

//     for ( int a = 0; a < na; ++a )
//     {
//       for ( int b = 0; b < nb; ++b )
//       {
//         for ( int c = 0; c < nc; ++c )
//         {
//           kernelData[ a * nb * nc + b * nc + c ] = array.linearIndex( a, b, c );
//         }
//       }
//     }
//   } );

//   for ( int a = 0; a < TestMultiDimensionalHelper::Array3d::size(); ++a )
//   {
//     EXPECT_EQ( data[a], a );
//   }
//   pmpl::deallocateData( data );

// }

// TEST( testMultiDimensionalArray, testLinearIndex )
// {
//   testLinearIndexCT();
//   testLinearIndexRT();
// }


// void testParenthesesOperatorCT()
// {
//   using Array = TestMultiDimensionalHelper::Array3d;
//   constexpr int na = Array::extent<0>();
//   constexpr int nb = Array::extent<1>();
//   constexpr int nc = Array::extent<2>();

//   pmpl::genericKernelWrapper( [] SHIVA_DEVICE () constexpr
//   {
//     forSequence< na >( [] ( auto const ica )
//     {
//       constexpr int a = decltype(ica)::value;
//       forSequence< nb >( [=] ( auto const icb )
//       {
//         constexpr int b = decltype(icb)::value;
//         forSequence< nc >( [=] ( auto const icc )
//         {
//           constexpr int c = decltype(icc)::value;
//           static_assert( pmpl::check( TestMultiDimensionalHelper::array3d.operator()< a, b, c >(),
//                                       3.14159 * MultiDimensionalArrayHelper::linearIndexHelper<2,3,4>::eval( a, b, c ),
//                                       1.0e-12 ) );
//           static_assert( pmpl::check( TestMultiDimensionalHelper::array3d( a, b, c ),
//                                       3.14159 * MultiDimensionalArrayHelper::linearIndexHelper<2,3,4>::eval( a, b, c ),
//                                       1.0e-12 ) );
//         } );
//       } );
//     } );
//   } );
// }

// void testParenthesesOperatorRT()
// {
//   double * data = nullptr;
//   pmpl::genericKernelWrapper( TestMultiDimensionalHelper::Array3d::size(), data, [] SHIVA_DEVICE ( double * const kernelData ) constexpr
//   {
//     TestMultiDimensionalHelper::Array3d const array{ initializer< TestMultiDimensionalHelper::Array3d >( std::make_integer_sequence< int, 2 * 3 * 4 >() ) };;
//     int const na = array.extent<0>();
//     int const nb = array.extent<1>();
//     int const nc = array.extent<2>();

//     for ( int a = 0; a < na; ++a )
//     {
//       for ( int b = 0; b < nb; ++b )
//       {
//         for ( int c = 0; c < nc; ++c )
//         {
//           kernelData[ a * nb * nc + b * nc + c ] = array( a, b, c );
//         }
//       }
//     }
//   } );

//   for ( int a = 0; a < TestMultiDimensionalHelper::Array3d::size(); ++a )
//   {
//     EXPECT_EQ( data[a], 3.14159 * a );
//   }
//   pmpl::deallocateData( data );


// }
// TEST( testMultiDimensionalArray, testParenthesesOperator )
// {
//   testParenthesesOperatorCT();
//   testParenthesesOperatorRT();
// }




// void testSquareBracketOperatorCT()
// {
//   using Array = TestMultiDimensionalHelper::Array1d;
//   constexpr int n = Array::extent<0>();

//   pmpl::genericKernelWrapper( [] SHIVA_DEVICE () constexpr
//   {
//     forSequence< n >( [] ( auto const ica )
//     {
//       constexpr int a = decltype(ica)::value;
//       static_assert( pmpl::check( TestMultiDimensionalHelper::array1d[ a ],
//                                   3.14159 * a,
//                                   1.0e-12 ) );
//     } );
//   } );
// }
// TEST( testMultiDimensionalArray, testSquareBracketOperator )
// {
//   testSquareBracketOperatorCT();
// }


// TEST( testMultiDimensionalArray, testView )
// {
//   using Array = TestMultiDimensionalHelper::Array1d;

//   pmpl::genericKernelWrapper( [] SHIVA_DEVICE () constexpr
//   {
//     constexpr MultiDimensionalSpan< double const, 4 > span( TestMultiDimensionalHelper::array1d.m_data );
//     forSequence< Array::extent<0>() >( [span] ( auto const ica )
//     {
//       constexpr int a = decltype(ica)::value;
//       static_assert( pmpl::check( span[ a ],
//                                   3.14159 * a,
//                                   1.0e-12 ) );
//     } );
//   } );

// }


int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
