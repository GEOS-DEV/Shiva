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


#include "../CArray.hpp"

#include "../SequenceUtilities.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;

template< typename ARRAY_TYPE, int ... VALUES >
static constexpr auto initializer( std::integer_sequence< int, VALUES ... > )
{
  return ARRAY_TYPE{ (VALUES * 3.14159) ... };
}

struct TestCArrayHelper
{

  using Array1d = CArrayNd< double const, 4 >;
  static constexpr Array1d array1d{ initializer< Array1d >( std::make_integer_sequence< int, 4 >() ) };

  using Array3d = CArrayNd< double const, 2, 3, 4 >;
  static constexpr Array3d array3d{ initializer< Array3d >( std::make_integer_sequence< int, 2 * 3 * 4 >() ) };
};


TEST( testCArray, testSingleValueInitialization )
{
  CArrayNd< double, 2, 3, 4 > array{ 0.0 };
  for ( int a = 0; a < 24; ++a )
  {
    EXPECT_EQ( array.data()[a], 0.0 );
  }
}

TEST( testCArray, testViewInitialization )
{
  double data[24]{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
  CArrayViewNd< double, 2, 3, 4 > array{ data };
  for ( int a = 0; a < 24; ++a )
  {
    EXPECT_EQ( array.data()[a], a );
  }

  double const data3c[2][3][4]{ { {  0, 1, 2, 3 },
    {  4, 5, 6, 7 },
    {  8, 9, 10, 11 } },
    { { 12, 13, 14, 15 },
      { 16, 17, 18, 19 },
      { 20, 21, 22, 23 } } };
  CArrayViewNd< double const, 2, 3, 4 > const array3c{ &(data3c[0][0][0]) };
  for ( int a = 0; a < 24; ++a )
  {
    EXPECT_EQ( array3c.data()[a], a );
  }



  double data3[2][3][4] = { { {0.0} } };
  CArrayViewNd< double, 2, 3, 4 > array3{ &(data3[0][0][0]) };
  for ( int a = 0; a < 2; ++a )
  {
    for ( int b = 0; b < 3; ++b )
    {
      for ( int c = 0; c < 4; ++c )
      {
        array3( a, b, c ) = 100 * a + 10 * b + c;
      }
    }
  }

}

void testTraitsHelper()
{
  int * data = nullptr;
  constexpr int N = 5;
  data = new int[N];
  pmpl::genericKernelWrapper( N, data, [] SHIVA_DEVICE ( int * const kernelData )
  {
    using Array = TestCArrayHelper::Array3d;
    static_assert( Array::rank() == 3 );
    static_assert( Array::extent< 0 >() == 2 );
    static_assert( Array::extent< 1 >() == 3 );
    static_assert( Array::extent< 2 >() == 4 );
    static_assert( Array::size() == 24 );

    kernelData[0] = Array::rank();
    kernelData[1] = Array::extent< 0 >();
    kernelData[2] = Array::extent< 1 >();
    kernelData[3] = Array::extent< 2 >();
    kernelData[4] = Array::size();
  } );
  EXPECT_EQ( data[0], 3 );
  EXPECT_EQ( data[1], 2 );
  EXPECT_EQ( data[2], 3 );
  EXPECT_EQ( data[3], 4 );
  EXPECT_EQ( data[4], 24 );

  delete[] data;
}
TEST( testCArray, testTraits )
{
  testTraitsHelper();
}

void testStridesHelper()
{
  pmpl::genericKernelWrapper( [] SHIVA_DEVICE ()
  {
    static_assert( CArrayHelper::stride< 2 >() == 1 );
    static_assert( CArrayHelper::stride< 3, 4 >() == 4 );
    static_assert( CArrayHelper::stride< 2, 3, 4 >() == 12 );
  } );
}

TEST( testCArray, testStrides )
{
  testStridesHelper();
}

#if !defined( SHIVA_USE_BOUNDS_CHECK ) || !defined(SHIVA_USE_DEVICE)
void testLinearIndexCT()
{
  pmpl::genericKernelWrapper( [] SHIVA_DEVICE ()
  {
    using Array = TestCArrayHelper::Array3d;
    constexpr int na = Array::extent< 0 >();
    constexpr int nb = Array::extent< 1 >();
    constexpr int nc = Array::extent< 2 >();
    forSequence< na >( [] ( auto const ica )
    {
      constexpr int a = decltype(ica)::value;
      forSequence< nb >( [] ( auto const icb )
      {
        constexpr int b = decltype(icb)::value;
        forSequence< nc >( [] ( auto const icc )
        {
          constexpr int c = decltype(icc)::value;
          static_assert( CArrayHelper::linearIndexHelper< 2, 3, 4 >::template eval< a, b, c >() == a * nb * nc + b * nc + c );
          static_assert( CArrayHelper::linearIndexHelper< 2, 3, 4 >::eval( a, b, c ) == a * nb * nc + b * nc + c );
        } );
      } );
    } );
  } );
}
#endif

void testLinearIndexRT()
{
  int * data = nullptr;
  constexpr int N = TestCArrayHelper::Array3d::size();
  data = new int[N];
  pmpl::genericKernelWrapper( N, data, [] SHIVA_DEVICE ( int * const kernelData )
  {
    int const na = TestCArrayHelper::array3d.extent< 0 >();
    int const nb = TestCArrayHelper::array3d.extent< 1 >();
    int const nc = TestCArrayHelper::array3d.extent< 2 >();

    for ( int a = 0; a < na; ++a )
    {
      for ( int b = 0; b < nb; ++b )
      {
        for ( int c = 0; c < nc; ++c )
        {
          kernelData[ a * nb * nc + b * nc + c ] = CArrayHelper::linearIndexHelper< 2, 3, 4 >::eval( a, b, c );
        }
      }
    }
  } );

  for ( int a = 0; a < TestCArrayHelper::Array3d::size(); ++a )
  {
    EXPECT_EQ( data[a], a );
  }
  delete [] data;

}

TEST( testCArray, testLinearIndex )
{
#if !defined( SHIVA_USE_BOUNDS_CHECK ) || !defined(SHIVA_USE_DEVICE)
  testLinearIndexCT();
#endif
  testLinearIndexRT();
}

#if !defined( SHIVA_USE_BOUNDS_CHECK ) || !defined(SHIVA_USE_DEVICE)
void testParenthesesOperatorCT()
{
  using Array = TestCArrayHelper::Array3d;
  constexpr int na = Array::extent< 0 >();
  constexpr int nb = Array::extent< 1 >();
  constexpr int nc = Array::extent< 2 >();

  pmpl::genericKernelWrapper( [] SHIVA_DEVICE ()
  {
    forSequence< na >( [] ( auto const ica )
    {
      constexpr int a = decltype(ica)::value;
      forSequence< nb >( [ = ] ( auto const icb )
      {
        constexpr int b = decltype(icb)::value;
        forSequence< nc >( [ = ] ( auto const icc )
        {
          constexpr int c = decltype(icc)::value;
          static_assert( pmpl::check( TestCArrayHelper::array3d.operator()< a, b, c >(),
                                      3.14159 * CArrayHelper::linearIndexHelper< 2, 3, 4 >::eval( a, b, c ),
                                      1.0e-12 ) );
          static_assert( pmpl::check( TestCArrayHelper::array3d( a, b, c ),
                                      3.14159 * CArrayHelper::linearIndexHelper< 2, 3, 4 >::eval( a, b, c ),
                                      1.0e-12 ) );
        } );
      } );
    } );
  } );
}
#endif

void testParenthesesOperatorRT()
{
  double * data = nullptr;
  constexpr int N = TestCArrayHelper::Array3d::size();
  data = new double[N];
  pmpl::genericKernelWrapper( N, data, [] SHIVA_DEVICE ( double * const kernelData )
  {
    TestCArrayHelper::Array3d const array{ initializer< TestCArrayHelper::Array3d >( std::make_integer_sequence< int, 2 * 3 * 4 >() ) };;
    int const na = array.extent< 0 >();
    int const nb = array.extent< 1 >();
    int const nc = array.extent< 2 >();

    for ( int a = 0; a < na; ++a )
    {
      for ( int b = 0; b < nb; ++b )
      {
        for ( int c = 0; c < nc; ++c )
        {
          kernelData[ a * nb * nc + b * nc + c ] = array( a, b, c );
        }
      }
    }
  } );

  for ( int a = 0; a < TestCArrayHelper::Array3d::size(); ++a )
  {
    EXPECT_EQ( data[a], 3.14159 * a );
  }
  delete [] data;


}
TEST( testCArray, testParenthesesOperator )
{
#if !defined( SHIVA_USE_BOUNDS_CHECK ) || !defined(SHIVA_USE_DEVICE)
  testParenthesesOperatorCT();
#endif
  testParenthesesOperatorRT();
}


template< int DIM >
void testBoundsCheckParenthesesOperator1dHelper( int const ilower = 0,
                                                 int const iupper = DIM - 1 )
{
  pmpl::genericKernelWrapper( [ilower, iupper] SHIVA_DEVICE ()
  {
    CArrayNd< double, DIM > array{ 0.0 };
    for ( int i = ilower; i <= iupper; ++i )
    {
      array( i ) = i;
    }
  } );
}
TEST( testCArray, testBoundsCheckParenthesesOperator1d )
{
  testBoundsCheckParenthesesOperator1dHelper< 2 >();
  EXPECT_DEATH( {testBoundsCheckParenthesesOperator1dHelper< 2 >( -1, -1 );}, "" );
  EXPECT_DEATH( {testBoundsCheckParenthesesOperator1dHelper< 2 >( 2, 2 );}, "" );
}

template< int DIM0 = 2, int DIM1 = 4 >
void testBoundsCheckParenthesesOperator2dHelper( int const ilower = 0,
                                                 int const iupper = DIM0 - 1,
                                                 int const jlower = 0,
                                                 int const jupper = DIM1 - 1 )
{
  pmpl::genericKernelWrapper( [ilower, iupper, jlower, jupper] SHIVA_DEVICE ()
  {
    CArrayNd< double, DIM0, DIM1 > array{ 0.0 };
    for ( int i = ilower; i <= iupper; ++i )
    {
      for ( int j = jlower; j <= jupper; ++j )
      {
        array( i, j ) = i * DIM1 + j;
      }
    }
  } );
}
TEST( testCArray, testBoundsCheckParenthesesOperator2d )
{
  testBoundsCheckParenthesesOperator2dHelper();

  EXPECT_DEATH( {testBoundsCheckParenthesesOperator2dHelper( -1, -1, 0, 3 );}, "" );
  EXPECT_DEATH( {testBoundsCheckParenthesesOperator2dHelper( 2, 2, 0, 3 );}, "" );
  EXPECT_DEATH( {testBoundsCheckParenthesesOperator2dHelper( 0, 1, -1, -1 );}, "" );
  EXPECT_DEATH( {testBoundsCheckParenthesesOperator2dHelper( 0, 1, 4, 4 );}, "" );
  EXPECT_DEATH( {testBoundsCheckParenthesesOperator2dHelper( -1, -1, -1, -1 );}, "" );
  EXPECT_DEATH( {testBoundsCheckParenthesesOperator2dHelper( -1, -1, 4, 4 );}, "" );
  EXPECT_DEATH( {testBoundsCheckParenthesesOperator2dHelper( 2, 2, -1, -1 );}, "" );
  EXPECT_DEATH( {testBoundsCheckParenthesesOperator2dHelper( 2, 2, 4, 4 );}, "" );

}

template< int DIM = 3 >
void testSquareBracketOperator1DHelper( int const ilower = 0,
                                        int const iupper = DIM - 1 )
{
  pmpl::genericKernelWrapper( [ilower, iupper] SHIVA_DEVICE ()
  {
    CArrayNd< double, DIM > array{ 0.0 };
    CArrayNd< double, DIM > const & constarray = array;

    static_assert( std::is_same_v< decltype( array[0] ), double & > );
    static_assert( std::is_same_v< decltype( constarray[0] ), double const & > );

    for ( int i = ilower; i <= iupper; ++i )
    {
      array[i] = i;
      SHIVA_ASSERT_MSG( fabs( array( i ) - static_cast< double >(i) ) < 1e-10, "values not equal" );
      SHIVA_ASSERT_MSG( fabs( constarray[ i ] - static_cast< double >(i) ) < 1e-10, "values not equal" );

    }
  } );
}
TEST( testCArray, testSquareBracketOperator1D )
{
  testSquareBracketOperator1DHelper();

  EXPECT_DEATH( {testSquareBracketOperator1DHelper< 2 >( -1, -1 );}, "" );
  EXPECT_DEATH( {testSquareBracketOperator1DHelper< 2 >( 3, 3 );}, "" );

}


void testSquareBracketOperator2DHelper()
{
  pmpl::genericKernelWrapper( [] SHIVA_DEVICE ()
  {
    constexpr int dims[2] = { 2, 4 };
    CArrayNd< double, dims[0], dims[1] > array{ 0.0 };

    // create slices
    auto slice0 = array[0];
    auto slice1 = array[1];
    // test to make sure the types of the slices are correct
    static_assert( std::is_same_v< decltype( slice0 ), CArrayViewNd< double, dims[1] > > );
    static_assert( std::is_same_v< decltype( slice1 ), CArrayViewNd< double, dims[1] > > );

    // create const slices
    CArrayNd< double, dims[0], dims[1] > const & constarray = array;
    auto cslice0 = constarray[0];
    auto cslice1 = constarray[1];
    // test to make sure the types of the slices are correct
    static_assert( std::is_same_v< decltype( cslice0 ), CArrayViewNd< const double, dims[1] > > );
    static_assert( std::is_same_v< decltype( cslice1 ), CArrayViewNd< const double, dims[1] > > );

    // test to make sure the slices point to the correct data
    SHIVA_ASSERT_MSG( slice0.data() == &array( 0, 0 ),
                      "addresses not equal: slice(%p) != array(%p)",
                      static_cast< void * >( slice0.data() ),
                      static_cast< void * >( &array( 0, 0 ) ) );

    SHIVA_ASSERT_MSG( slice1.data() == &array( 1, 0 ),
                      "addresses not equal: slice(%p) != array(%p)",
                      static_cast< void * >( slice1.data() ),
                      static_cast< void * >( &array( 1, 0 ) ) );

    SHIVA_ASSERT_MSG( cslice0.data() == &array( 0, 0 ),
                      "addresses not equal: slice(%p) != array(%p)",
                      static_cast< void const * >( cslice0.data() ),
                      static_cast< void * >( &array( 0, 0 ) ) );

    SHIVA_ASSERT_MSG( cslice1.data() == &array( 1, 0 ),
                      "addresses not equal: slice(%p) != array(%p)",
                      static_cast< void const * >( cslice1.data() ),
                      static_cast< void * >( &array( 1, 0 )  ) );

    // check values in the slices are set correctly
    for ( int i1 = 0; i1 < dims[1]; ++i1 )
    {
      slice0[i1] = i1 + 1;
      slice1[i1] = 100 + i1 + 1;
      SHIVA_ASSERT_MSG( fabs( array( 0, i1 ) - ( i1 + 1 ) ) < 1e-10, "values not equal" );
      SHIVA_ASSERT_MSG( fabs( array( 1, i1 ) - ( 100 + i1 + 1 ) ) < 1e-10, "values not equal" );

      SHIVA_ASSERT_MSG( fabs( slice0[i1] - ( cslice0[i1] ) ) < 1e-10, "values not equal" );
      SHIVA_ASSERT_MSG( fabs( slice1[i1] - ( cslice1[i1] ) ) < 1e-10, "values not equal" );
    }
  } );
}
TEST( testCArray, testSquareBracketOperator2D )
{
  testSquareBracketOperator2DHelper();
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
