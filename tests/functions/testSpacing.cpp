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



#include "../spacing/Spacing.hpp"
#include "common/SequenceUtilities.hpp"
#include "common/pmpl.hpp"

#include <gtest/gtest.h>
#include <cmath>

using namespace shiva;

template struct shiva::EqualSpacing< float, 6 >;



template< typename ... T >
struct ReferenceSolution;

template< typename REAL_TYPE >
struct ReferenceSolution< EqualSpacing< REAL_TYPE, 2 > >
{
  static constexpr REAL_TYPE coords[2] = { -1.0, 1.0 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< EqualSpacing< REAL_TYPE, 3 > >
{
  static constexpr REAL_TYPE coords[3] = { -1.0, 0.0, 1.0 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< EqualSpacing< REAL_TYPE, 4 > >
{
  static constexpr REAL_TYPE coords[4] = { -1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< EqualSpacing< REAL_TYPE, 5 > >
{
  static constexpr REAL_TYPE coords[5] = { -1.0, -0.5, 0.0, 0.5, 1.0 };
};



template< typename REAL_TYPE >
struct ReferenceSolution< GaussLegendreSpacing< REAL_TYPE, 2 > >
{
  static constexpr REAL_TYPE coords[2] = { -0.57735026918962576451, 0.57735026918962576451 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< GaussLegendreSpacing< REAL_TYPE, 3 > >
{
  static constexpr REAL_TYPE coords[3] = { -0.77459666924148337704, 0.0, 0.77459666924148337704 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< GaussLegendreSpacing< REAL_TYPE, 4 > >
{
  static constexpr REAL_TYPE coords[4] = { -0.86113631159405257522,
                                           -0.3399810435848562648,
                                           0.3399810435848562648,
                                           0.86113631159405257522 };
};


template< typename REAL_TYPE >
struct ReferenceSolution< GaussLobattoSpacing< REAL_TYPE, 2 > >
{
  static constexpr REAL_TYPE coords[2] = { -1.0, 1.0 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< GaussLobattoSpacing< REAL_TYPE, 3 > >
{
  static constexpr REAL_TYPE coords[3] = { -1.0, 0.0, 1.0 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< GaussLobattoSpacing< REAL_TYPE, 4 > >
{
  static constexpr REAL_TYPE coords[4] = { -1.0, -0.44721359549995793928, 0.44721359549995793928, 1.0 };
};

template< typename REAL_TYPE >
struct ReferenceSolution< GaussLobattoSpacing< REAL_TYPE, 5 > >
{
  static constexpr REAL_TYPE coords[5] = { -1.0,
                                           -0.6546536707079771438,
                                           0.0,
                                           0.6546536707079771438,
                                           1.0 };
};



template< template< typename, int > typename SPACING, typename REAL_TYPE, int N >
SHIVA_GLOBAL void runtimeValuesKernel( REAL_TYPE * const values )
{
  for ( int a = 0; a < N; ++a )
  {
    values[a] = SPACING< REAL_TYPE, N >::coordinate( a );
  }
}

template< template< typename, int > typename SPACING, typename REAL_TYPE, int N >
void testSpacingValuesAtRuntime()
{
  using SpacingType = SPACING< REAL_TYPE, N >;
  using Ref = ReferenceSolution< SpacingType >;

#if defined(SHIVA_USE_DEVICE)
  constexpr int bytes = N * sizeof(REAL_TYPE);
  REAL_TYPE *values;
  deviceMallocManaged( &values, bytes );
  runtimeValuesKernel< SPACING, REAL_TYPE, N ><< < 1, 1 >> > ( values );
  deviceDeviceSynchronize();
#else
  REAL_TYPE values[N];
  runtimeValuesKernel< SPACING, REAL_TYPE, N >( values );
#endif

  for ( int a = 0; a < N; ++a )
  {
    EXPECT_NEAR( values[a], Ref::coords[a], 1e-14 );
  }

#if defined(SHIVA_USE_DEVICE)
  deviceFree( values );
#endif
}



template< template< typename, int > typename SPACING, typename REAL_TYPE, std::size_t ... I >
SHIVA_GLOBAL void compileTimeValuesKernel( std::index_sequence< I... > )
{
  constexpr int N = sizeof...(I);
  using SpacingType = SPACING< REAL_TYPE, N >;
  using Ref = ReferenceSolution< SpacingType >;

  constexpr REAL_TYPE tolerance = 1e-13;
  forSequence< N >( [&] ( auto const a ) constexpr
  {
    static_assert( pmpl::check( SpacingType::template coordinate< a >(), Ref::coords[a], tolerance ) );
  } );
}

template< template< typename, int > typename SPACING, typename REAL_TYPE, typename INDEX_SEQUENCE >
void testSpacingValuesAtCompileTime( INDEX_SEQUENCE iSeq )
{
#if defined(SHIVA_USE_DEVICE)
  compileTimeValuesKernel< SPACING, REAL_TYPE ><< < 1, 1 >> > ( iSeq );
#else
  compileTimeValuesKernel< SPACING, REAL_TYPE >( iSeq );
#endif
}



TEST( testSpacing, testEqualSpacingRT )
{
  testSpacingValuesAtRuntime< EqualSpacing, float, 2 >( );
  testSpacingValuesAtRuntime< EqualSpacing, double, 3 >( );
  testSpacingValuesAtRuntime< EqualSpacing, double, 4 >( );
  testSpacingValuesAtRuntime< EqualSpacing, double, 5 >( );

}

TEST( testSpacing, testEqualSpacingCT )
{
  testSpacingValuesAtCompileTime< EqualSpacing, double >( std::make_index_sequence< 2 >{} );
  testSpacingValuesAtCompileTime< EqualSpacing, double >( std::make_index_sequence< 3 >{} );
  testSpacingValuesAtCompileTime< EqualSpacing, double >( std::make_index_sequence< 4 >{} );
  testSpacingValuesAtCompileTime< EqualSpacing, double >( std::make_index_sequence< 5 >{} );
}


TEST( testSpacing, testGaussLegendreSpacingRT )
{
  testSpacingValuesAtRuntime< GaussLegendreSpacing, double, 2 >( );
  testSpacingValuesAtRuntime< GaussLegendreSpacing, double, 3 >( );
  testSpacingValuesAtRuntime< GaussLegendreSpacing, double, 4 >( );
}

TEST( testSpacing, testGaussLegendreSpacingCT )
{
  testSpacingValuesAtCompileTime< GaussLegendreSpacing, double >( std::make_index_sequence< 2 >{} );
  testSpacingValuesAtCompileTime< GaussLegendreSpacing, double >( std::make_index_sequence< 3 >{} );
  testSpacingValuesAtCompileTime< GaussLegendreSpacing, double >( std::make_index_sequence< 4 >{} );
}

TEST( testSpacing, testGaussLobattoSpacingRT )
{
  testSpacingValuesAtRuntime< GaussLobattoSpacing, double, 2 >( );
  testSpacingValuesAtRuntime< GaussLobattoSpacing, double, 3 >( );
  testSpacingValuesAtRuntime< GaussLobattoSpacing, double, 4 >( );
  testSpacingValuesAtRuntime< GaussLobattoSpacing, double, 5 >( );
}

TEST( testSpacing, testGaussLobattoSpacingCT )
{
  testSpacingValuesAtCompileTime< GaussLobattoSpacing, double >( std::make_index_sequence< 2 >{} );
  testSpacingValuesAtCompileTime< GaussLobattoSpacing, double >( std::make_index_sequence< 3 >{} );
  testSpacingValuesAtCompileTime< GaussLobattoSpacing, double >( std::make_index_sequence< 4 >{} );
  testSpacingValuesAtCompileTime< GaussLobattoSpacing, double >( std::make_index_sequence< 5 >{} );
}


int main( int argc, char * * argv )
{
  shiva::EqualSpacing< float, 6 > junk;
  std::cout << junk.coordinate< 0 >() << std::endl;

  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
