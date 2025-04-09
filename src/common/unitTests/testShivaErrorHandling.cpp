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


#include "../ShivaErrorHandling.hpp"
#include "../pmpl.hpp"

#include <gtest/gtest.h>

using namespace shiva;

void test_shivaAssertionFailed( bool const callAbort )
{
  shivaAssertionFailed( __FILE__, __LINE__, callAbort, "host assertion failed" );
  pmpl::genericKernelWrapper( [callAbort] SHIVA_DEVICE ()
  {
    shivaAssertionFailed( __FILE__, __LINE__, callAbort, "device assertion failed" );
  }, callAbort );
}

TEST( testShivaErrorHandling, test_shivaAssertionFailed )
{
  EXPECT_DEATH( {test_shivaAssertionFailed( true );}, "" );
}

TEST( testShivaErrorHandling, test_shivaAssertionFailed_NoDeath )
{
  test_shivaAssertionFailed( false );
}

int main( int argc, char * * argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  int const result = RUN_ALL_TESTS();
  return result;
}
