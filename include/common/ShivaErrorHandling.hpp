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

/**
 * @file ShivaErrorHandling.hpp
 * @brief This file contains functions to assist in the handling of errors and assertions.
 */

#pragma once

#include "ShivaMacros.hpp"

/**
 * @brief This function is used to print an assertion failure message and
 * terminate the program.
 * @param file The name of the file where the assertion failed.
 * @param line The line number where the assertion failed.
 * @param callAbort If true, the program will abort or trap after printing the message.
 * @param fmt The format string for the message to print.
 */
SHIVA_HOST_DEVICE inline
void shivaAssertionFailed( const char * file, int line, bool const callAbort, const char * fmt, ... )
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  printf( "Assertion failed [%s:%d]: ", file, line );
  printf( fmt );
  printf( "\n" );
#if defined(__CUDA_ARCH__)
  __trap();
#elif defined(__HIP_DEVICE_COMPILE__)
  __builtin_trap();
#endif
#else // Host
  fprintf( stderr, "Assertion failed [%s:%d]: ", file, line );
  va_list args;
  va_start( args, fmt );
  vfprintf( stderr, fmt, args );
  va_end( args );
  fprintf( stderr, "\n" );
  if( callAbort )
  {
    // LCOV_EXCL_START
    fprintf( stderr, "Aborting...\n" );
    std::abort();
    // LCOV_EXCL_STOP
  }
#endif
}
