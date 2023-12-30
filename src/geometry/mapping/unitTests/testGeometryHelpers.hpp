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

#pragma once

#include <gtest/gtest.h>


template< typename CELLTYPE, typename SETDATAFUNC, typename CHECKFUNC >
void testConstructionAndSettersHelper( SETDATAFUNC && setData,
                                       CHECKFUNC && checkData )
{
  CELLTYPE cell;
  CELLTYPE const & cellConst = cell;

  typename CELLTYPE::DataType & data = cell.getData();
  typename CELLTYPE::DataType const & constData = cellConst.getData();

  setData( data );
  checkData( constData );
}


template< typename CELLTYPE, typename FUNC >
void testJacobianHelper( FUNC && func )
{
  CELLTYPE cell;
  CELLTYPE const & cellConst = cell;

  typename CELLTYPE::DataType & data = cell.getData();

  func( data, cellConst );
}
