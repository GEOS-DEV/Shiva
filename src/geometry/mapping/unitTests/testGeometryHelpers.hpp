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
