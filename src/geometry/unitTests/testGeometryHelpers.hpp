#pragma once

#include <gtest/gtest.h>


template< typename CELLTYPE, typename FUNC >
void testConstructionAndSettersHelper( FUNC && setData )
{
  CELLTYPE cell;
  CELLTYPE const & cellConst = cell;

  typename CELLTYPE::DataType & data = cell.getData();
  typename CELLTYPE::DataType const & constData = cellConst.getData();

  setData( data, constData );
}


template< typename CELLTYPE, typename FUNC >
void testJacobianHelper( FUNC && func )
{
  CELLTYPE cell;
  CELLTYPE const & cellConst = cell;

  typename CELLTYPE::DataType & data = cell.getData();

  func( data, cellConst );
}