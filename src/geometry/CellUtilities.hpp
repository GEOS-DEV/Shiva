#include "CellHexahedron.hpp"

namespace shiva
{
namespace cellUtilites
{

  template< typename INDEX_BASE_TYPE, typename REAL_TYPE >
  void jacobian( CellHexahedronUniformIJK< INDEX_BASE_TYPE, REAL_TYPE > const & cell, 
                 typename CellHexahedronUniformIJK< INDEX_BASE_TYPE, REAL_TYPE >::JacobianType & J )
  {
    typename CellHexahedronUniformIJK< INDEX_BASE_TYPE, REAL_TYPE >::DataType const & h = cell.getData();
    J = 0.5 * h;
  }
  
  template< typename INDEX_BASE_TYPE, typename REAL_TYPE >
  REAL_TYPE inverseJacobian( CellHexahedronUniformIJK< INDEX_BASE_TYPE, REAL_TYPE > const & cell, 
                             typename CellHexahedronUniformIJK< INDEX_BASE_TYPE, REAL_TYPE >::JacobianType & invJ )
  {
    typename CellHexahedronUniformIJK< INDEX_BASE_TYPE, REAL_TYPE >::DataType const & h = cell.getData();
    invJ = 2 / h;
    return 0.125 * h * h * h;
  }


  template< typename INDEX_BASE_TYPE, typename REAL_TYPE >
  void jacobian( CellHexahedronIJK< INDEX_BASE_TYPE, REAL_TYPE > const & cell, 
                 typename CellHexahedronIJK< INDEX_BASE_TYPE, REAL_TYPE >::JacobianType & J )
  {
    typename CellHexahedronIJK< INDEX_BASE_TYPE, REAL_TYPE >::DataType const & h = cell.getData();
    J[0] = 0.5 * h[0];
    J[1] = 0.5 * h[1];
    J[2] = 0.5 * h[2];
  }

  template< typename INDEX_BASE_TYPE, typename REAL_TYPE >
  REAL_TYPE inverseJacobian( CellHexahedronIJK< INDEX_BASE_TYPE, REAL_TYPE > const & cell, 
                        typename CellHexahedronIJK< INDEX_BASE_TYPE, REAL_TYPE >::JacobianType & invJ )
  {
    typename CellHexahedronIJK< INDEX_BASE_TYPE, REAL_TYPE >::DataType const & h = cell.getData();
    invJ[0] = 2 / h[0];
    invJ[1] = 2 / h[1];
    invJ[2] = 2 / h[2];
    return 0.125 * h[0] * h[1] * h[2];
  }

  template< typename CELL_TYPE >
  void jacobian( CELL_TYPE const &, 
                 typename CELL_TYPE::JacobianType & )
  {}



}
}