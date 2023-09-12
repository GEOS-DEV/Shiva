#pragma once

#include "common/ShivaMacros.hpp"
#include "types/MultiIndex.hpp"
#include "types/types.hpp"
namespace shiva
{
namespace geometry
{

template< typename REAL_TYPE >
class Cube
{
public:
  using JacobianType = Scalar<REAL_TYPE>;
  using DataType = REAL_TYPE;

  constexpr static bool jacobianIsConstInCell() { return true; }

  DataType       & getData()       { return m_h; }
  DataType const & getData() const { return m_h; }

private:
  DataType m_h;
};


namespace utilities
{

template< typename REAL_TYPE >
void jacobian( Cube< REAL_TYPE > const & cell, 
               typename Cube< REAL_TYPE >::JacobianType::type & J )
{
  typename Cube< REAL_TYPE >::DataType const & h = cell.getData();
  J = 0.5 * h;
}


template< typename REAL_TYPE >
void inverseJacobian( Cube< REAL_TYPE > const & cell, 
                      typename Cube< REAL_TYPE >::JacobianType::type & invJ,
                      REAL_TYPE & detJ )
{
  typename Cube< REAL_TYPE >::DataType const & h = cell.getData();
  invJ = 2 / h;
  detJ = 0.125 * h * h * h;
}

} // namespace utilities
} // namespace geometry
} // namespace shiva
