#pragma once

#include "common/ShivaMacros.hpp"
#include "types/types.hpp"
namespace shiva
{

namespace geometry
{
template< typename REAL_TYPE >
class RectangularCuboid
{
public:
  using JacobianType = CArray1d<REAL_TYPE,3>;
  using DataType = REAL_TYPE[3];

  constexpr static bool jacobianIsConstInCell() { return true; }

  REAL_TYPE const & getLength( int const i ) const { return m_length[i]; }

  DataType const & getLengths() const { return m_length; }

  void setLength( int const i, REAL_TYPE const & h_i ) 
  { m_length[i] = h_i; }

  void setLength( DataType const & h ) 
  { 
    m_length[0] = h[0]; 
    m_length[1] = h[1]; 
    m_length[2] = h[2]; 
  }


private:
  DataType m_length;
};


namespace utilities
{
  
template< typename REAL_TYPE >
void jacobian( RectangularCuboid< REAL_TYPE > const & cell, 
               typename RectangularCuboid< REAL_TYPE >::JacobianType::type & J )
{
  typename RectangularCuboid< REAL_TYPE >::DataType const & h = cell.getLengths();
  J[0] = 0.5 * h[0];
  J[1] = 0.5 * h[1];
  J[2] = 0.5 * h[2];
}



template< typename REAL_TYPE >
void inverseJacobian( RectangularCuboid< REAL_TYPE > const & cell, 
                           typename RectangularCuboid< REAL_TYPE >::JacobianType::type & invJ,
                           REAL_TYPE & detJ )
{
  typename RectangularCuboid< REAL_TYPE >::DataType const & h = cell.getLengths();
  invJ[0] = 2.0 / h[0];
  invJ[1] = 2.0 / h[1];
  invJ[2] = 2.0 / h[2];
  detJ = 0.125 * h[0] * h[1] * h[2];
}

} // namespace utilities
} // namespace geometry
} // namespace shiva