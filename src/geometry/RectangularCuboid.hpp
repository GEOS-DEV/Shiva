#include "types/MultiIndex.hpp"
namespace shiva
{

namespace geometry
{
template< typename REAL_TYPE >
class RectangularCuboid
{
public:
  using JacobianType = REAL_TYPE[3];
  using DataType = REAL_TYPE[3];

  DataType       & getData()       { return m_h; }
  DataType const & getData() const { return m_h; }

private:
  constexpr static bool jacobianIsConstInCell = true;
  DataType m_h;
};


namespace utilities
{
  
template< typename REAL_TYPE >
void jacobian( RectangularCuboid< REAL_TYPE > const & cell, 
               typename RectangularCuboid< REAL_TYPE >::JacobianType & J )
{
  typename RectangularCuboid< REAL_TYPE >::DataType const & h = cell.getData();
  J[0] = 0.5 * h[0];
  J[1] = 0.5 * h[1];
  J[2] = 0.5 * h[2];
}

template< typename REAL_TYPE >
REAL_TYPE inverseJacobian( RectangularCuboid< REAL_TYPE > const & cell, 
                      typename RectangularCuboid< REAL_TYPE >::JacobianType & invJ )
{
  typename RectangularCuboid< REAL_TYPE >::DataType const & h = cell.getData();
  invJ[0] = 2 / h[0];
  invJ[1] = 2 / h[1];
  invJ[2] = 2 / h[2];
  return 0.125 * h[0] * h[1] * h[2];
}

} // namespace utilities
} // namespace geometry
} // namespace shiva