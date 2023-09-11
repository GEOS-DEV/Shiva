#include "types/MultiIndex.hpp"
namespace shiva
{
namespace geometry
{

template< typename REAL_TYPE >
class Cube
{
public:
  using JacobianType = REAL_TYPE;
  using DataType = REAL_TYPE;

  DataType       & getData()       { return m_h; }
  DataType const & getData() const { return m_h; }

private:
  constexpr static bool jacobianIsConstInCell = true;
  DataType m_h;
};


namespace utilities
{

template< typename REAL_TYPE >
void jacobian( Cube< REAL_TYPE > const & cell, 
               typename Cube< REAL_TYPE >::JacobianType & J )
{
  typename Cube< REAL_TYPE >::DataType const & h = cell.getData();
  J = 0.5 * h;
}

template< typename REAL_TYPE >
REAL_TYPE inverseJacobian( Cube< REAL_TYPE > const & cell, 
                           typename Cube< REAL_TYPE >::JacobianType & invJ )
{
  typename Cube< REAL_TYPE >::DataType const & h = cell.getData();
  invJ = 2 / h;
  return 0.125 * h * h * h;
}


} // namespace utilities
} // namespace geometry
} // namespace shiva
