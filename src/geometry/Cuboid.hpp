#pragma once

namespace shiva
{

namespace geometry
{

template< typename REAL_TYPE >
class Cuboid
{
public:
  using JacobianType = REAL_TYPE[3][3];
  using DataType = REAL_TYPE[8][3];

  DataType       & getData()       { return m_VertexCoords; }
  DataType const & getData() const { return m_VertexCoords; }

private:
  constexpr static bool jacobianIsConstInCell = false;
  DataType m_VertexCoords;
};

namespace utilities
{
template< typename CELL_TYPE >
void jacobian( CELL_TYPE const &, 
               typename CELL_TYPE::JacobianType & )
{}

} //namespace utilities
} // namespace geometry
} // namespace shiva