#include "types/MultiIndex.hpp"
namespace shiva
{
template< typename INDEX_BASE_TYPE, typename REAL_TYPE >
class CellHexahedronUnstructured
{
public:
  using IndexType = INDEX_BASE_TYPE;
  using JacobianType = REAL_TYPE[3][3];
  using DataType = REAL_TYPE[8][3];

  DataType       & getData()       { return m_VertexCoords; }
  DataType const & getData() const { return m_VertexCoords; }

private:
  constexpr static bool jacobianIsConstInCell = false;
  DataType m_VertexCoords;
};

template< typename INDEX_BASE_TYPE, typename REAL_TYPE >
class CellHexahedronUniformIJK
{
public:
  using IndexType = MultiIndex<3,INDEX_BASE_TYPE>;
  using JacobianType = REAL_TYPE;
  using DataType = REAL_TYPE;

  DataType       & getData()       { return m_h; }
  DataType const & getData() const { return m_h; }

private:
  constexpr static bool jacobianIsConstInCell = true;
  DataType m_h;
};


template< typename INDEX_BASE_TYPE, typename REAL_TYPE >
class CellHexahedronIJK
{
public:
  using IndexType = MultiIndex<3,INDEX_BASE_TYPE>;
  using JacobianType = REAL_TYPE[3];
  using DataType = REAL_TYPE[3];

  DataType       & getData()       { return m_h; }
  DataType const & getData() const { return m_h; }

private:
  constexpr static bool jacobianIsConstInCell = true;
  DataType m_h;
};

}