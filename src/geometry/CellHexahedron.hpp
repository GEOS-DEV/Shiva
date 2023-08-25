
namespace shiva
{
template< typename INDEX_TYPE, typename REAL_TYPE >
class CellHexahedronUnstructured
{
public:
  using IndexType = INDEX_TYPE;
  using JacobianType = REAL_TYPE[3][3];
  using DataType = REAL_TYPE[8][3];

  REAL_TYPE calculateJacobianOfCell( JacobianType& ) const 
  { return 0;}

  REAL_TYPE calculateJacobianOfPoint( REAL_TYPE const (&parentCoords)[3], JacobianType & J ) const;

  DataType       & getData()       { return m_VertexCoords; }
  DataType const & getData() const { return m_VertexCoords; }


private:
  constexpr static bool jacobianIsConstInCell = false;
  DataType m_VertexCoords;
};

template< typename INDEX_TYPE, typename REAL_TYPE >
class CellHexahedronUniformIJK
{
public:
  using IndexType = INDEX_TYPE;
  using JacobianType = REAL_TYPE;
  using DataType = REAL_TYPE;

  REAL_TYPE calculateJacobianOfCell( JacobianType & J ) const
  {
    J = 0.5 * m_h;
    return 0.125 * m_h * m_h * m_h;
  }

  REAL_TYPE calculateJacobianOfPoint( REAL_TYPE const (&)[3], JacobianType & ) const
  { return 0;}

  DataType       & getData()       { return m_h; }
  DataType const & getData() const { return m_h; }

private:
  constexpr static bool jacobianIsConstInCell = true;
  DataType m_h;
};


template< typename INDEX_TYPE, typename REAL_TYPE >
class CellHexahedronIJK
{
public:
  using IndexType = INDEX_TYPE;
  using JacobianType = REAL_TYPE[3];
  using DataType = REAL_TYPE[3];

  REAL_TYPE calculateJacobianOfCell( JacobianType & J ) const
  {
    J[0] = 0.5 * m_h[0];
    J[1] = 0.5 * m_h[1];
    J[2] = 0.5 * m_h[2];
    return 0.125 * m_h[0] * m_h[1] * m_h[2];
  }
  
  REAL_TYPE calculateJacobianOfPoint( REAL_TYPE const (&)[3], JacobianType & ) const 
  { return 0;}

  DataType       & getData()       { return m_h; }
  DataType const & getData() const { return m_h; }

private:
  constexpr static bool jacobianIsConstInCell = true;
  DataType m_h;
};

}