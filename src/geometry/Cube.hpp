#pragma once

#include "common/ShivaMacros.hpp"
#include "types/IndexTypes.hpp"
#include "types/types.hpp"
namespace shiva
{
namespace geometry
{

template< typename REAL_TYPE >
class Cube
{
public:
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int Dimension() {return 3;};
  using JacobianType = Scalar< REAL_TYPE >;
  using DataType = REAL_TYPE;
  using CoordType = REAL_TYPE[3];
  using IndexType = MultiIndexRange< int, 2, 2, 2 >;

  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE bool jacobianIsConstInCell() { return true; }

  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE DataType const & getLength() const { return m_length; }

  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE void setLength( DataType const & h )
  { m_length = h; }


private:
  DataType m_length;
};


namespace utilities
{

template< typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void jacobian( Cube< REAL_TYPE > const & cell,
               typename Cube< REAL_TYPE >::JacobianType::type & J )
{
  typename Cube< REAL_TYPE >::DataType const & h = cell.getLength();
  J = 0.5 * h;
}


template< typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void inverseJacobian( Cube< REAL_TYPE > const & cell,
                      typename Cube< REAL_TYPE >::JacobianType::type & invJ,
                      REAL_TYPE & detJ )
{
  typename Cube< REAL_TYPE >::DataType const & h = cell.getLength();
  invJ = 2 / h;
  detJ = 0.125 * h * h * h;
}

} // namespace utilities
} // namespace geometry
} // namespace shiva
