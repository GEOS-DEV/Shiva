#pragma once

namespace shiva
{
namespace geometry
{
namespace utilities
{

template< template< typename > typename SHAPE_TYPE, typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto 
jacobian( SHAPE_TYPE< REAL_TYPE > const & cell )
{
  typename SHAPE_TYPE< REAL_TYPE >::JacobianType J;
  jacobian( cell, J.data );
  return J;
}

template< template< typename > typename SHAPE_TYPE, typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto 
inverseJacobian( SHAPE_TYPE< REAL_TYPE > const & cell )
{
  typename SHAPE_TYPE< REAL_TYPE >::JacobianType invJ;
  REAL_TYPE detJ;
  inverseJacobian( cell, invJ.data, detJ );
  return make_tuple( detJ, invJ );
}

template< template< typename > typename SHAPE_TYPE, typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto 
jacobian( SHAPE_TYPE< REAL_TYPE > const & cell,
               REAL_TYPE const (&parentCoords)[3] )
{
  typename SHAPE_TYPE< REAL_TYPE >::JacobianType J{ { {0} } };
  jacobian( cell, parentCoords, J.data );
  return J;
}

template< template< typename > typename SHAPE_TYPE, typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto 
inverseJacobian( SHAPE_TYPE< REAL_TYPE > const & cell,
                      REAL_TYPE const (&parentCoords)[3] )
{
  typename SHAPE_TYPE< REAL_TYPE >::JacobianType invJ{ { {0} } };
  REAL_TYPE detJ;
  inverseJacobian( cell, parentCoords, invJ.data, detJ );
  return make_tuple( detJ, invJ );
}



} // namespace utilities
} // namespace geometry
} // namespace shiva
