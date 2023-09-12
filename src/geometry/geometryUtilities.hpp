#pragma once

namespace shiva
{
namespace geometry
{
namespace utilities
{

template< template<typename> typename SHAPE_TYPE, typename REAL_TYPE >
auto jacobian( SHAPE_TYPE< REAL_TYPE > const & cell )
{
  typename SHAPE_TYPE< REAL_TYPE >::JacobianType J;
  jacobian( cell, J.data );
  return J;
}

template< template<typename> typename SHAPE_TYPE, typename REAL_TYPE >
auto inverseJacobian( SHAPE_TYPE< REAL_TYPE > const & cell )
{
  typename SHAPE_TYPE< REAL_TYPE >::JacobianType invJ;
  REAL_TYPE const detJ = inverseJacobian( cell, invJ.data );
  return make_tuple( detJ, invJ );
}


} // namespace utilities
} // namespace geometry
} // namespace shiva