/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Copyright (c) 2023  Lawrence Livermore National Security LLC
 * Copyright (c) 2023  TotalEnergies
 * Copyright (c) 2023- Shiva Contributors
 * All rights reserved
 *
 * See Shiva/LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

/**
 * @file UniformScaling.hpp
 */

#pragma once

#include "common/ShivaMacros.hpp"
#include "common/IndexTypes.hpp"
#include "common/types.hpp"
#include "common/CArray.hpp"

namespace shiva
{
namespace geometry
{

/**
 * @brief UniformScaling is a transformation/mapping scales the size of a
 * geometric object uniformly in each direction.
 * @tparam REAL_TYPE The floating point type.
 */
template< typename REAL_TYPE, typename BASIS = void >
class UniformScaling
{
public:
  /// Alias for the floating point type for the transform.
  using RealType = REAL_TYPE;

  /// Alias for the floating point type for the Jacobian transformation.
  using JacobianType = Scalar< REAL_TYPE >;

  /// Alias for the floating point type for the data members of the cube.
  using DataType = REAL_TYPE;

  /**
   * @brief Returns a boolean indicating whether the Jacobian is constant in the
   * cell. This is used to determine whether the Jacobian should be computed once
   * per cell or once per quadrature point.
   * @return true
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE bool jacobianIsConstInCell() { return true; }

  /**
   * @brief Returns the length dimension of the cube.
   * @return The length of the cube.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE DataType const & getData() const { return m_length; }

  /**
   * @brief Sets the length dimension of the cube.
   * @return The length of the cube.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE DataType & setData(){ return m_length; }

  /**
   * @brief Sets the length dimension of the cube.
   * @param h The length of the cube.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE void setData( DataType const & h )
  { m_length = h; }


private:
  /// Data member that stores the length dimension of the cube.
  DataType m_length{1.0};
};


namespace utilities
{

/**
 * @brief Calculates the Jacobian transformation of a cube from a cube with
 * range from (-1,1) in each dimension.
 * @tparam REAL_TYPE The floating point type.
 * @param cell The cube object
 * @param J The Jacobian transformation.
 */
template< typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void jacobian( UniformScaling< REAL_TYPE > const & cell,
                                                             typename UniformScaling< REAL_TYPE >::JacobianType & J )
{
  typename UniformScaling< REAL_TYPE >::DataType const & h = cell.getData();
  J( 0 ) = 0.5 * h;
}

/**
 * @brief Calculates the inverse Jacobian transormation of a cube from a cube
 * with range from (-1,1) in each dimension.
 * @tparam REAL_TYPE The floating point type.
 * @param cell The cube object
 * @param invJ The inverse Jacobian transformation.
 * @param detJ The determinant of the Jacobian transformation.
 */
template< typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void inverseJacobian( UniformScaling< REAL_TYPE > const & cell,
                                                                    typename UniformScaling< REAL_TYPE >::JacobianType & invJ,
                                                                    REAL_TYPE & detJ )
{
  typename UniformScaling< REAL_TYPE >::DataType const & h = cell.getData();
  invJ( 0 ) = 2 / h;
  detJ = 0.125 * h * h * h;
}

} // namespace utilities
} // namespace geometry
} // namespace shiva
