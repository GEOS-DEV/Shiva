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
 * @file Scaling.hpp
 */

#pragma once

#include "shiva/common/ShivaMacros.hpp"
#include "shiva/common/types.hpp"
#include "shiva/common/CArray.hpp"

#include <type_traits>
namespace shiva
{

namespace geometry
{

/**
 * @brief Class definition for a Scaling geometry.
 * direction.
 * @tparam REAL_TYPE The floating point type.
 *
 * A "rectangular cuboid" is defined here as a 3-dimensional volume with 6
 * quadralatrial sides, 3 lengths in each direction with all corner angles being
 * 90 degrees.
 * <a href="https://en.wikipedia.org/wiki/Rectangular_cuboid"> Rectangular
 * Cuboid (Wikipedia)</a>

 */
template< typename REAL_TYPE, typename BASIS = void >
class Scaling
{
public:

  /// Alias for the floating point type for the transform.
  using RealType = REAL_TYPE;

  /// The type used to represent the Jacobian transformation operator
  using JacobianType = CArrayNd< REAL_TYPE, 3 >;

  /// Alias for the floating point type for the data members that represent the
  /// dimensions of the rectangular cuboid.
  using DataType = REAL_TYPE[3];

  /**
   * @brief Returns a boolean indicating whether the Jacobian is constant in the
   * cell. This is used to determine whether the Jacobian should be computed once
   * per cell or once per quadrature point.
   * @return true
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE bool jacobianIsConstInCell() { return true; }

  /**
   * @brief Returns the length dimensions of the rectangular cuboid.
   * @return A const reference to the stored length dimensions.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE DataType const & getData() const { return m_length; }

  /**
   * @brief Provides mutable access to the length dimensions of the rectangular cuboid.
   * @return A reference to the stored length dimensions.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE DataType & getData() { return m_length; }

  /**
   * @brief provides a reference to the member data.
   * @return a mutable reference to the member data.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE DataType & setData() { return m_length; }


  /**
   * @brief Sets the length dimensions of the rectangular cuboid.
   * @param h The length dimensions of the rectangular cuboid.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE void setData( DataType const & h )
  {
    m_length[0] = h[0];
    m_length[1] = h[1];
    m_length[2] = h[2];
  }


private:
  /// Data member that stores the length dimensions of the rectangular cuboid.
  DataType m_length{1.0, 1.0, 1.0};
};


namespace utilities
{

/**
 * @brief Calculates the Jacobian transformation for a rectangular cuboid.
 * @tparam REAL_TYPE The floating point type.
 * @param[in] cell The rectangular cuboid for which the Jacobian is calculated.
 * @param[out] J The diagonal Jacobian entries, overwritten with half the current lengths.
 */
template< typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
jacobian( Scaling< REAL_TYPE > const & cell,
          typename Scaling< REAL_TYPE >::JacobianType & J )
{
  typename Scaling< REAL_TYPE >::DataType const & h = cell.getData();
  J( 0 ) = 0.5 * h[0];
  J( 1 ) = 0.5 * h[1];
  J( 2 ) = 0.5 * h[2];
}


/**
 * @brief Calculates the constant Jacobian, independent of quadrature arguments.
 * @tparam QUADRATURE The quadrature type, ignored because the Jacobian is constant.
 * @tparam QA The quadrature indices, ignored for any index pack length, including zero.
 * @tparam REAL_TYPE The floating point type.
 * @param[in] cell The rectangular cuboid for which the Jacobian is calculated.
 * @param[out] J The diagonal Jacobian entries, overwritten with half the current lengths.
 *
 * The constraint excludes QUADRATURE equal to REAL_TYPE so explicit scalar calls
 * select the non-quadrature overload:
 * @code
 * jacobian< REAL_TYPE >( cell, J );
 * @endcode
 */
template< typename QUADRATURE,
          int ... QA,
          typename REAL_TYPE,
          std::enable_if_t< !std::is_same_v< QUADRATURE, REAL_TYPE >, int > = 0 >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
jacobian( Scaling< REAL_TYPE > const & cell,
          typename Scaling< REAL_TYPE >::JacobianType & J )
{
  jacobian( cell, J );
}

/**
 * @brief Calculates the inverse Jacobian transformation and detJ for a
 * rectangular cuboid.
 * @tparam REAL_TYPE The floating point type.
 * @param cell The rectangular cuboid for which the inverse Jacobian is
 * calculated.
 * @param invJ The inverse Jacobian transformation operator.
 * @param detJ The determinant of the Jacobian transformation operator.
 */
template< typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void inverseJacobian( Scaling< REAL_TYPE > const & cell,
                                                                    typename Scaling< REAL_TYPE >::JacobianType & invJ,
                                                                    REAL_TYPE & detJ )
{
  typename Scaling< REAL_TYPE >::DataType const & h = cell.getData();
  invJ( 0 ) = 2.0 / h[0];
  invJ( 1 ) = 2.0 / h[1];
  invJ( 2 ) = 2.0 / h[2];
  detJ = 0.125 * h[0] * h[1] * h[2];
}

} // namespace utilities
} // namespace geometry
} // namespace shiva
