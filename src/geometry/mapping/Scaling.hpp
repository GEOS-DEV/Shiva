/**
 * @file Scaling.hpp
 */

#pragma once

#include "common/ShivaMacros.hpp"
#include "common/types.hpp"
namespace shiva
{

namespace geometry
{

/**
 * @brief Class definition for a Scaling geometry.
 * direction.
 * @tparam REAL_TYPE The floating point type.
 *
 * A "rectangular cuboid" is defined here as a 3-dimensional volume with 6 quadralatrial sides, 3 lengths in each direction with all corner
 *angles
 * being 90 degrees.
 * <a href="https://en.wikipedia.org/wiki/Rectangular_cuboid"> Rectangular Cuboid (Wikipedia)</a>

 */
template< typename REAL_TYPE >
class Scaling
{
public:

  /// The type used to represent the Jacobian transformation operator
  using JacobianType = CArray1d< REAL_TYPE, 3 >;

  /// Alias for the floating point type for the data members that represent the
  /// dimensions of the rectangular cuboid.
  using DataType = REAL_TYPE[3];

  /**
   * @brief Returns a boolean indicating whether the Jacobian is constant in
   * the cell. This is used to determine whether the Jacobian should be
   * computed once per cell or once per quadrature point.
   * @return true
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE bool jacobianIsConstInCell() { return true; }

  /**
   * @brief Returns the length dimension of the rectangular cuboid.
   * @param i The index representing the direction of the requested length.
   * @return The requested length of the rectangular cuboid.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE REAL_TYPE const & getLength( int const i ) const { return m_length[i]; }

  /**
   * @brief Returns the length dimensions of the rectangular cuboid.
   * @return The length dimensions of the rectangular cuboid.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE DataType const & getLengths() const { return m_length; }

  /**
   * @brief Sets the length dimension of the rectangular cuboid.
   * @param i The index representing the direction of the requested length.
   * @param h_i The requested length of the rectangular cuboid.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE void setLength( int const i, REAL_TYPE const & h_i )
  { m_length[i] = h_i; }

  /**
   * @brief Sets the length dimensions of the rectangular cuboid.
   * @param h The length dimensions of the rectangular cuboid.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE void setLength( DataType const & h )
  {
    m_length[0] = h[0];
    m_length[1] = h[1];
    m_length[2] = h[2];
  }


private:
  /// Data member that stores the length dimensions of the rectangular cuboid.
  DataType m_length;
};


namespace utilities
{

/**
 * @brief Calculates the Jacobian transformation for a rectangular cuboid.
 * @tparam REAL_TYPE The floating point type.
 * @param cell The rectangular cuboid for which the Jacobian is calculated.
 * @param J The Jacobian transformation operator.
 */
template< typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
jacobian( Scaling< REAL_TYPE > const & cell,
          typename Scaling< REAL_TYPE >::JacobianType::type & J )
{
  typename Scaling< REAL_TYPE >::DataType const & h = cell.getLengths();
  J[0] = 0.5 * h[0];
  J[1] = 0.5 * h[1];
  J[2] = 0.5 * h[2];
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
                                                                    typename Scaling< REAL_TYPE >::JacobianType::type & invJ,
                                                                    REAL_TYPE & detJ )
{
  typename Scaling< REAL_TYPE >::DataType const & h = cell.getLengths();
  invJ[0] = 2.0 / h[0];
  invJ[1] = 2.0 / h[1];
  invJ[2] = 2.0 / h[2];
  detJ = 0.125 * h[0] * h[1] * h[2];
}

} // namespace utilities
} // namespace geometry
} // namespace shiva