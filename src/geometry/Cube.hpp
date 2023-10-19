/**
 * @file Cube.hpp
 */

#pragma once

#include "common/ShivaMacros.hpp"
#include "common/IndexTypes.hpp"
#include "common/types.hpp"
namespace shiva
{
namespace geometry
{

/**
 * @brief Cube is a geometry type with a single length dimension in each 
 * direction. 
 * @tparam REAL_TYPE The floating point type.
 * 
 * A cube is a 3-dimensional volume that is...well...a cube.
 * <a href="https://mathworld.wolfram.com/Cube.html"> Cube</a>
 */
template< typename REAL_TYPE >
class Cube
{
public:
  /**
   * @brief The number of dimension of the cube.
   * @return The number dimension of the cube.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numDims() {return 3;};

  /// Alias for the floating point type for the Jacobian transformation.
  using JacobianType = Scalar< REAL_TYPE >;

  /// Alias for the floating point type for the data members of the cube.
  using DataType = REAL_TYPE;

  /// Alias for the floating point type for the coordinates of the cube.
  using CoordType = REAL_TYPE[3];

  /// Alias for the index type of the cube.
  using IndexType = MultiIndexRange< int, 2, 2, 2 >;

  /**
   * @brief Returns a boolean indicating whether the Jacobian is constant in 
   * the cell. This is used to determine whether the Jacobian should be 
   * computed once per cell or once per quadrature point.
   * @return true
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE bool jacobianIsConstInCell() { return true; }

  /**
   * @brief Returns the length dimension of the cube.
   * @return The length of the cube.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE DataType const & getLength() const { return m_length; }

  /**
   * @brief Sets the length dimension of the cube.
   * @param h The length of the cube.
   */
  constexpr SHIVA_HOST_DEVICE SHIVA_FORCE_INLINE void setLength( DataType const & h )
  { m_length = h; }


private:
  /// Data member that stores the length dimension of the cube.
  DataType m_length;
};


namespace utilities
{

/**
 * @brief Calculates the Jacobian transormation of a cube from a cube with
 * range from (-1,1) in each dimension.
 * @tparam REAL_TYPE The floating point type.
 * @param cell The cube object
 * @param J The Jacobian transformation.
 */
template< typename REAL_TYPE >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void jacobian( Cube< REAL_TYPE > const & cell,
               typename Cube< REAL_TYPE >::JacobianType::type & J )
{
  typename Cube< REAL_TYPE >::DataType const & h = cell.getLength();
  J = 0.5 * h;
}

/**
 * @brief Calculates the inverse Jacobian transormation of a cube from a cube with
 * range from (-1,1) in each dimension.
 * @tparam REAL_TYPE The floating point type.
 * @param cell The cube object
 * @param invJ The inverse Jacobian transformation.
 * @param detJ The determinant of the Jacobian transformation.
 */
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
