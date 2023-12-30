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
 * @file NCube.hpp
 */

#pragma once

#include "common/ShivaMacros.hpp"
#include "common/IndexTypes.hpp"
#include "common/types.hpp"
#include "common/MathUtilities.hpp"

namespace shiva
{
namespace geometry
{

/**
 * @brief NCube represents a n-cube.
 * @tparam REAL_TYPE The floating point type.
 * @tparam N The number of dimensions of the n-cube.
 * @tparam MIN The minimum coordinate of the n-cube.
 * @tparam MAX The maximum coordinate of the n-cube.
 * @tparam DIVISOR The divisor of the coordinates of the n-cube. This is
 * required because the coordinates of the n-cube are integers, but the
 * coordinates of the n-cube are floating point numbers.
 *
 * A n-cube is a generalization of a cube in n-dimensions.
 * <a href="https://en.wikipedia.org/wiki/Hypercube"> Wikipedia Hypercube</a>
 * <a href="https://mathworld.wolfram.com/Hypercube.html"> Wolfram Hypercube</a>
 */
template< typename REAL_TYPE, int N, int MIN, int MAX, int DIVISOR >
class NCube
{
public:

  static_assert( MIN < MAX, "MIN must be less than MAX" );
  static_assert( DIVISOR > 0, "DIVISOR must be greater than 0" );

  /**
   * @brief The number of dimension of the cube.
   * @return The number dimension of the cube.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numDims() {return N;}

  /// Alias for the floating point type.
  using RealType = REAL_TYPE;

  /// Alias for the floating point type for the coordinates.
  using CoordType = REAL_TYPE[N];

  // /// Alias for the index type of the cube.
  // using IndexType = MultiIndexRange< int, 2, 2, 2 >;

  /**
   * @brief Returns the number of m-cubes in the n-cube.
   * @tparam M The number of dimensions of the m-cube.
   * @return The number of m-cubes in the n-cube. An m-cube is the lower
   * dimensional object contained in the n-cube. For instance, the 0-cube is a
   * vertex, the 1-cube is a line, the 2-cube is a square, the 3-cube is a cube,
   * etc. M must be less than or equal to N
   */
  template< int M >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numMCubes()
  {
    static_assert( M <= N, "M must be less than or equal to N" );
    return mathUtilities::pow< int, N - M >( 2 ) * mathUtilities::factorial< int, N >::value / ( mathUtilities::factorial< int, M >::value * mathUtilities::factorial< int, N - M >::value );
  }

  /**
   * @brief Returns the number of vertices (0-cube) in the n-cube.
   * @return The number of vertices (0-cube) in the n-cube.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numVertices()
  {
    return numMCubes< 0 >();
  }

  /**
   * @brief Returns the number of edges (1-cube) in the n-cube.
   * @return The number of edges (1-cube) in the n-cube.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numEdges()
  {
    if constexpr ( N > 0 )
    {
      return numMCubes< 1 >();
    }
    else
    {
      return 0;
    }
  }

  /**
   * @brief Returns the number of faces (2-cube) in the n-cube.
   * @return The number of faces (2-cube) in the n-cube.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numFaces()
  {
    if constexpr ( N > 1 )
    {
      return numMCubes< 2 >();
    }
    else
    {
      return 0;
    }
  }

  /**
   * @brief Returns the number of cells (3-cube) in the n-cube.
   * @return The number of cells (3-cube) in the n-cube.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numCells()
  {
    if constexpr ( N > 2 )
    {
      return numMCubes< 3 >();
    }
    else
    {
      return 0;
    }
  }

  /**
   * @brief Returns the number of hyperfaces (n-1-cube) in the n-cube.
   * @return The number of hyperfaces (n-1-cube) in the n-cube.
   *
   * The hyperfaces can be considered the number of n-1 dimensional objects in
   * an n-cube. For instance, the hyperfaces of a cube are the faces of the cube.
   * The hyperfaces of a square are the edges of the square. The hyperfaces of a
   * line are the vertices of the line.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numHyperFaces()
  {
    return 2 * N;
  }

  /**
   * @brief returns the minimum coordinate of the n-cube.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE minCoord()
  {
    return static_cast< RealType >(MIN) / DIVISOR;
  }

  /**
   * @brief returns the maximum coordinate of the n-cube.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE maxCoord()
  {
    return static_cast< RealType >(MAX) / DIVISOR;
  }


  /**
   * @brief Returns the length dimension of the cube.
   * @return The length of the cube.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE length()
  {
    return static_cast< RealType >( MAX - MIN ) / DIVISOR;
  }


  /**
   * @brief Returns the volume of the n-cube.
   * @return The volume of the n-cube.
   */
  template< int DIM = N >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE
  typename std::enable_if< ( DIM > 0 ), REAL_TYPE >::type
  volume()
  {
    return volumeHelper();
  }


private:
  /**
   * @brief Helper recusive function for volume calculation.
   * @tparam DIM The current dimension for the recursion.
   * @return The volume of the n-cube.
   */
  template< int DIM = N >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE
  volumeHelper()
  {
    if constexpr ( DIM == 0 )
    {
      return 1;
    }
    else
    {
      return length() * volumeHelper< DIM - 1 >();
    }
  }

};

/**
 * @brief Alias for a 2-cube.
 * @tparam REAL_TYPE The floating point type.
 */
template< typename REAL_TYPE >
using Square = NCube< REAL_TYPE, 2, -1, 1, 1 >;

/**
 * @brief Alias for a 3-cube.
 * @tparam REAL_TYPE The floating point type.
 */
template< typename REAL_TYPE >
using Cube = NCube< REAL_TYPE, 3, -1, 1, 1 >;

} // namespace geometry
} // namespace shiva
