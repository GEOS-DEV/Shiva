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
 * @file NSimplex.hpp
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
 * @brief NSimplex represents a n-simplex.
 * @tparam REAL_TYPE The floating point type.
 * @tparam N The number of dimensions of the n-simplex.
 * @tparam MIN The minimum coordinate of the n-simplex.
 * @tparam MAX The maximum coordinate of the n-simplex.
 * @tparam DIVISOR The divisor of the coordinates of the n-simplex. This is
 * required because the coordinates of the n-simplex are integers, but the
 * coordinates of the n-simplex are floating point numbers.
 *
 * A n-simplex is a generalization of a triangle (n = 2) or tetrahedron (n = 3)
 * to arbitrary dimensions
 * <a href="https://en.wikipedia.org/wiki/Simplex"> Wikipedia Simplex</a>
 */
template< typename REAL_TYPE, int N, int MIN, int MAX, int DIVISOR >
class NSimplex
{
public:

  static_assert( MIN < MAX, "MIN must be less than MAX" );
  static_assert( DIVISOR > 0, "DIVISOR must be greater than 0" );

  /**
   * @brief The number of dimension of the simplex.
   * @return The number dimension of the simplex.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numDims() {return N;}

  /// Alias for the floating point type.
  using RealType = REAL_TYPE;

  /// Alias for the floating point type for the coordinates.
  using CoordType = REAL_TYPE[N];

  /**
   * @brief Returns the number of m-simplexes in the n-simplex.
   * @tparam M The number of dimensions of the m-simplex.
   * @return The number of m-simplexes in the n-simplex. An m-simplex is the lower
   * dimensional object contained in the n-simplex. For instance, the 0-simplex is a
   * vertex, the 1-simplex is a line, the 2-simplex is a triangle, the 3-simplex is
   * a tetrahedron, etc. M must be less than or equal to N
   */
  template< int M >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numMSimplexes()
  {
    static_assert( M <= N, "M must be less than or equal to N" );
    return mathUtilities::binomialCoefficient< int, N + 1, M + 1 >::value;
  }

  /**
   * @brief Returns the number of vertices (0-simplex) in the n-simplex.
   * @return The number of vertices (0-simplex) in the n-simplex.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numVertices()
  {
    return numMSimplexes< 0 >();
  }

  /**
   * @brief Returns the number of edges (1-simplex) in the n-simplex.
   * @return The number of edges (1-simplex) in the n-simplex.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numEdges()
  {
    if constexpr ( N > 0 )
    {
      return numMSimplexes< 1 >();
    }
    else
    {
      return 0;
    }
  }

  /**
   * @brief Returns the number of faces (2-simplex) in the n-simplex.
   * @return The number of faces (2-simplex) in the n-simplex.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numFaces()
  {
    if constexpr ( N > 1 )
    {
      return numMSimplexes< 2 >();
    }
    else
    {
      return 0;
    }
  }

  /**
   * @brief Returns the number of cells (3-simplex) in the n-simplex.
   * @return The number of cells (3-simplex) in the n-simplex.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numCells()
  {
    if constexpr ( N > 2 )
    {
      return numMSimplexes< 3 >();
    }
    else
    {
      return 0;
    }
  }

  /**
   * @brief Returns the number of hyperfaces (n-1-simplex) in the n-simplex.
   * @return The number of hyperfaces (n-1-simplex) in the n-simplex.
   *
   * The hyperfaces can be considered the number of n-1 dimensional objects in
   * an n-simplex. For instance, the hyperfaces of a tetrahedron are the faces of
   * the tetrahedron. The hyperfaces of a triangle are the edges of the triangle.
   * The hyperfaces of a line are the vertices of the line.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int numHyperFaces()
  {
    return numMSimplexes< N - 1 >();
  }

  /**
   * @brief returns the minimum coordinate of the n-simplex.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE minCoord()
  {
    return static_cast< RealType >(MIN) / DIVISOR;
  }

  /**
   * @brief returns the maximum coordinate of the n-simplex.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE maxCoord()
  {
    return static_cast< RealType >(MAX) / DIVISOR;
  }


  /**
   * @brief Returns the length dimension of the simplex.
   * @return The length of the simplex.
   */
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE length()
  {
    return static_cast< RealType >( MAX - MIN ) / DIVISOR;
  }


  /**
   * @brief Returns the volume of the n-simplex.
   * @return The volume of the n-simplex.
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
   * @return The volume of the n-simplex.
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
      return length() / DIM * volumeHelper< DIM - 1 >();
    }
  }

};

/**
 * @brief Alias for a 2-simplex.
 * @tparam REAL_TYPE The floating point type.
 */
template< typename REAL_TYPE >
using Triangle = NSimplex< REAL_TYPE, 2, 0, 1, 1 >;

/**
 * @brief Alias for a 3-simplex
 * @tparam REAL_TYPE The floating point type.
 */
template< typename REAL_TYPE >
using Tetrahedron = NSimplex< REAL_TYPE, 3, 0, 1, 1 >;

} // namespace geometry
} // namespace shiva
