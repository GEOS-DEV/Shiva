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

#pragma once

#include "common/SequenceUtilities.hpp"
#include "common/ShivaMacros.hpp"
#include "common/types.hpp"

#include "functions/bases/BasisProduct.hpp"


#include <utility>

namespace shiva
{
namespace geometry
{

/**
 * @class InterpolatedShape
 * @brief Defines a class that provides static functions to calculate quantities
 * required from the parent element in a finite element method.
 * @tparam REAL_TYPE The floating point type to use
 * @tparam STANDARD_GEOMETRY The standard geometric form of the interpolated shape (domain)
 * @tparam FUNCTIONAL_SPACE_TYPE The functional space type
 * @tparam BASIS_TYPE Pack of basis types to apply to each direction of the
 * parent element. There should be a basis defined for each direction.
 */
template< typename REAL_TYPE, typename STANDARD_GEOMETRY, typename ... BASIS_TYPE >
class InterpolatedShape
{

public:

  /// The type used to represent the cell/geometry
  using StandardGeom = STANDARD_GEOMETRY;
//  using FunctionalSpaceType = FUNCTIONAL_SPACE_TYPE;
//  using IndexType = typename Geometry::IndexType;

  /// Alias for the floating point type
  using RealType = REAL_TYPE;

  /// Alias for the type that represents a coordinate
  using CoordType = typename StandardGeom::CoordType;

  /// The type used to represent the product of basis functions
  using BasisCombinationType = functions::BasisProduct< REAL_TYPE, BASIS_TYPE ... >;

  /// The number of dimensions on the InterpolatedShape
  static inline constexpr int numDims = sizeof...(BASIS_TYPE);

  /// The number of vertices on the InterpolatedShape
  static inline constexpr int numVertices = StandardGeom::numVertices();

  /// alias for an integer_sequence that contains the number of support points
  /// in each basis.
  using numSupportPointsSequence = std::integer_sequence< int, BASIS_TYPE::numSupportPoints... >;

  /// Instantiation of the integer sequence that holds the number of support points in each basis.
  static inline constexpr numSupportPointsSequence basisSupportCounts{};



  static_assert( numDims == StandardGeom::numDims(), "numDims mismatch between cell and number of basis specified" );

  /// @copydoc functions::BasisProduct::supportLoop
  template< typename FUNC >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE void
  supportLoop( FUNC && func )
  {
    BasisCombinationType::supportLoop( std::forward< FUNC >( func ) );
  }

  /**
   * @copydoc functions::BasisProduct::value
   */
  template< int ... BASIS_FUNCTION_INDICES, typename COORD_TYPE = RealType[numDims] >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE RealType
  value( COORD_TYPE const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );
    return ( BasisCombinationType::template value< BASIS_FUNCTION_INDICES... >( parentCoord ) );
  }

  /**
   * @copydoc functions::BasisProduct::gradient
   */
  template< int ... BASIS_FUNCTION_INDICES, typename COORD_TYPE = RealType[numDims] >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE CArrayNd< RealType, numDims >
  gradient( COORD_TYPE const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );
    return ( BasisCombinationType::template gradient< BASIS_FUNCTION_INDICES... >( parentCoord ) );
  }
};


} // namespace geometry
} // namespace shiva
