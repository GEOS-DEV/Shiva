/*
 * ------------------------------------------------------------------------------------------------------------
 * SPDX-License-Identifier: LGPL-2.1-only
 *
 * Copyright (c) 2018-2020 Lawrence Livermore National Security LLC
 * Copyright (c) 2018-2020 The Board of Trustees of the Leland Stanford Junior University
 * Copyright (c) 2018-2020 TotalEnergies
 * Copyright (c) 2019-     GEOSX Contributors
 * All rights reserved
 *
 * See top level LICENSE, COPYRIGHT, CONTRIBUTORS, NOTICE, and ACKNOWLEDGEMENTS files for details.
 * ------------------------------------------------------------------------------------------------------------
 */

#ifndef GEOS_FINITEELEMENT_ELEMENTFORMULATIONS_ELEMENTFORMULATIONS_LAGRANGEBASIS1_HPP_
#define GEOS_FINITEELEMENT_ELEMENTFORMULATIONS_ELEMENTFORMULATIONS_LAGRANGEBASIS1_HPP_

/**
 * @file LagrangeBasis1.hpp
 */

#include "common/DataTypes.hpp"

namespace geos
{
namespace finiteElement
{

/**
 * This class contains the implementation for a first order (linear) Lagrange
 * polynomial basis. The parent space is defined by:
 *
 *                 o-------------o  ---> xi
 *  Index:         0             1
 *  Coordinate:   -1             1
 *
 */
class LagrangeBasis1
{
public:
  /// The number of support points for the basis
  constexpr static localIndex numSupportPoints = 2;


  /**
   * @brief Calculate the parent coordinates for the xi0 direction, given the
   *   linear index of a support point.
   * @param supportPointIndex The linear index of support point
   * @return parent coordinate in the xi0 direction.
   */
  GEOS_HOST_DEVICE
  inline
  constexpr static real64 parentSupportCoord( const localIndex supportPointIndex )
  {
    return -1.0 + 2.0 * (supportPointIndex & 1);
  }

  /**
   * @brief The value of the basis function for a support point evaluated at a
   *   point along the axes.
   * @param index The index of the support point.
   * @param xi The coordinate at which to evaluate the basis.
   * @return The value of basis function.
   */
  GEOS_HOST_DEVICE
  inline
  constexpr static real64 value( const int index,
                                 const real64 xi )
  {
    return 0.5 + 0.5 * xi * parentSupportCoord( index );
  }


  /**
   * @brief The gradient of the basis function for a support point evaluated at
   *   a point along the axes.
   * @param index The index of the support point associated with the basis
   *   function.
   * @param xi The coordinate at which to evaluate the gradient.
   * @return The gradient of basis function.
   */
  GEOS_HOST_DEVICE
  inline
  constexpr static real64 gradient( const int index,
                                    const real64 xi )
  {
    GEOS_UNUSED_VAR( xi );
    return 0.5 * parentSupportCoord( index );
  }


}
}


#endif /* GEOS_FINITEELEMENT_ELEMENTFORMULATIONS_ELEMENTFORMULATIONS_LAGRANGEBASIS1_HPP_ */
