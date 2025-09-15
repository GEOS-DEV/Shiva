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

#include "LinearTransform.hpp"
#include "Scaling.hpp"
#include "UniformScaling.hpp"

namespace shiva
{
namespace geometry
{
namespace utilities
{

template< typename BASE_GEOMETRY, typename REAL_TYPE = typename BASE_GEOMETRY::RealType >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto
jacobian( BASE_GEOMETRY const & cell )
{
  typename BASE_GEOMETRY::JacobianType J;
  jacobian( cell, J );
  return J;
}

template< typename BASE_GEOMETRY, typename REAL_TYPE = typename BASE_GEOMETRY::RealType >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto
inverseJacobian( BASE_GEOMETRY const & cell )
{
  typename BASE_GEOMETRY::JacobianType invJ;
  REAL_TYPE detJ;
  inverseJacobian( cell, invJ, detJ );
  return make_tuple( detJ, invJ );
}

template< typename BASE_GEOMETRY, typename REAL_TYPE = typename BASE_GEOMETRY::RealType >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto
jacobian( BASE_GEOMETRY const & cell,
          REAL_TYPE const (&parentCoords)[3] )
{
  typename BASE_GEOMETRY::JacobianType J{0.0 };
  jacobian( cell, parentCoords, J );
  return J;
}

template< typename BASE_GEOMETRY, typename REAL_TYPE = typename BASE_GEOMETRY::RealType >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto
inverseJacobian( BASE_GEOMETRY const & cell,
                 REAL_TYPE const (&parentCoords)[3] )
{
  typename BASE_GEOMETRY::JacobianType invJ{ 0.0 };
  REAL_TYPE detJ;
  inverseJacobian( cell, parentCoords, invJ, detJ );
  return make_tuple( detJ, invJ );
}



} // namespace utilities
} // namespace geometry
} // namespace shiva
