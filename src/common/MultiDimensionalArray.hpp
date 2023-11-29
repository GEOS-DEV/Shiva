/**
 * @file MultiDimensionalArray.hpp
 * @brief This file contains the implementation of the MultiDimensionalArray class.
 */

#pragma once

#include "MultiDimensionalBase.hpp"
#include "MultiDimensionalSlice.hpp"

namespace shiva
{

/**
 * @struct MultiDimensionalArray
 * @brief This struct provides a compile-time dimesion multidimensional array.
 * @tparam T This is the type of the data stored in the array.
 * @tparam DIMS These are the dimensions of the array.
 */
template< typename T, int ... DIMS >
struct MultiDimensionalArray : MultiDimensionalBase< MultiDimensionalArray< T, DIMS ... >, T, DIMS ... >
{
  using Base = MultiDimensionalBase< MultiDimensionalArray< T, DIMS ... >, T, DIMS ... >;

  /// The type of the an element in the array.
  using element_type = T;

  /// The data in the array.
  T m_data[Base::size()];
};

template< typename T >
using Scalar = MultiDimensionalArray< T,1 >;

template< typename T, int ... DIMS >
using mdArray = MultiDimensionalArray< T, DIMS ... >;


} // namespace shiva
