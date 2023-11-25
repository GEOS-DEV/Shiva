/**
 * @file MultiDimensionalSpan.hpp
 * @brief This file contains the implementation of the MultiDimensionalSpan class.
 */

#pragma once

#include "MultiDimensionalBase.hpp"

namespace shiva
{

/**
 * @struct MultiDimensionalSpan
 * @brief This struct provides a compile-time dimesion multidimensional array.
 * @tparam T This is the type of the data stored in the array.
 * @tparam DIMS These are the dimensions of the array.
 */
template< typename T, int ... DIMS >
struct MultiDimensionalSpan : public MultiDimensionalBase< MultiDimensionalSpan< T, DIMS ... >, T, DIMS ... >
{
  using Base = MultiDimensionalBase< MultiDimensionalSpan< T, DIMS ... >, T, DIMS ... >;

  /// The type of the an element in the array.
  using element_type = T;

  constexpr MultiDimensionalSpan( T const (&buffer)[ MultiDimensionalArrayHelper::size<DIMS...>() ] ):
    m_data( &(buffer[0]) )
  {}

  /// The data in the array.
  T * const m_data;
};

// template< typename T >
// using Scalar = MultiDimensionalSpan< T,1 >;

template< typename T, int DIM >
using MultiDimensionalSpan1d = MultiDimensionalSpan< T, DIM >;

template< typename T, int DIM1, int DIM2 >
using MultiDimensionalSpan2d = MultiDimensionalSpan< T, DIM1, DIM2 >;

template< typename T, int DIM1, int DIM2, int DIM3 >
using MultiDimensionalSpan3d = MultiDimensionalSpan< T, DIM1, DIM2,DIM3 >;

template< typename T, int ... DIMS >
using MultiDimensionalSpanNd = MultiDimensionalSpan< T, DIMS ... >;







} // namespace shiva
