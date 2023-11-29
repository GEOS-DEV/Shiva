/**
 * @file MultiDimensionalSlice.hpp
 * @brief This file contains the implementation of the MultiDimensionalSlice class.
 */

#pragma once

#include "MultiDimensionalBase.hpp"

namespace shiva
{


/**
 * @struct MultiDimensionalSlice
 * @brief This struct provides a compile-time dimesion multidimensional array.
 * @tparam T This is the type of the data stored in the array.
 * @tparam DIMS These are the dimensions of the array.
 */
template< typename T, int ... DIMS >
struct MultiDimensionalSlice : public MultiDimensionalBase< MultiDimensionalSlice< T, DIMS ... >, T, DIMS ... >
{
  using Base = MultiDimensionalBase< MultiDimensionalSlice< T, DIMS ... >, T, DIMS ... >;

  /// The type of the an element in the array.
  using element_type = T;


  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE 
  MultiDimensionalSlice( T * const buffer ):
    m_data( buffer )
  {}


  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE 
  MultiDimensionalSlice< T, DIMS ... > toSpanConst() const
  {
    return MultiDimensionalSlice< std::add_const_t<T>, DIMS ... >( m_data );
  }


  /// The data in the array.
  T * const m_data;
};

// template< typename T >
// using Scalar = MultiDimensionalSlice< T,1 >;

template< typename T, int DIM >
using MultiDimensionalSlice1d = MultiDimensionalSlice< T, DIM >;

template< typename T, int DIM1, int DIM2 >
using MultiDimensionalSlice2d = MultiDimensionalSlice< T, DIM1, DIM2 >;

template< typename T, int DIM1, int DIM2, int DIM3 >
using MultiDimensionalSlice3d = MultiDimensionalSlice< T, DIM1, DIM2,DIM3 >;

template< typename T, int ... DIMS >
using MultiDimensionalSliceNd = MultiDimensionalSlice< T, DIMS ... >;







} // namespace shiva
