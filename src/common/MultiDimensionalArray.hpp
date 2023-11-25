/**
 * @file MultiDimensionalArray.hpp
 * @brief This file contains the implementation of the MultiDimensionalArray class.
 */

#pragma once

#include "MultiDimensionalBase.hpp"

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

  // template< typename ... U >
  // constexpr MultiDimensionalArray( U ... args ): m_data{ args ... }
  // {}

  template <typename ... Ts,
            std::enable_if_t< ( sizeof...(Ts) != 0 && 
                                sizeof...(Ts) < Base::size() ) || 
                              !std::is_same_v<MultiDimensionalArray, std::decay_t<T>>, int> = 0>
  constexpr MultiDimensionalArray( Ts&&... args):
    m_data{ std::forward<Ts>(args)...}
  {}



  /// The data in the array.
  T m_data[Base::size()];
};

template< typename T >
using Scalar = MultiDimensionalArray< T,1 >;

template< typename T, int DIM >
using MultiDimensionalArray1d = MultiDimensionalArray< T, DIM >;

template< typename T, int DIM1, int DIM2 >
using MultiDimensionalArray2d = MultiDimensionalArray< T, DIM1, DIM2 >;

template< typename T, int DIM1, int DIM2, int DIM3 >
using MultiDimensionalArray3d = MultiDimensionalArray< T, DIM1, DIM2,DIM3 >;

template< typename T, int ... DIMS >
using MultiDimensionalArrayNd = MultiDimensionalArray< T, DIMS ... >;




template< typename T, int DIM >
using CArray1d = MultiDimensionalArray< T, DIM >;

template< typename T, int DIM1, int DIM2 >
using CArray2d = MultiDimensionalArray< T, DIM1, DIM2 >;

template< typename T, int DIM1, int DIM2, int DIM3 >
using CArray3d = MultiDimensionalArray< T, DIM1, DIM2,DIM3 >;

template< typename T, int ... DIMS >
using CArrayNd = MultiDimensionalArray< T, DIMS ... >;


} // namespace shiva
