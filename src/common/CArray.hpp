/**
 * @file CArray.hpp
 * @brief This file contains the implementation of the CArray class.
 */

#pragma once


#include "common/ShivaMacros.hpp"

#include "MultiDimensionalArrayHelper.hpp"

#include <utility>



namespace shiva
{




/**
 * @struct CArray
 * @brief This struct provides a compile-time dimesion multidimensional array.
 * @tparam T This is the type of the data stored in the array.
 * @tparam DIMS These are the dimensions of the array.
 */
template< typename T, typename DATA_BUFFER, int ... DIMS >
struct CArray
{
  /// The type of the an element in the array.
  using element_type = T;

  /// The type of the data stored in the array.
  using value_type = std::remove_cv_t< T >;

  /// The type of the indices used to access the array.
  using index_type = int;

  /// The number of dimensions in the array.
  static inline constexpr int rank() { return sizeof ... ( DIMS ); }

  /// The dimensions of the array.
  template< int INDEX >
  static inline constexpr int extent() { return MultiDimensionalArrayHelper::get< INDEX, DIMS ... >(); }

  /// The size of the data in array...i.e. the product of the dimensions.
  static inline constexpr int size() { return ( DIMS * ... ); }

  template< typename U=DATA_BUFFER,
                     std::enable_if_t< std::is_pointer_v< U >, int > = 0 >
  constexpr CArray( T const (&buffer)[ MultiDimensionalArrayHelper::size<DIMS...>() ] ):
    m_data( &(buffer[0]) )
  {}

  template< typename ... U >
  constexpr CArray( U ... args ): m_data{ args ... }
  {}


  template< typename U=DATA_BUFFER >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE std::enable_if_t< std::is_pointer_v< U >, CArray & >
  operator=( T (&buffer)[ MultiDimensionalArrayHelper::size<DIMS...>() ] )
  {
    m_data = buffer;
    return *this;
  }

  /**
   * @brief Square bracket operator to access the data in a 1d array.
   * @tparam N rank of the array
   * @param i the index indicating the data offset to access.
   * @return reference to the value
   */
  template< int N=rank() >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE 
  std::enable_if_t< N==1, value_type & >
  operator[]( index_type const i )
  {
    return m_data[ i ];
  }

  /**
   * @brief Square bracket operator to access the data in a 1d array.
   * @tparam N rank of the array
   * @param i the index indicating the data offset to access.
   * @return reference to const value
   */
  template< int N=rank() >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE 
  std::enable_if_t< N==1, value_type const & >
  operator[]( index_type const i ) const
  {
    return m_data[ i ];
  }

  /**
   * @brief Templated operator() to access the data in the array.
   * @tparam ...INDICES The indices that specify the data to access.
   * @return A reference to the data at the specified indices.
   */
  template< int... INDICES >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE T& operator()( )
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ MultiDimensionalArrayHelper::linearIndexHelper<DIMS...>::template eval< INDICES... >() ];
  }

  /**
   * @brief Templated operator() to provide const access the data in the array.
   * @tparam ...INDICES The indices that specify the data to access.
   * @return A reference to const to the data at the specified indices.
   */
  template< int... INDICES >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE const T& operator()( ) const
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ MultiDimensionalArrayHelper::linearIndexHelper<DIMS...>::template eval< INDICES... >() ];
  }
  
  /**
   * @brief parentheses operator accessor to data in the array.
   * @tparam ...INDICES The type of the indices that specify the data to access.
   * @param ...indices The pack of indices that specify the data to access.
   * @return A reference to the data at the specified indices.
   */
  template< typename ... INDICES >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE T& operator()( INDICES... indices )
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ MultiDimensionalArrayHelper::linearIndexHelper<DIMS...>::eval( indices ... ) ];
  }

  /**
   * @brief parentheses operator access to const data in the array.
   * @tparam ...INDICES The type of the indices that specify the data to access.
   * @param ...indices The pack of indices that specify the data to access.
   * @return A reference to const data at the specified indices.
   */
  template< typename ... INDICES >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE const T& operator()( INDICES... indices ) const
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ MultiDimensionalArrayHelper::linearIndexHelper<DIMS...>::eval( indices ... ) ];
  }

  /// The data in the array.
//  T m_data[ size() ];
  DATA_BUFFER m_data;
};

template< typename T >
using Scalar = CArray< T, T[1], 1 >;

template< typename T, int DIM >
using CArray1d = CArray< T, T[DIM], DIM >;

template< typename T, int DIM1, int DIM2 >
using CArray2d = CArray< T, T[DIM1*DIM2], DIM1, DIM2 >;

template< typename T, int DIM1, int DIM2, int DIM3 >
using CArray3d = CArray< T, T[DIM1*DIM2*DIM3], DIM1, DIM2,DIM3 >;

template< typename T, int ... DIMS >
using CArrayNd = CArray< T, T[MultiDimensionalArrayHelper::size< DIMS ... >()], DIMS ... >;






template< typename T >
using ScalarView = CArray< T, T,1 >;

template< typename T, int DIM >
using CArrayView1d = CArray< T, T * const, DIM >;

template< typename T, int DIM1, int DIM2 >
using CArrayView2d = CArray< T, T * const , DIM1, DIM2 >;

template< typename T, int DIM1, int DIM2, int DIM3 >
using CArrayView3d = CArray< T, T * const, DIM1, DIM2,DIM3 >;

template< typename T, int ... DIMS >
using CArrayViewNd = CArray< T, T * const , DIMS ... >;



} // namespace shiva
