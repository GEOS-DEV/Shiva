/**
 * @file MultiDimensionalBase.hpp
 * @brief This file contains the implementation of the MultiDimensionalBase class.
 */

#pragma once


#include "common/ShivaMacros.hpp"

#include "MultiDimensionalArrayHelper.hpp"

#include <utility>



namespace shiva
{

/**
 * @struct MultiDimensionalBase
 * @brief This struct provides a compile-time dimesion multidimensional array.
 * @tparam T This is the type of the data stored in the array.
 * @tparam DIMS These are the dimensions of the array.
 */
template< typename MD_LEAF, typename T, int ... DIMS >
struct MultiDimensionalBase
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

  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE T * data() const
  { 
    return static_cast< MD_LEAF const & >(*this).m_data; 
  }

  /**
   * @brief Square bracket operator to access the data in a 1d array.
   * @tparam N rank of the array
   * @param i the index indicating the data offset to access.
   * @return reference to const value
   */
  template< int N=rank() >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE 
  std::enable_if_t< N==1, T & >
  operator[]( index_type const i ) const
  {
    return data()[ i ];
  }

  template< int N=rank() >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE 
  std::enable_if_t< N>1, T & >
  operator[]( index_type const i ) const
  {
    return data()[ i ];
  }


  /**
   * @brief Templated operator() to provide const access the data in the array.
   * @tparam ...INDICES The indices that specify the data to access.
   * @return A reference to const to the data at the specified indices.
   */
  template< int... INDICES >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE T & operator()( ) const
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return data()[ MultiDimensionalArrayHelper::linearIndexHelper<DIMS...>::template eval< INDICES... >() ];
  }

  /**
   * @brief parentheses operator access to const data in the array.
   * @tparam ...INDICES The type of the indices that specify the data to access.
   * @param ...indices The pack of indices that specify the data to access.
   * @return A reference to const data at the specified indices.
   */
  template< typename ... INDICES >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE T & operator()( INDICES... indices ) const
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return data()[ MultiDimensionalArrayHelper::linearIndexHelper<DIMS...>::eval( indices ... ) ];
  }

};




} // namespace shiva
