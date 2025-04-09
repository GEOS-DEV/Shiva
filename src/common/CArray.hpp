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
 * @file CArray.hpp
 * @brief This file contains the implementation of the CArray class.
 */

#pragma once


#include "common/ShivaMacros.hpp"

#include "CArrayHelper.hpp"

#include <utility>
#include <cmath>


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
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int rank() { return sizeof ... ( DIMS ); }

  /// The dimensions of the array.
  template< int INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int extent() { return CArrayHelper::get< INDEX, DIMS ... >(); }

  /// The size of the data in array...i.e. the product of the dimensions.
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int size() { return ( DIMS * ... ); }

  /**
   * @brief Constructor for wrapping a pointer to data
   * @tparam U Type to used to check if the data is a pointer for SFINAE.
   * @tparam ENABLE SFINAE parameter.
   * @param buffer data that the view will point to.
   */
  template< typename U = DATA_BUFFER,
            std::enable_if_t< std::is_pointer_v< U >, int > ENABLE = 0 >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE explicit CArray( T const (&buffer)[ CArrayHelper::size< DIMS... >() ] ):
    m_data( buffer )
  {}

  /**
   * @copydoc CArray
   */
  template< typename U = DATA_BUFFER,
            std::enable_if_t< std::is_pointer_v< U >, int > = 0 >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE explicit CArray( std::remove_const_t< T >(&buffer)[ CArrayHelper::size< DIMS... >() ] ):
    m_data( buffer )
  {}

  /**
   * @brief Constructor for list initialization.
   * @tparam ...U the type of the arguments.
   * @param ...args the data to initialize the array with.
   */
  template< typename ... U >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE CArray( U ... args ): m_data{ args ... }
  {}

  /**
   * @brief accessor for m_data
   * @return reference to m_data
   */
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE
  DATA_BUFFER & data() { return m_data; }

  /**
   * @brief const accessor for m_data
   * @return reference to const m_data
   */
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE
  DATA_BUFFER const & data() const { return m_data; }

  /**
   * @brief Templated operator() to access the data in the array.
   * @tparam ...INDICES The indices that specify the data to access.
   * @return A reference to the data at the specified indices.
   */
  template< int... INDICES >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE
  T& operator()( )
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ CArrayHelper::linearIndexHelper< DIMS... >::template eval< INDICES... >() ];
  }

  /**
   * @brief const operator() to access the data in the array.
   * @tparam ...INDICES The indices that specify the data to access.
   * @return A reference to the data at the specified indices.
   */
  template< int... INDICES >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE
  std::add_const_t< T >& operator()( ) const
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ CArrayHelper::linearIndexHelper< DIMS... >::template eval< INDICES... >() ];
  }

  /**
   * @brief parentheses operator accessor to data in the array.
   * @tparam ...INDICES The type of the indices that specify the data to access.
   * @param ...indices The pack of indices that specify the data to access.
   * @return A reference to the data at the specified indices.
   */
  template< typename ... INDICES >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE
  T& operator()( INDICES... indices )
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ CArrayHelper::linearIndexHelper< DIMS... >::eval( indices ... ) ];
  }

  /**
   * @brief const parentheses operator accessor to data in the array.
   * @tparam ...INDICES The type of the indices that specify the data to access.
   * @param ...indices The pack of indices that specify the data to access.
   * @return A reference to the data at the specified indices.
   */
  template< typename ... INDICES >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE
  std::add_const_t< T >& operator()( INDICES... indices ) const
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ CArrayHelper::linearIndexHelper< DIMS... >::eval( indices ... ) ];
  }

private:
  /**
   * @struct SubArrayHelper
   * @brief This struct is used to help with the creation of subarrays.
   */
  struct SubArrayHelper
  {
    /// This alias is used to create a const subarray type.

    template< int ... SUB_DIMS >
    using const_type = CArray< T const, T const * const, SUB_DIMS... >;

    /// This alias is used to create a subarray type.
    template< int ... SUB_DIMS >
    using type = CArray< T, T * const, SUB_DIMS... >;
  };

  /**
   * @brief Helper function to enable operator[].
   * @tparam CARRAY_TYPE The type of the array to create. Either const_type or type.
   * @param index The index to access.
   * @return A reference to the data at the specified index or the subarray.
   */
  template< template< int ... > typename CARRAY_TYPE >
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE
  decltype(auto) squareBracketOperatorHelper( index_type index ) const;

public:
  /**
   * @brief operator[] to access the data in the array or to slice a multidimensional array.
   * @param index The index to access.
   * @return A reference to the data at the specified index or the subarray.
   */
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE
  decltype(auto) operator[]( index_type index ) const
  {
    return squareBracketOperatorHelper< SubArrayHelper::template const_type >( index );
  }

  /**
   * @brief operator[] to access the data in the array or to slice a mutlidimensional array.
   * @param index The index to access.
   * @return A reference to the data at the specified index or the subarray.
   */
  SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE
  decltype(auto) operator[]( index_type index )
  {
    if constexpr ( std::is_reference_v< decltype(squareBracketOperatorHelper< SubArrayHelper::template type >( index )) > )
    {
      return const_cast< T & >(squareBracketOperatorHelper< SubArrayHelper::template type >( index ));
    }
    else
    {
      return squareBracketOperatorHelper< SubArrayHelper::template type >( index );
    }
  }


private:
  /// The data in the array.
  DATA_BUFFER m_data;
};

/// @copydoc CArray::squareBracketOperatorHelper
template< typename T, typename DATA_BUFFER, int ... DIMS >
template< template< int ... > typename CARRAY_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE
decltype(auto)
CArray< T, DATA_BUFFER, DIMS ... >::squareBracketOperatorHelper( index_type index ) const
{
  static_assert( sizeof...(DIMS) >= 1, "operator[] is only valid for sizeof...(DIMS) >= 1" );

#if defined( SHIVA_USE_BOUNDS_CHECK )
  constexpr int DIM = CArrayHelper::IntPeeler< DIMS... >::first;
  SHIVA_ASSERT_MSG( index >= 0 && index < DIM,
                    "Index out of bounds: 0 < index(%jd) < dim(%jd)",
                    static_cast< intmax_t >( index ),
                    static_cast< intmax_t >( DIM ) );
#endif

  if constexpr ( sizeof...(DIMS) > 1 )
  {
    using SubArrayDims = typename CArrayHelper::IntPeeler< DIMS... >::rest;
    using SubArrayType = typename CArrayHelper::ApplyDims< SubArrayDims, CARRAY_TYPE >::type;
    return SubArrayType( const_cast< T * >( &m_data[ index * CArrayHelper::stride< DIMS..., 1 >() ] ) );
  }
  else
  {
    return operator()( index );
  }
}

/**
 * @brief Alias for a scalar.
 * @tparam T Type held in the scalar
 */
template< typename T >
using Scalar = CArray< T, T[1], 1 >;

/**
 * @brief Alias for a N-d array.
 * @tparam T Type held in the array.
 * @tparam DIMS The dimensions of the array.
 */
template< typename T, int ... DIMS >
using CArrayNd = CArray< T, T[CArrayHelper::size< DIMS ... >()], DIMS ... >;


/**
 * @brief Alias for a scalar view.
 * @tparam T Type held in the scalar
 */
template< typename T >
using ScalarView = CArray< T, T, 1 >;

/**
 * @brief Alias for a N-d array view.
 * @tparam T Type held in the array.
 * @tparam DIMS The dimensions of the array.
 */
template< typename T, int ... DIMS >
using CArrayViewNd = CArray< T, T * const, DIMS ... >;



} // namespace shiva
