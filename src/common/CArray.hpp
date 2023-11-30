/**
 * @file CArray.hpp
 * @brief This file contains the implementation of the CArray class.
 */

#pragma once


#include "common/ShivaMacros.hpp"

#include "CArrayHelper.hpp"

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
  static inline constexpr int extent() { return CArrayHelper::get< INDEX, DIMS ... >(); }

  /// The size of the data in array...i.e. the product of the dimensions.
  static inline constexpr int size() { return ( DIMS * ... ); }

  /**
   * @brief Constructor for wrapping a pointer to data
   * @tparam U Type to used to check if the data is a pointer for SFINAE.
   * @tparam ENABLE SFINAE parameter.
   * @param buffer data that the view will point to.
   */
  template< typename U = DATA_BUFFER,
            std::enable_if_t< std::is_pointer_v< U >, int > ENABLE = 0 >
  constexpr explicit CArray( T const (&buffer)[ CArrayHelper::size< DIMS... >() ] ):
    m_data( buffer )
  {}

  /**
   * @copydoc CArray
   */
  template< typename U = DATA_BUFFER,
            std::enable_if_t< std::is_pointer_v< U >, int > = 0 >
  constexpr explicit CArray( std::remove_const_t< T >(&buffer)[ CArrayHelper::size< DIMS... >() ] ):
    m_data( buffer )
  {}

  /**
   * @brief Constructor for list initialization.
   * @tparam ...U the type of the arguments.
   * @param ...args the data to initialize the array with.
   */
  template< typename ... U >
  constexpr CArray( U ... args ): m_data{ args ... }
  {}

  /**
   * @brief accessor for m_data
   * @return reference to m_data
   */
  DATA_BUFFER & data() { return m_data; }

  /**
   * @brief const accessor for m_data
   * @return reference to const m_data
   */
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
  /// The data in the array.
  DATA_BUFFER m_data;
};

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
