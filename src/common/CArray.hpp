/**
 * @file CArray.hpp
 * @brief This file contains the implementation of the CArray class.
 */

#pragma once


#include "common/ShivaMacros.hpp"
#include <utility>



namespace shiva
{

/**
 * @namespace shiva::cArrayDetail
 * @brief The cArrayDetail namespace contains some stride calculations and 
 * linearIndex calculations for the CArray class.
 */
namespace cArrayDetail
{


/**
 * @brief This recursive function helps to calculates the strides for 
 *   dimensions passed in as a template parameter pack.
 * @tparam DIM This is the first dimension peeled off the parameter pack.
 * @tparam DIMS These are the remaining dimensions of the pack.
 * @return This returns the stride of the largest dimension in the pack.
 */
template< int DIM, int ... DIMS >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
strideHelper()
{
  if constexpr ( sizeof ... ( DIMS ) == 0 )
  {
    return DIM;
  }
  else
  {
    return DIM * strideHelper< DIMS ... >();
  }
}

/**
 * @brief This function calculates the stride of the largest dimension in the 
 *   pack.
 * @tparam DIM This is the first dimension peeled off the parameter pack.
 * @tparam DIMS These are the remaining dimensions of the pack.
 * @return This returns the stride of the largest dimension in the pack.
 */
template< int DIM, int ... DIMS >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
stride()
{
  return strideHelper< DIMS ..., 1 >();
}

/**
 * @struct linearIndexHelper
 * @brief struct to facilitate the calculation of a linear index from a pack 
 * of dimensions by peeling the dimensions pack one at a time.
 * @tparam DIM This is the first dimension peeled off the parameter pack.
 * @tparam DIMS These are the remaining dimensions of the pack.
 */
template< int DIM, int ... DIMS >
struct linearIndexHelper
{
  /**
   * @brief This recursive function calculates the linear index from a pack
   * of indices by peeling off the indices one at a time, and multiplying 
   * by the stride at that "level" of the product sum.
   * @tparam INDEX This is the first index peeled off the parameter pack.
   * @tparam INDICES These are the remaining indices of the pack.
   * @return This returns the linear index.
   */
  template< int INDEX, int ... INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
  level()
  {
    constexpr int thisStride = strideHelper< DIMS ..., 1 >();
    if constexpr ( sizeof ... ( DIMS ) == 0 )
    {
      return INDEX * thisStride;
    }
    else
    {
      return INDEX * thisStride + linearIndexHelper< DIMS ... >::template level< INDICES ... >();
    }
  }

  /**
   * @copydoc level()
   */
  template< typename INDEX_TYPE, typename ... INDICES_TYPE >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
  level( INDEX_TYPE const index, INDICES_TYPE const ... indices )
  {
    constexpr int thisStride = strideHelper< DIMS ..., 1 >();
    if constexpr ( sizeof ... ( DIMS ) == 0 )
    {
      return index * thisStride;
    }
    else
    {
      return index * thisStride + linearIndexHelper< DIMS ... >::template level( std::forward< INDICES_TYPE const >( indices )... );
    }
  }
};


/**
 * @brief function to get a specific value from a pack of indices.
 * @tparam COUNT The index of the pack to return.
 * @tparam INDEX The first index peeled off the pack.
 * @tparam INDICES The remaining indices in the pack.
 * @return The value of the INDICES pack at COUNT.
 */
template< int COUNT, int INDEX, int ... INDICES >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
get()
{
  if constexpr ( COUNT == 0 )
  {
    return INDEX;
  }
  else
  {
    return get< COUNT - 1, INDICES ... >();
  }
}

} // namespace cArrayDetail


/**
 * @struct CArray
 * @brief This struct provides a compile-time dimesion multidimensional array.
 * @tparam T This is the type of the data stored in the array.
 * @tparam DIMS These are the dimensions of the array.
 */
template< typename T, int ... DIMS >
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
  static inline constexpr int extent() { return cArrayDetail::get< INDEX, DIMS ... >(); }

  /// The size of the data in array...i.e. the product of the dimensions.
  static inline constexpr int size() { return ( DIMS * ... ); }

  /**
   * @brief This function calculates the linearIndex from a pack of indices.
   * @tparam INDICES The indices to use to calculate the linear index.
   * @return the linearIndex of the indices.
   */
  template< int ... INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
  linearIndex()
  {
    return cArrayDetail::linearIndexHelper< DIMS ... >::template level< INDICES ... >();
  }

  /**
   * @brief This function calculates the linearIndex from a pack of indices.
   * @tparam INDEX_TYPE The type of the indices.
   * @param indices The indices to use to calculate the linear index.
   * @return the linearIndex of the indices.
   */
  template< typename ... INDEX_TYPE >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
  linearIndex( INDEX_TYPE ... indices )
  {
    return cArrayDetail::linearIndexHelper< DIMS ... >::template level( std::forward< INDEX_TYPE >( indices )... );
  }


  /**
   * @brief Templated operator() to access the data in the array.
   * @tparam ...INDICES The indices that specify the data to access.
   * @return A reference to the data at the specified indices.
   */
  template< int... INDICES >
  constexpr T& operator()( )
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ linearIndex< INDICES... >() ];
  }

  /**
   * @brief Templated operator() to provide const access the data in the array.
   * @tparam ...INDICES The indices that specify the data to access.
   * @return A reference to const to the data at the specified indices.
   */
  template< int... INDICES >
  constexpr const T& operator()( ) const
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ linearIndex< INDICES... >() ];
  }

  
  /**
   * @brief parentheses operator accessor to data in the array.
   * @tparam ...INDICES The type of the indices that specify the data to access.
   * @param ...indices The pack of indices that specify the data to access.
   * @return A reference to the data at the specified indices.
   */
  template< typename ... INDICES >
  constexpr T& operator()( INDICES... indices )
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ linearIndex( indices ... ) ];
  }

  /**
   * @brief parentheses operator access to const data in the array.
   * @tparam ...INDICES The type of the indices that specify the data to access.
   * @param ...indices The pack of indices that specify the data to access.
   * @return A reference to const data at the specified indices.
   */
  template< typename ... INDICES >
  constexpr const T& operator()( INDICES... indices ) const
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ linearIndex( indices ... ) ];
  }

  /// The data in the array.
  T m_data[ size() ];
};

} // namespace shiva
