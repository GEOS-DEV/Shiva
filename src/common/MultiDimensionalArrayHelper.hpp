/**
 * @file MultiDimensionalArrayHelper.hpp
 * @brief This file contains the helper functions for the MultiDimensionalArray and MultiDimensionalSpan.
 */

#pragma once


#include "common/ShivaMacros.hpp"
#include <utility>



namespace shiva
{

/**
 * @namespace shiva::MultiDimensionalArrayHelper
 * @brief The MultiDimensionalArrayHelper namespace contains some stride calculations and 
 * linearIndex calculations for the MultiDimensionalBase class.
 */
namespace MultiDimensionalArrayHelper
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


  template< int ... INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
  eval()
  {
    return level< INDICES ... >();
  }


  template< typename ... INDEX_TYPE >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
  eval( INDEX_TYPE ... indices )
  {
    return level( std::forward< INDEX_TYPE >( indices )... );
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

template< int ... DIMS >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
size()
{
  return ( DIMS * ... );
}

} // namespace MultiDimensionalArrayHelper
} // namespace shiva
