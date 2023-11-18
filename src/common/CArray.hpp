#pragma once


#include "common/ShivaMacros.hpp"
#include <utility>



namespace shiva
{

namespace cArrayDetail
{
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

template< int DIM, int ... DIMS >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
stride()
{
  return strideHelper< DIMS ..., 1 >();
}

template< int DIM, int ... DIMS >
struct linearIndexHelper
{
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



} // namespace cArrayDetail

template< typename T, int ... DIMS >
struct CArray
{
  static inline constexpr int numDims = sizeof ... ( DIMS );
  static inline constexpr int dims[ numDims ] = { DIMS ... };
  static inline constexpr int length = ( DIMS * ... );

  template< int ... INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
  linearIndex()
  {
    return cArrayDetail::linearIndexHelper< DIMS ... >::template level< INDICES ... >();
  }

  template< typename ... INDEX_TYPE >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
  linearIndex( INDEX_TYPE ... indices )
  {
    return cArrayDetail::linearIndexHelper< DIMS ... >::template level( std::forward< INDEX_TYPE >( indices )... );
  }

  template< int... INDICES >
  constexpr T& operator()( )
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ linearIndex< INDICES... >() ];
  }

  template< int... INDICES >
  constexpr const T& operator()( ) const
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ linearIndex< INDICES... >() ];
  }

  template< typename ... INDICES >
  constexpr T& operator()( INDICES... indices )
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ linearIndex( indices ... ) ];
  }

  template< typename ... INDICES >
  constexpr const T& operator()( INDICES... indices ) const
  {
    static_assert( sizeof...(INDICES) == sizeof...(DIMS), "Incorrect number of indices" );
    return m_data[ linearIndex( indices ... ) ];
  }

  T m_data[ length ];
};

} // namespace shiva
