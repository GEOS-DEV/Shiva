
#pragma once

#include "LinearIndex.hpp"
#include "MultiIndex.hpp"

#include <utility>

namespace shiva
{
namespace detail
{
template< int ... INDICES >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
strideHelper( std::integer_sequence< int, INDICES... >,
              int const * const ranges )
{
  return ( ranges[INDICES] * ... *1);
}

template< int INDEX, typename BASE_INDEX_TYPE, BASE_INDEX_TYPE... RANGES >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE int
stride( MultiIndexRange< BASE_INDEX_TYPE, RANGES... > const & index )
{
  using IndexType = MultiIndexRange< BASE_INDEX_TYPE, RANGES... >;
  return detail::strideHelper( std::make_integer_sequence< int, (IndexType::NUM_INDICES - 1) - INDEX >{},
                               &(index.ranges[0]) );
}

template< typename T, int... INDICES >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE typename T::BaseIndexType
linearIndexHelper( T const & index,
                   std::integer_sequence< int, INDICES... > )
{
  return (  ( index.data[INDICES] * stride< INDICES >( index ) ) + ... );
}



template< int DIM, typename BASE_INDEX_TYPE, BASE_INDEX_TYPE... RANGES, typename FUNC >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE void forRangeHelper( MultiIndexRange< BASE_INDEX_TYPE, RANGES... > const & start,
                     MultiIndexRange< BASE_INDEX_TYPE, RANGES... > & index,
                     FUNC && func )
{
  using IndexType = MultiIndexRange< BASE_INDEX_TYPE, RANGES... >;
  if constexpr ( DIM == (IndexType::NUM_INDICES - 1) )
  {
    int & a = index.data[DIM];
    for ( a = start.data[DIM]; a < index.range(DIM); ++a )
    {
      func( index );
    }
  }
  else
  {
    int & a = index.data[DIM];
    for ( a = start.data[DIM]; a < index.range(DIM); ++a )
    {
      forRangeHelper< DIM + 1 >( start, index, func );
    }
  }
}

} // namespace detail



template< typename BASE_INDEX_TYPE, BASE_INDEX_TYPE... RANGES >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE BASE_INDEX_TYPE
linearIndex( MultiIndexRange< BASE_INDEX_TYPE, RANGES... > const & index )
{
  using IndexType = MultiIndexRange< BASE_INDEX_TYPE, RANGES... >;
  return detail::linearIndexHelper( index, std::make_integer_sequence< int, IndexType::NUM_INDICES >{} );
}

template< typename INDEX_TYPE >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE INDEX_TYPE
linearIndex( INDEX_TYPE const & index )
{
  return index;
}



template< typename BASE_INDEX_TYPE, BASE_INDEX_TYPE... RANGES, typename FUNC >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE void forRange( MultiIndexRange< BASE_INDEX_TYPE, RANGES... > & index, FUNC && func )
{
  using IndexType = MultiIndexRange< BASE_INDEX_TYPE, RANGES... >;
  IndexType const start = index;
  detail::forRangeHelper< 0 >( start, index, std::forward< FUNC >( func ) );
}


} // namespace shiva
