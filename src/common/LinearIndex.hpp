
#pragma once

#include <utility>

namespace shiva
{

#if 1
/**
 * @brief LinearIndex is a templated linear index type consisting of a single
 * index of type T.
 * @tparam T The type of the index.
 */
template< typename T >
using LinearIndex = T;
#else
template< typename BASE_INDEX_TYPE >
struct LinearIndex
{

  LinearIndex( LinearIndex const & rhs )
  {
    data = rhs.data;
  }

  LinearIndex( BASE_INDEX_TYPE const & rhs )
  {
    data = rhs;
  }

  LinearIndex( LinearIndex && rhs )
  {
    data = std::move( rhs.data );
  }

  LinearIndex( BASE_INDEX_TYPE && rhs )
  {
    data = std::move( rhs );
  }

  LinearIndex & operator=( LinearIndex const & rhs )
  {
    data = rhs.data;
    return *this;
  }

  LinearIndex & operator=( BASE_INDEX_TYPE const & rhs )
  {
    data = rhs;
    return *this;
  }

  LinearIndex & operator++()
  {
    ++data;
    return *this;
  }

  LinearIndex & operator--()
  {
    --data;
    return *this;
  }

  bool operator==( LinearIndex const & rhs ) const
  {
    return data == rhs.data;
  }

  bool operator==( BASE_INDEX_TYPE const & rhs ) const
  {
    return data == rhs;
  }

  bool operator<( LinearIndex const & rhs ) const
  {
    return data < rhs.data;
  }

  bool operator<( BASE_INDEX_TYPE const & rhs ) const
  {
    return data < rhs;
  }

  using BaseIndexType = BASE_INDEX_TYPE;
  BASE_INDEX_TYPE data = 0;
};
#endif

} // namespace shiva
