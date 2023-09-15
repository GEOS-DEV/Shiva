
#pragma once

namespace shiva
{

template< typename BASE_INDEX_TYPE >
struct Index
{
  using BaseIndexType = BASE_INDEX_TYPE;
  BASE_INDEX_TYPE data = 0;
};

template< int NUM_INDICES, typename BASE_INDEX_TYPE >
struct MultiIndex
{
  using BaseIndexType = BASE_INDEX_TYPE;
  BASE_INDEX_TYPE data[NUM_INDICES] = {0};
};
} // namespace shiva