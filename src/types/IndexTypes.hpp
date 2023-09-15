
#pragma once

namespace shiva
{

template< typename BASE_INDEX_TYPE >
struct LinearIndex
{
  using BaseIndexType = BASE_INDEX_TYPE;
  BASE_INDEX_TYPE data = 0;
};

template< typename BASE_INDEX_TYPE, int NUM_INDICES >
struct MultiIndex
{
  using BaseIndexType = BASE_INDEX_TYPE;
  BASE_INDEX_TYPE data[NUM_INDICES] = {0};
};




template< typename BASE_INDEX_TYPE >
BASE_INDEX_TYPE linearIndex( LinearIndex< BASE_INDEX_TYPE > const & index )
{
  return index.data;
}




} // namespace shiva