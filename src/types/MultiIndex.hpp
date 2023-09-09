
#pragma once

namespace shiva
{

template< int NUM_INDICES, typename BASE_INDEX_TYPE >
struct MultiIndex
{
  BASE_INDEX_TYPE data[NUM_INDICES] = {0};
};

}