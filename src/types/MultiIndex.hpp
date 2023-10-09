
#pragma once

#include "common/ShivaMacros.hpp"

#include <utility>

namespace shiva
{

template< typename BASE_INDEX_TYPE, int NUM_INDICES >
struct MultiIndex
{
  using BaseIndexType = BASE_INDEX_TYPE;
  BASE_INDEX_TYPE data[NUM_INDICES] = {0};
};


template< typename BASE_INDEX_TYPE, BASE_INDEX_TYPE... RANGES >
struct MultiIndexRange
{
  using BaseIndexType = BASE_INDEX_TYPE;
  static inline constexpr int NUM_INDICES = sizeof...(RANGES);
  static inline constexpr int ranges[NUM_INDICES] = { RANGES ...};
  BASE_INDEX_TYPE data[NUM_INDICES] = {0};
};

} // namespace shiva
