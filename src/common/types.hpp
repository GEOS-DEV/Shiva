/**
 * @file types.hpp
 * @brief Wrappers and definitions for the types used in shiva.
 */
#pragma once

#include "common/ShivaMacros.hpp"

/// @brief Macro to define whether or not to use camp.
#define SHIVA_USE_CAMP
#if defined(SHIVA_USE_CAMP)
#include <camp/camp.hpp>
#else

#if defined(SHIVA_USE_CUDA)
#include <cuda/std/tuple>
#else
#include <tuple>
#endif

#endif

namespace shiva
{

#if defined(SHIVA_USE_CAMP)

/**
 * @brief Wrapper for camp::tuple.
 * @tparam T Types of the elements of the tuple.
 */
template< typename ... T >
using tuple = camp::tuple< T ... >;

/**
 * @brief Wrapper for camp::make_tuple.
 * @tparam T Types of the elements of the tuple.
 * @param t Elements of the tuple.
 * @return A tuple with the elements passed as arguments.
 */
template< typename ... T >
SHIVA_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto make_tuple( T && ... t )
{
  return camp::make_tuple( std::forward< T >( t ) ... );
}

#else
#if defined(SHIVA_USE_CUDA)
/**
 * @brief Wrapper for cuda::std::tuple.
 * @tparam T Types of the elements of the tuple.
 */
template< typename ... T >
using tuple = cuda::std::tuple< T ... >;

/**
 * @brief Wrapper for cuda::std::make_tuple.
 * @tparam T Types of the elements of the tuple.
 * @param t Elements of the tuple.
 * @return A tuple with the elements passed as arguments.
 */
template< typename ... T >
auto make_tuple( T && ... t )
{
  return cuda::std::make_tuple( std::forward< T >( t ) ... );
}
#else
/**
 * @brief Wrapper for std::tuple.
 * @tparam T Types of the elements of the tuple.
 */
template< typename ... T >
using tuple = std::tuple< T ... >;

/**
 * @brief Wrapper for std::make_tuple.
 * @tparam T Types of the elements of the tuple.
 * @param t Elements of the tuple.
 * @return A tuple with the elements passed as arguments.
 */
template< typename ... T >
auto make_tuple( T && ... t )
{
  return std::make_tuple( std::forward< T >( t ) ... );
}
#endif
#endif

/**
 * @brief alias for std::integer_sequence<int, T...>.
 * @tparam T Types of the elements of the sequence.
 */
template< int ... T >
using int_sequence = std::integer_sequence< int, T ... >;

/**
 * @brief alias for std::make_integer_sequence<int, N>.
 * @tparam N Size of the sequence.
 */
template< int N >
using make_int_sequence = std::make_integer_sequence< int, N >;

/**
 * @brief Wrapper for a scalar type.
 * @tparam T Type of the scalar.
 */
template< typename T >
struct Scalar
{
  /// alias for T
  using type = T;

  /// contains the scalar data
  T data;
};

/**
 * @brief Wrapper for a 1d c-array .
 * @tparam T Type of the c-array.
 */
template< typename T, int N >
struct CArray1d
{
  /**
   * @brief Returns a reference to the i-th element of the array.
   * @param i The index.
   * @return A reference to the i-th element of the array.
   */
  constexpr inline T operator[] ( int const i ) const { return data[i]; }
  constexpr inline T & operator[] ( int const i )   { return data[i]; }

  /// alias for T[N]
  using type = T[N];

  /// contains the data in the array
  T data[N];
};

/**
 * @brief Wrapper for a 2d c-array .
 * @tparam T Type of the c-array.
 */
template< typename T, int N, int M >
struct CArray2d
{
  /// alias for T[N][M]
  using type = T[N][M];

  /// contains the data in the array
  T data[N][M];
};

}
