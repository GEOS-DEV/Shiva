/**
 * @file SequenceUtilities.hpp
 * @brief This file contains utilities for the manipulation of parameter packs.
 */

#pragma once

#include "common/ShivaMacros.hpp"
#include <type_traits>
#include <utility>

namespace shiva
{

template < typename FUNC, typename T, T... Indices >
constexpr auto unpack(std::integer_sequence<T, Indices...>, FUNC && func)
{
  return func(Indices...);
}

template <typename FUNC, typename SEQ>
constexpr auto unpack(FUNC &&func)
{
  return unpack(SEQ{}, std::forward<FUNC>(func));
}


// Helper class to capture the int pack from std::integer_sequence
// to assist in `using` statements
template < template <int...> class Template, typename T>
struct SequenceAlias;

template < template <int...> class Template, int... Seq >
struct SequenceAlias< Template, std::integer_sequence<int, Seq...> >
{
  using type = Template<Seq...>;
};

template< typename <int...> class Template, int... Seq >
using SequenceAlias_t = typename SequenceAlias< Template, Seq... >;

// Base template for concat
template<typename T, typename Seq1, typename Seq2>
struct Concat;

// Specialization for combining two integer_sequences
template< typename T, T... Ints1, T... Ints2 >
struct Concat<T, std::integer_sequence<T, Ints1...>, std::integer_sequence<T, Ints2...>>
{
    using type = std::integer_sequence<T, Ints1..., Ints2...>;
};

// Helper template alias for ease of use
template< typename Seq1, typename Seq2 >
using Concat_t = typename Concat<typename Seq1::value_type, Seq1, Seq2>::type;

template < typename Seq, std::size_t Size1, std::size_t Size2 >
struct Split;

template< typename Seq, std::size_t Size1, std::size_t Size2 >
struct Split
{
  static_assert( Seq::size() == Size1 + Size2, "Invalid split sizes for the sequence");

  template<std::size_t... Indices>
  static constexpr auto extractFirst(std::index_sequence<Indices...>) -> std::integer_sequence<std::size_t, std::get<Indices>(std::make_tuple(Seq::value...))...>;

  template<std::size_t... Indices>
  static constexpr auto extractSecond( std::index_sequence<Indices...> ) -> std::integer_sequence<std::size_t, std::get<Size1 + Indices>(std::make_tuple(Seq::value...))...>;

  using first = decltype(extractFirst(std::make_index_sequence<FirstSize>{}));
  using second = decltype(extractSecond(std::make_index_sequence<SecondSize>{}));
};

template < typename T, std::size_t Size1, std::size_t Size2, T... Seq >
using SplitPack = typename Split< std::integer_sequence< T, Seq... >, Size1, Size2 >;


template<int N>
struct SingleValSequence
{
  using type = std::integer_sequence<int, N>;
};

// Helper template alias for ease of use
template<int N>
using SingleValSequence_t = typename SingleValSequence<N>::type;




template< int I, typename T, typename ... Ts>
struct PackPeeler
{
  /// The type of the first value in the pack.
  using type = typename PackPeeler< I - 1, Ts... >::type;
};

template< typename T, typename ... Ts>
struct PackPeeler<0,T,Ts...>
{
  /// The type of the first value in the pack.
  using type = T;
};



template< int I, int FIRST, int ... REST >
struct IntPackPeeler
{
  /// The type of the first value in the pack.
  static constexpr int value() { return IntPackPeeler< I - 1, REST... >::value(); }
};

template< int FIRST, int ... REST>
struct IntPackPeeler<0,FIRST,REST...>
{
  /// The type of the first value in the pack.
  static constexpr int value() { return FIRST; }
};


template <typename T, int ...PACK> struct ParameterPacker
{
    template <template <typename, int...> typename TT, int ... OTHER_DIMS >
    using type = TT<T, PACK..., OTHER_DIMS...>;
};





/**
 * @namespace shiva::sequenceUtilities
 * @brief The sequenceUtilitiesImpl namespace contains implementation details
 * for SequenceUtilities.
 */
namespace sequenceUtilitiesImpl
{

/**
 * @brief This generic declaration is used to allow for specialization that
 * deduces a parameter pack from an integer_sequence.
 * @tparam T This is a generic parameter pack
 */
template< typename ... T >
struct SequenceExpansion
{};

/**
 * @brief This specialization deduces a parameter pack from an integer_sequence
 * @tparam DIMENSION_INDICES The parameter pack of "int"s deduced from the
 * integer_sequence.
 */
template< int ... DIMENSION_INDICES >
struct SequenceExpansion< std::integer_sequence< int, DIMENSION_INDICES... > >
{

  /**
   * @brief This function uses the DIMENSION_INDICES parameter pack expanded in
   * the struct to execute a function that takes the parameter pack as an
   * argument.
   * @tparam FUNC This is the type of the function to call.
   * @tparam ARGS A Parameter pack that contains the arguments types to the
   * function that were passed in to the execute function.
   * @param func This is the function of type FUNC to call.
   * @param args These are the arguments to pass to func that were passed in to
   * the execute function.
   * @return The return value of func().
   */
  template< typename FUNC, typename ... ARGS >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto execute( FUNC && func, ARGS && ... args )
  {
    // This checks to see if the function is invocable with the parameter pack
    // provided as a template parameter. In other words, if the FUNC is a
    // function that takes the parameter pack as a function argument.
    if constexpr ( std::is_invocable_v< FUNC, std::integral_constant< int, DIMENSION_INDICES >..., ARGS ... > )
    {
      // calling func with the parameter packs ( args..., DIMENSION_INDICES...)
      // as arguments.
      return func( std::forward< ARGS >( args )...,
                   std::integral_constant< int, DIMENSION_INDICES >{} ... );
    }
    // Otherwise, the function is templated on the parameter pack, so we
    // pass the parameter pack as a template argument.
    else
    {
      // call func with DIMENSION_INDICES... as template arguments, and args...
      // as function arguments.
      return func.template operator()< DIMENSION_INDICES... >(std::forward< ARGS >( args )...);
    }
  }

  /**
   * @brief This function uses the DIMENSION_INDICES parameter pack expanded in
   * the struct to execute a "compile time for loop" that takes a calls a
   * function that takes a single element of the parameter pack as an argument.
   * @tparam FUNC This is the type of the function to call.
   * @param func This is the function of type FUNC to call.
   * @return The return value of func().
   */
  template< typename FUNC >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto staticFor( FUNC && func )
  {
    // This checks to see if the function is invocable with an element of the
    // parameter pack provided (i.e. int) as a function parameter. In other
    // words, if the FUNC expects the DIMENSION_INDICES in the argument list,
    // then make the appropriate call.
    if constexpr ( std::is_invocable_v< FUNC, std::integral_constant< int, 0 > > )
    {
      // fold that expands calls to func with each element of the parameter
      // pack DIMENSION_INDICES as an argument. The fold calls it for each
      // value of DIMENSION_INDICES.
      return (func( std::integral_constant< int, DIMENSION_INDICES >{} ), ...);
    }
    // Otherwise, the function is templated on an integer, so we
    // pass the element of the integer parameter pack as a template argument.
    else
    {
      // fold that expands calls to func with each element of the parameter
      // pack DIMENSION_INDICES as an template parameter. The fold calls it for
      // each value of DIMENSION_INDICES.
      return (func.template operator()< DIMENSION_INDICES >(), ...);
    }
  }
};

} // namespace sequenceUtilitiesImpl


/**
 * @brief This function creates an integer_sequence<0,1,2,...,END-1> and calls
 * sequenceUtilitiesImpl::SequenceExpansion::execute to deduce the int... and
 * call func of type FUNC, passing back the int... as either a template parameter
 * or a function argument.
 * @tparam END This is the number of elements in the
 * integer_sequence<0,1,2,...,END-1>
 * @tparam FUNC This is the type of the function to call.
 * @tparam ARGS A Parameter pack that contains the arguments types to the
 * function that were passed in to the execute function.
 * @param func This is the function of type FUNC to call.
 * @param args These are the arguments to pass to func that were passed in to
 * the execute function.
 * @return The return value of func().
 */
template< int END, typename FUNC, typename ... ARGS >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto executeSequence( FUNC && func,
                                                                    ARGS && ... args )
{
  return
    sequenceUtilitiesImpl::SequenceExpansion< std::make_integer_sequence< int, END > >::
    template execute( std::forward< FUNC >( func ),
                      std::forward< ARGS >( args )... );
}

/**
 * @brief This function creates an integer_sequence<0,1,2,...,END-1> and calls
 * sequenceUtilitiesImpl::SequenceExpansion::staticFor to deduce the int... and
 * call func of type FUNC, passing back an int in the (int...) as either a
 * template parameter or a function argument.
 * @tparam END This is the number of elements in the
 * integer_sequence<0,1,2,...,END-1>
 * @tparam FUNC This is the type of the function to call.
 * @param func This is the function of type FUNC to call.
 * @return The return value of func().
 */
template< int END, typename FUNC >
SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE auto forSequence( FUNC && func )
{
  return
    sequenceUtilitiesImpl::SequenceExpansion< std::make_integer_sequence< int, END > >::
    template staticFor( std::forward< FUNC >( func ) );
}




} // namespace shiva
