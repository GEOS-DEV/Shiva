#pragma once 

#include "types/types.hpp"
#include <utility>

namespace shiva
{
namespace discretizations
{
namespace finiteElementMethod
{
template< typename REAL_TYPE, template< typename > typename CELL_TYPE, typename ... BASIS_TYPE  >
class ParentElement
{
  template< typename ... ARGS >
  struct pack {};

  public:
  using CellType = CELL_TYPE<REAL_TYPE>;
//  using FunctionalSpaceType = FUNCTIONAL_SPACE_TYPE;
//  using IndexType = typename CellType::IndexType;
  using JacobianType = typename CellType::JacobianType;
  using RealType = REAL_TYPE;
  using CoordType = typename CellType::CoordType;

  using BasisType = pack<BASIS_TYPE...>;

  static constexpr int Dimension = sizeof...(BASIS_TYPE);

  
  static_assert( Dimension == CellType::Dimension(), "Dimension mismatch between cell and number of basis specified" );

  template< int ... BASIS_FUNCTION_INDICES >
  constexpr static RealType value( CoordType const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == Dimension, "Wrong number of basis function indicies specified" );
    return SequenceUnpacker< std::make_integer_sequence< int, Dimension> >::template value<BASIS_FUNCTION_INDICES...>( parentCoord );
  }

  template< int ... BASIS_FUNCTION_INDICES >
  constexpr static CArray1d<RealType,Dimension> gradient( CoordType const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == Dimension, "Wrong number of basis function indicies specified" );
    return SequenceUnpacker< std::make_integer_sequence< int, Dimension> >::template gradient<BASIS_FUNCTION_INDICES...>( parentCoord );
  }


private:
  template< typename... INTEGER_SEQUENCES >
  struct SequenceUnpacker
  {};

  template< int ... DIMENSION_INDICES >
  struct SequenceUnpacker< std::integer_sequence<int, DIMENSION_INDICES...> >
  {
    template< int ... BASIS_FUNCTION_INDICES >
    constexpr static RealType value( CoordType const & parentCoord )
    {
      return ( BASIS_TYPE::template value<BASIS_FUNCTION_INDICES>( parentCoord[DIMENSION_INDICES] ) * ... );
    }


    template< int ... BASIS_FUNCTION_INDICES >
    constexpr static CArray1d<RealType,Dimension> gradient( CoordType const & parentCoord )
    {
        return { gradientComponent<DIMENSION_INDICES,BASIS_FUNCTION_INDICES...>( parentCoord )... };
    }

    template< int GRADIENT_COMPONENT, int ... BASIS_FUNCTION_INDICES >
    constexpr static RealType gradientComponent( CoordType const & parentCoord )
    {
//        printf( "gradientComponent<%d>() \n", GRADIENT_COMPONENT );
        return ( gradientComponentHelper<BASIS_TYPE, GRADIENT_COMPONENT, BASIS_FUNCTION_INDICES, DIMENSION_INDICES >( parentCoord ) * ... );
    }

    template< typename BASIS_FUNCTION, int GRADIENT_COMPONENT, int BASIS_FUNCTION_INDEX, int COMPONENT_INDEX >
    constexpr static RealType gradientComponentHelper( CoordType const & parentCoord )
    {
//        printf( "%d, %d \n", GRADIENT_COMPONENT, COMPONENT_INDEX );
        if constexpr ( GRADIENT_COMPONENT == COMPONENT_INDEX )
        {
            return BASIS_FUNCTION::template gradient<BASIS_FUNCTION_INDEX>( parentCoord[COMPONENT_INDEX] );
        }
        else
        {
            return ( BASIS_FUNCTION::template value<BASIS_FUNCTION_INDEX>( parentCoord[COMPONENT_INDEX] ) );
        }
    }
  };

  


};


} // namespace finiteElementMethod
} // namespace discretizations
} // namespace shiva
