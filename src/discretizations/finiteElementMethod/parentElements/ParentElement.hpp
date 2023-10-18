#pragma once

#include "common/SequenceUtilities.hpp"
#include "common/ShivaMacros.hpp"
#include "common/types.hpp"


#include <utility>

namespace shiva
{
namespace discretizations
{
namespace finiteElementMethod
{
template< typename REAL_TYPE, template< typename > typename CELL_TYPE, typename ... BASIS_TYPE >
class ParentElement
{
  template< typename ... ARGS >
  struct pack {};

public:
  using CellType = CELL_TYPE< REAL_TYPE >;
//  using FunctionalSpaceType = FUNCTIONAL_SPACE_TYPE;
//  using IndexType = typename CellType::IndexType;
  using JacobianType = typename CellType::JacobianType;
  using RealType = REAL_TYPE;
  using CoordType = typename CellType::CoordType;

  using BasisType = pack< BASIS_TYPE ... >;

  static inline constexpr int Dimension = sizeof...(BASIS_TYPE);


  static_assert( Dimension == CellType::Dimension(), "Dimension mismatch between cell and number of basis specified" );

  template< int ... BASIS_FUNCTION_INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE RealType 
  value( CoordType const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == Dimension, "Wrong number of basis function indicies specified" );

    return
#if __cplusplus >= 202002L
    executeSequence< Dimension >( [&]< int ... DIMENSION_INDICES > () constexpr
    {
      return ( BASIS_TYPE::template value< BASIS_FUNCTION_INDICES >( parentCoord[DIMENSION_INDICES] ) * ... );
    } );
#else
    executeSequence< Dimension >( [&] ( auto ... DIMENSION_INDICES ) constexpr
    {
      return ( BASIS_TYPE::template value< BASIS_FUNCTION_INDICES >( parentCoord[decltype(DIMENSION_INDICES)::value] ) * ... );
    } );

#endif
  }

  template< int ... BASIS_FUNCTION_INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE CArray1d< RealType, Dimension > 
  gradient( CoordType const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == Dimension, "Wrong number of basis function indicies specified" );

#if __cplusplus >= 202002L
    return executeSequence< Dimension >( [&]< int ... i >() constexpr -> CArray1d< RealType, Dimension >
    {
      auto gradientComponent = [&] ( auto const iGrad, auto const  ... BASIS_FUNCTION_INDEX ) constexpr
      {
        return ( gradientComponentHelper< BASIS_TYPE,
                                          decltype(iGrad)::value,
                                          BASIS_FUNCTION_INDICES,
                                          BASIS_FUNCTION_INDEX >( parentCoord ) * ... );
      };

      return { (executeSequence< Dimension >( gradientComponent, std::integral_constant< int, i >{} ) )...  };
    } );
#else
    return executeSequence< Dimension >( [&] ( auto ... a ) constexpr -> CArray1d< RealType, Dimension >
    {
      auto gradientComponent = [&] ( auto GRADIENT_COMPONENT, auto ... BASIS_FUNCTION_INDEX ) constexpr
      {
        return ( gradientComponentHelper< BASIS_TYPE, GRADIENT_COMPONENT, BASIS_FUNCTION_INDICES, BASIS_FUNCTION_INDEX >( parentCoord ) * ... );
      };

      return { (executeSequence< Dimension >( gradientComponent, a ) )...  };
    } );
#endif
  }


private:
  template< typename BASIS_FUNCTION, int GRADIENT_COMPONENT, int BASIS_FUNCTION_INDEX, int COMPONENT_INDEX >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE RealType 
  gradientComponentHelper( CoordType const & parentCoord )
  {
    if constexpr ( GRADIENT_COMPONENT == COMPONENT_INDEX )
    {
      return BASIS_FUNCTION::template gradient< BASIS_FUNCTION_INDEX >( parentCoord[COMPONENT_INDEX] );
    }
    else
    {
      return ( BASIS_FUNCTION::template value< BASIS_FUNCTION_INDEX >( parentCoord[COMPONENT_INDEX] ) );
    }
  }



};


} // namespace finiteElementMethod
} // namespace discretizations
} // namespace shiva
