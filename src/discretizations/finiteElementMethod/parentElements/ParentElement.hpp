#pragma once 

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


  template< int ... BF_INDEX >
  constexpr static RealType value( CoordType const & parentCoord )
  {
    static_assert( sizeof...(BF_INDEX) == Dimension, "Wrong number of basis function indicies specified" );
    return valueHelper<BF_INDEX...>( parentCoord, std::make_integer_sequence<int,Dimension>{} );
  }
  
  private:
  template< int ... BF_INDEX, int ... DIMENSION_INDICES >
  constexpr static RealType valueHelper( CoordType const & parentCoord, std::integer_sequence<int, DIMENSION_INDICES...> )
  {
    return ( BASIS_TYPE::template value<BF_INDEX>( parentCoord[DIMENSION_INDICES] ) * ... );
  }

};

} // namespace finiteElementMethod
} // namespace discretizations
} // namespace shiva
