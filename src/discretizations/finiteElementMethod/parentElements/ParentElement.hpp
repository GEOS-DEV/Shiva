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
  using CellType = CELL_TYPE<REAL_TYPE>;
//  using FunctionalSpaceType = FUNCTIONAL_SPACE_TYPE;
  using IndexType = typename CellType::IndexType;
  using JacobianType = typename CellType::JacobianType;
  using RealType = typename CellType::RealType;
  using CoordType = typename CellType::CoordType;
  static constexpr int Dimension = sizeof...(BASIS_TYPE);





  template< int BF_INDEX >
  RealType value( CoordType const & parentCoord ) const
  {
    return valueHelper<BF_INDEX>( parentCoord, std::make_integer_sequence<int,Dimension>{} );
  }
  
  template< int BF_INDEX, int ... DIMENSION_INDICES >
  RealType valueHelper( CoordType const & parentCoord, std::integer_sequence<int, DIMENSION_INDICES...> )
  {
    return ( BASIS_TYPE::template value<BF_INDEX>( parentCoord[DIMENSION_INDICES] ) * ... );
  }

};

} // namespace finiteElementMethod
} // namespace discretizations
} // namespace shiva
