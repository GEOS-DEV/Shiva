#pragma once

#include "common/SequenceUtilities.hpp"
#include "common/ShivaMacros.hpp"
#include "common/types.hpp"
#include "functions/bases/BasisProduct.hpp"


#include <utility>

namespace shiva
{
namespace discretizations
{
namespace finiteElementMethod
{

/**
 * @class ParentElement
 * @brief Defines a class that provides static functions to calculate quantities
 * required from the parent element in a finite element method.
 * @tparam REAL_TYPE The floating point type to use
 * @tparam SHAPE The cell type/geometry
 * @tparam FUNCTIONAL_SPACE_TYPE The functional space type
 * @tparam BASIS_TYPE Pack of basis types to apply to each direction of the
 * parent element. There should be a basis defined for each direction.
 */
template< typename REAL_TYPE, typename SHAPE, typename ... BASIS_TYPE >
class ParentElement
{

public:

  /// The type used to represent the cell/geometry
  using ShapeType = SHAPE;
//  using FunctionalSpaceType = FUNCTIONAL_SPACE_TYPE;
//  using IndexType = typename ShapeType::IndexType;

  /// Alias for the floating point type
  using RealType = REAL_TYPE;

  /// Alias for the type that represents a coordinate
  using CoordType = typename ShapeType::CoordType;

  /// The type used to represent the product of basis functions
  using BASIS_PRODUCT_TYPE = functions::BasisProduct< REAL_TYPE, BASIS_TYPE... >;


  /// The number of dimensions on the ParentElement
  static inline constexpr int numDims = sizeof...(BASIS_TYPE);

  /// The number of degrees of freedom on the ParentElement in each 
  /// dimension/basis.
  static inline constexpr int numSupportPoints[numDims] = {BASIS_TYPE::numSupportPoints...};


  static_assert( numDims == ShapeType::numDims(), "numDims mismatch between cell and number of basis specified" );

  /**
   * @copydoc functions::BasisProduct::value
   */
  template< int ... BASIS_FUNCTION_INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE RealType
  value( CoordType const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );
    return ( BASIS_PRODUCT_TYPE::template value< BASIS_FUNCTION_INDICES... >( parentCoord ) );
  }


  /**
   * @copydoc functions::BasisProduct::gradient
   */
  template< int ... BASIS_FUNCTION_INDICES >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE CArray1d< RealType, numDims >
  gradient( CoordType const & parentCoord )
  {
    static_assert( sizeof...(BASIS_FUNCTION_INDICES) == numDims, "Wrong number of basis function indicies specified" );
    return ( BASIS_PRODUCT_TYPE::template gradient< BASIS_FUNCTION_INDICES... >( parentCoord ) );
  }

  template<typename VAR_TYPE >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE 
  value( CoordType const & parentCoord, VAR_TYPE const & var )
  {
    REAL_TYPE rval = {0};
    forSequence< numSupportPoints[0] >( [&] ( auto const ica ) constexpr
    {
      constexpr int a = decltype(ica)::value;
      forSequence< numSupportPoints[1] >( [&] ( auto const icb ) constexpr
      {
        constexpr int b = decltype(icb)::value;
        forSequence< numSupportPoints[2] >( [&] ( auto const icc ) constexpr
        {
          constexpr int c = decltype(icc)::value;
          rval = rval + ( value< a, b, c >( parentCoord ) * var[a][b][c] );
        });
      });
    });
    return rval;
  }


  template<typename VAR_TYPE >
  SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE CArray1d< RealType, numDims > 
  gradient( CoordType const & parentCoord, VAR_TYPE const & var )
  {
    CArray1d< RealType, numDims > rval = {0};
    forSequence< numSupportPoints[0] >( [&] ( auto const ica ) constexpr
    {
      constexpr int a = decltype(ica)::value;
      forSequence< numSupportPoints[1] >( [&] ( auto const icb ) constexpr
      {
        constexpr int b = decltype(icb)::value;
        forSequence< numSupportPoints[2] >( [&] ( auto const icc ) constexpr
        {
          constexpr int c = decltype(icc)::value;
          CArray1d< RealType, numDims > const grad = gradient< a, b, c >( parentCoord );
          rval[0] = rval[0] + grad[0] * var[a][b][c] ;
          rval[1] = rval[1] + grad[1] * var[a][b][c] ;
          rval[2] = rval[2] + grad[2] * var[a][b][c] ;
        });
      });
    });
    return rval;
  }

  // template< int DIM = 0, typename VAR_TYPE, typename ... BASIS_INDEX_PACK >
  // SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE REAL_TYPE 
  // value( CoordType const & parentCoord, VAR_TYPE const & var )
  // {
  //   if constexpr ( DIM==(numDims-1) )
  //   {
  //     return executeSequence< numDofs[DIM] >( [&]< int ... BASIS_INDICES >()
  //     {
  //       return ( value< BASIS_INDICES... >( parentCoord ) * var + ... );
  //     });
  //   }
  //   else
  //   {
  //     return 
  //     executeSequence< numDofs[DIM] >( [&]< int ... BASIS_INDICES >()
  //     {
  //       return value< DIM+1, VAR_TYPE, BASIS_INDICES >( parentCoord );
  //     });
  //   }
  // }



private:
  
//   template< typename VAR_TYPE, 
//             typename BASIS_FUNCTION, 
//             typename ... BASIS_FUNCTIONS >
//   SHIVA_STATIC_CONSTEXPR_HOSTDEVICE_FORCEINLINE VAR_TYPE
//   valueHelper( CoordType const & parentCoord,
//                VAR_TYPE const & var )
//   {
//     static constexpr int packSize = sizeof...(BASIS_FUNCTIONS);
//     if constexpr ( packSize == 0 )
//     {
//       return executeSequence([&]( auto ... indices )
//       {
// //        return value<indices...>( parentCoord ) * var[indices...][][] );
//       });
//     }
//     else
//     {
//       static constexpr int numPts = BASIS_FUNCTIONS::numPts();
//       executeSequence()
//       return ( BASIS_FUNCTIONS::value( parentCoord ) * valueHelper< VAR_TYPE, BASIS_FUNCTIONS... >( parentCoord, var ) );
//     }
//   }



};


} // namespace finiteElementMethod
} // namespace discretizations
} // namespace shiva
