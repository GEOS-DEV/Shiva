#pragma once



template< typename PARENT_ELEMENT >
class FiniteElementSpace
{
public:
  
    /// Alias for the parent element type
    using ParentElementType = PARENT_ELEMENT;
  
    /// Alias for the floating point type
    using RealType = typename ParentElementType::RealType;
  
    /// Alias for the type that represents a coordinate
    using CoordType = typename ParentElementType::CoordType;
  
    /// Alias for the type that represents a coordinate
    using IndexType = typename ParentElementType::IndexType;
  
    /// Alias for the type that represents a coordinate
    using JacobianType = typename ParentElementType::JacobianType;
      
    /// Alias for the type that represents a coordinate
    using BasisFunctionType = typename ParentElementType::BasisFunctionType;

  

};