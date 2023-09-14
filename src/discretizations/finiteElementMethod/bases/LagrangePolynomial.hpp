#pragma once


template< typename REAL_TYPE, int ORDER >
class LagrangePolynomial
{
public:
  constexpr static int order = ORDER;
  constexpr static int numSupportPoints = ORDER + 1;

private:
  REAL_TYPE m_coords[numSupportPoints];
}