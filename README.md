[![DOI](https://zenodo.org/badge/667204042.svg)](https://zenodo.org/doi/10.5281/zenodo.10425646)
[![codecov](https://codecov.io/gh/GEOS-DEV/Shiva/graph/badge.svg?token=rE4QnY9daH)](https://codecov.io/gh/GEOS-DEV/Shiva)

# Shiva
Shiva Discretization Library

Requirements: a C++20 compiler. CUDA builds require CUDA 12 or newer.

HIP consumers must explicitly require C++20 on each target that compiles HIP sources using Shiva. The Shiva interface propagates the C++ and CUDA standards; HIP standard propagation is unavailable at the project's minimum CMake version. Replace `my_target` with your consuming target:

```cmake
set_target_properties(my_target PROPERTIES
  HIP_STANDARD 20
  HIP_STANDARD_REQUIRED ON)
```

# Project Goals
The goals are:

- Provide portable device callable functionality to enable the application of discrete numerical methods such as the Finite Element Method, and Finite Difference Method, etc.
- Provide compile time interface to functionality through use of templates and `constexpr` for the special case where compile time knowledge is available.
- Specialize for common use cases to achieve maximum performance. For example a Hexahedral element with 1st order Lagrange shape functions and Gauss-Legendre quadrature.
- Support wide range of base geometries (tet, prism, pyramid, hex, polyhedral, etc), and discretization methods ( FEM, FVM, MPM, etc. )
