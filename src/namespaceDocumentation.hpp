#error Documentation only


/**
 * @namespace shiva
 * @brief The top level namespace for Shiva. This namespace contains all of Shiva.
 */
namespace shiva
{

/**
  * @namespace shiva::geometry
  * @brief The geometry namespace contains all of the shiva/geometry classes and functions.
  */
namespace geometry
{}

/**
 * @namespace shiva::discretizations
 * @brief The discretizations namespace contains classes and functions for
 * defining a numerical discretization. For instance the finite element method,
 * the finite volume method, and the finite difference method are all 
 * methods that would be defined in this namespace.
 */
namespace discretizations
{

/**
 * @namespace shiva::discretizations::finiteElementMethod
 * @brief The finiteElementMethod namespace contains classes and functions for
 * defining a numerical discretization using the finite element method.
 */
namespace finiteElementMethod
{

/**
 * @namespace shiva::discretizations::finiteElementMethod::basis
 * @brief The basis namespace contains classes and functions for defining
 * basis functions for the finite element method.
 */
namespace basis
{
} // namespace basis
} // namespace finiteElementMethod
} // namespace discretizations

} // namespace shiva