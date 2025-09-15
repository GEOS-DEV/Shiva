cmake_minimum_required(VERSION 3.23.1)
include(CTest)

# One function for both CPU & CUDA cases.
function(shiva_add_gtest name)
  set(options)
  set(oneValueArgs)
  set(multiValueArgs SOURCES CUDA_SOURCES LIBS)
  cmake_parse_arguments(SAG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  add_executable(${name} ${SAG_SOURCES})

  # If CUDA toolchain is active, tack on CUDA sources & props.
  if (CMAKE_CUDA_COMPILER AND SAG_CUDA_SOURCES)
    target_sources(${name} PRIVATE ${SAG_CUDA_SOURCES})
    set_target_properties(${name} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    # Your headers probably branch on this; usually comes via Shiva INTERFACE anyway.
    target_compile_definitions(${name} PRIVATE SHIVA_USE_CUDA=1)
  endif()

  target_link_libraries(${name} PRIVATE
    Shiva              # pulls include/shiva + build/include/shiva
    GTest::gtest_main
    ${SAG_LIBS}
  )
  target_compile_features(${name} PRIVATE cxx_std_17)

  add_test(NAME ${name} COMMAND ${name})
endfunction()


# --- Common tests ---
shiva_add_gtest(shiva_test_carray           common/testCArray.cpp)
shiva_add_gtest(shiva_test_index_types      common/testIndexTypes.cpp)
shiva_add_gtest(shiva_test_nested_seq_utils common/testNestedSequenceUtilities.cpp)
shiva_add_gtest(shiva_test_seq_utils        common/testSequenceUtilities.cpp)
shiva_add_gtest(shiva_test_err_handling     common/testShivaErrorHandling.cpp)

shiva_add_gtest(shiva_test_parent_element   discretizations/testParentElement.cpp)

shiva_add_gtest(shiva_test_lagrange_basis   functions/testLagrangeBasis.cpp)
shiva_add_gtest(shiva_test_quadrature       functions/testQuadrature.cpp)
shiva_add_gtest(shiva_test_spacing          functions/testSpacing.cpp)

shiva_add_gtest(shiva_test_linear_transform geometry/mapping/testLinearTransform.cpp)
shiva_add_gtest(shiva_test_scaling          geometry/mapping/testScaling.cpp)
shiva_add_gtest(shiva_test_uniform_scaling  geometry/mapping/testUniformScaling.cpp)
shiva_add_gtest(shiva_test_interpolated     geometry/shapes/testInterpolatedShape.cpp)
shiva_add_gtest(shiva_test_ncube            geometry/shapes/testNCube.cpp)
shiva_add_gtest(shiva_test_nsimplex         geometry/shapes/testNSimplex.cpp)

# --- CUDA variants (only built when CUDA is enabled by parent or standalone) ---
# Example: add CUDA-only tests or CUDA-flavored versions:
# shiva_add_cuda_test(shiva_test_carray_cuda common/testCArrayCuda.cu)
