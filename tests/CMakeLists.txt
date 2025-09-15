cmake_minimum_required(VERSION 3.23.1)

# Define only when BLT is loaded and tests are enabled
# (guard this in your tests/CMakeLists.txt or top-level)
# require: blt_add_executable, blt_add_test, and Shiva target

# Default output dir if you didn't set one elsewhere
if(NOT DEFINED TEST_OUTPUT_DIRECTORY)
  set(TEST_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests")
endif()

# Default output dir (optional)
if(NOT DEFINED TEST_OUTPUT_DIRECTORY)
  set(TEST_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/tests")
endif()

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -Wpedantic -pedantic-errors -Wshadow -Wfloat-equal -Wcast-align -Wcast-qual" )

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang") # For Clang or AppleClang
  set( CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fstandalone-debug" )
endif()


macro(shiva_add_gtest)
  set(options)
  set(oneValueArgs NAME)
  set(multiValueArgs SOURCES HEADERS DEPENDS_ON)
  cmake_parse_arguments(ARG "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT ARG_NAME)
    message(FATAL_ERROR "shiva_add_gtest: NAME is required")
  endif()
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "shiva_add_gtest(${ARG_NAME}): SOURCES is required")
  endif()

  # Base deps: always link Shiva (propagates include paths + interface defs)
  set(_deps Shiva gtest)

  # User-specified extras
  if(ARG_DEPENDS_ON)
    list(APPEND _deps ${ARG_DEPENDS_ON})
  endif()

  # Turn on toolchain deps as available in BLT
  if(ENABLE_CUDA)
    list(APPEND _deps cuda)
  endif()
  if(ENABLE_HIP)
    # BLT usually exposes blt::hip; fallback to 'hip' if that’s your setup
    if(TARGET blt::hip)
      list(APPEND _deps blt::hip)
    else()
      list(APPEND _deps hip)
    endif()
  endif()

  blt_add_executable(
    NAME       ${ARG_NAME}
    SOURCES    ${ARG_SOURCES}
    HEADERS    ${ARG_HEADERS}
    OUTPUT_DIR ${TEST_OUTPUT_DIRECTORY}
    DEPENDS_ON ${_deps}
  )

  blt_add_test(NAME ${ARG_NAME} COMMAND ${ARG_NAME})
endmacro()


shiva_add_gtest( NAME shiva_test_carray           SOURCES common/testCArray.cpp )
shiva_add_gtest( NAME shiva_test_index_types      SOURCES common/testIndexTypes.cpp )
shiva_add_gtest( NAME shiva_test_nested_seq_utils SOURCES common/testNestedSequenceUtilities.cpp )
shiva_add_gtest( NAME shiva_test_seq_utils        SOURCES common/testSequenceUtilities.cpp )
shiva_add_gtest( NAME shiva_test_err_handling     SOURCES common/testShivaErrorHandling.cpp )

shiva_add_gtest( NAME shiva_test_parent_element   SOURCES discretizations/finiteElementMethod/testParentElement.cpp )

shiva_add_gtest( NAME shiva_test_lagrange_basis   SOURCES functions/testLagrangeBasis.cpp )
shiva_add_gtest( NAME shiva_test_quadrature       SOURCES functions/testQuadrature.cpp )
shiva_add_gtest( NAME shiva_test_spacing          SOURCES functions/testSpacing.cpp )

shiva_add_gtest( NAME shiva_test_linear_transform SOURCES geometry/mapping/testLinearTransform.cpp )
shiva_add_gtest( NAME shiva_test_scaling          SOURCES geometry/mapping/testScaling.cpp )
shiva_add_gtest( NAME shiva_test_uniform_scaling  SOURCES geometry/mapping/testUniformScaling.cpp )
shiva_add_gtest( NAME shiva_test_interpolated     SOURCES geometry/shapes/testInterpolatedShape.cpp )
shiva_add_gtest( NAME shiva_test_ncube            SOURCES geometry/shapes/testNCube.cpp )
shiva_add_gtest( NAME shiva_test_nsimplex         SOURCES geometry/shapes/testNSimplex.cpp )

# --- CUDA variants (only built when CUDA is enabled by parent or standalone) ---
# Example: add CUDA-only tests or CUDA-flavored versions:
# shiva_add_cuda_test(shiva_test_carray_cuda common/testCArrayCuda.cu)
