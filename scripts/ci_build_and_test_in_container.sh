#!/bin/bash
env

echo "running nproc"
nproc

echo "running free -m"
free -m

# The or_die function run the passed command line and
# exits the program in case of non zero error code
function or_die () {
    "$@"
    local status=$?
    echo status = $status
    if [[ $status != 0 ]] ; then
        echo ERROR $status command: $@
        exit $status
    fi
}

# Working in the root of the cloned repository
or_die cd $(dirname $0)/..

if [[ -z "${HOST_CONFIG}" ]]; then
  echo "Environment variable \"HOST_CONFIG\" is undefined."
  exit 1
fi

if [[ -z "${CMAKE_BUILD_TYPE}" ]]; then
  echo "Environment variable \"CMAKE_BUILD_TYPE\" is undefined."
  exit 1
fi

SHIVA_BUILD_DIR=/tmp/build
SHIVA_INSTALL_DIR=/tmp/install
or_die python3 scripts/config-build.py \
               -hc ${HOST_CONFIG} \
               -bt ${CMAKE_BUILD_TYPE} \
               -bp ${SHIVA_BUILD_DIR} \
               -ip ${SHIVA_INSTALL_DIR}

or_die cd ${SHIVA_BUILD_DIR}

# Code style check
if [[ "$*" == *--test-code-style* ]]; then
  or_die ctest --output-on-failure -R "testUncrustifyCheck"
  exit 0
fi

# Documentation check
if [[ "$*" == *--test-doxygen* ]]; then
  or_die ctest --output-on-failure -R "testDoxygenCheck"
  exit 0
fi

# code checks
if [[ "$*" == *----code_checks* ]]; then
  or_die ctest --output-on-failure -R "testCppCheck|testClangTidy"
  exit 0
fi

if [[ "$*" == *--build-exe* ]]; then
  or_die make -j $(nproc)

  if [[ "$*" != *--disable-unit-tests* ]]; then
    or_die ctest --output-on-failure -E "testUncrustifyCheck|testDoxygenCheck|testCppCheck|testClangTidy"
  fi
fi




exit 0
