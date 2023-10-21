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

if [[ -z "${GEOSX_DIR}" ]]; then
  echo "Environment variable \"GEOSX_DIR\" is undefined."
  exit 1
fi


# The -DBLT_MPI_COMMAND_APPEND="--allow-run-as-root;--oversubscribe" option is added for OpenMPI.
#
# OpenMPI prevents from running as `root` user by default.
# And by default user is `root` in a docker container.
# Using this option therefore offers a minimal and convenient way to run the unit tests.
#
# The option `--oversubscribe` tells OpenMPI to allow more MPI ranks than the node has cores.
# This is needed because our unit test `blt_mpi_smoke` is run in parallel with _hard coded_ 4 ranks.
# While some of our ci nodes may have less cores available.
# 
# In case we have more powerful nodes, consider removing `--oversubscribe` and use `--use-hwthread-cpus` instead.
# This will tells OpenMPI to discover the number of hardware threads on the node,
# and use that as the number of slots available. (There is a distinction between threads and cores).
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
if [[ "$*" == *--test-documentation* ]]; then
  or_die ctest --output-on-failure -R "testDoxygenCheck"
  exit 0
fi

or_die make
or_die ctest --output-on-failure


exit 0