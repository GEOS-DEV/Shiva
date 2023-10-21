#!/bin/bash
env

BUILD_DIR=${GITHUB_WORKSPACE}
BUILD_DIR_MOUNT_POINT=/tmp/Shiva


# We need to keep track of the building container (hence the `CONTAINER_NAME`)
# so we can extract the data from it later (if needed). Another solution would have been to use a mount point,
# but that would not have solved the problem for the TPLs (we would require extra action to copy them to the mount point).
CONTAINER_NAME=shiva_build
# Now we can build shiva.
docker run \
  --name=${CONTAINER_NAME} \
  --volume=${BUILD_DIR}:${BUILD_DIR_MOUNT_POINT} \
  --cap-add=ALL \
  -e HOST_CONFIG=${HOST_CONFIG:hostconfigs/environment.cmake} \
  -e CMAKE_C_COMPILER=${CMAKE_C_COMPILER} \
  -e CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \
  -e CMAKE_BUILD_TYPE \
  ${DOCKER_REPOSITORY} \
  ${BUILD_DIR_MOUNT_POINT}/scripts/ci_build_and_test_in_container.sh ${BUILD_AND_TEST_ARGS};