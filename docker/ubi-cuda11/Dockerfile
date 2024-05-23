ARG TMP_DIR=/tmp 
FROM nvidia/cuda:11.8.0-devel-ubi8 AS ubi8.8-cuda11


# install required packages 
RUN yum install -y \
    curl \
    git \
    make \
    clang \ 
    clang-tools-extra \
    python3

# Installing latest CMake version available on Lassen
ARG CMAKE_VERSION=3.23.5
RUN curl -fsSL -k  https://cmake.org/files/v${CMAKE_VERSION%.[0-9]*}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz | tar --directory=/usr/local --strip-components=1 -xzf -
