FROM redhat/ubi8:8.8 AS ubi8


# install required packages 
RUN yum install -y \
    curl \
    git \
    make \
    clang \ 
    clang-tools-extra \
    python3

ARG CMAKE_VERSION=3.23.5
RUN curl -fsSL -k  https://cmake.org/files/v${CMAKE_VERSION%.[0-9]*}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz | tar --directory=/usr/local --strip-components=1 -xzf -

#  ENV CC=/opt/rh/devtoolset-12/root/usr/bin/gcc \
#      CXX=/opt/rh/devtoolset-12/root/usr/bin/g++
