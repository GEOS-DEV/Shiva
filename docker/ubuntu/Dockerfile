FROM ubuntu:22.04 AS ubuntu22

# install required packages 
RUN ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        curl \
        python3 \
        clang \
        doxygen \
        cppcheck \
        clang-tidy

ARG CMAKE_VERSION=3.23.5
RUN curl -fsSL -k  https://cmake.org/files/v${CMAKE_VERSION%.[0-9]*}/cmake-${CMAKE_VERSION}-linux-x86_64.tar.gz | tar --directory=/usr/local --strip-components=1 -xzf -
