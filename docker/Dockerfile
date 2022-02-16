FROM ubuntu:18.04

# Set up timezone to avoid getting stuck at `tzdata` setup.
ENV TZ=America/Montreal
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update
RUN apt install -y tzdata

# Install necessary packages.
RUN apt-get update && apt-get install -y git curl wget software-properties-common python3.7 python3.7-dev python3-pip libgmp-dev gcc-8 g++-8 tmux vim

# Download bazel.
RUN wget https://github.com/bazelbuild/bazel/releases/download/3.7.2/bazel-3.7.2-linux-x86_64
RUN chmod +x /bazel-3.7.2-linux-x86_64
RUN mv /bazel-3.7.2-linux-x86_64 /usr/bin/bazel

# Add python alternatives.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1

# Override gcc/g++.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 60 --slave /usr/bin/g++ g++ /usr/bin/g++-8

# Install some basic things for all python versions.
RUN echo 1 | update-alternatives --config python3
RUN python3 -m pip install --upgrade pip setuptools grpcio-tools

# Add `python` so that `/usr/bin/env` finds it. This is used by `bazel`.
RUN ln -s /usr/bin/python3 /usr/bin/python

ADD . /envlogger/
WORKDIR /envlogger
