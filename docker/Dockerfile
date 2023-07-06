FROM ubuntu:20.04

# ARG variable for Python 3 minor version.
# This can be set with `--build-arg PY3_VERSION=10` for Python 3.10.
ARG PY3_VERSION=10

# Set up timezone to avoid getting stuck at `tzdata` setup.
ENV TZ=America/Montreal
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update
RUN apt install -y tzdata

# Install `software-properties-common` to obtain add-apt-repository.
RUN apt update && apt install -y software-properties-common

# Add deadsnake PPA to get Python 3.10.
RUN add-apt-repository ppa:deadsnakes/ppa

# Install necessary packages.
RUN apt-get update && apt-get install -y git curl wget software-properties-common python3.$PY3_VERSION python3.$PY3_VERSION-dev libgmp-dev gcc-9 g++-9 tmux vim

# Install distutils if not Python 3.10 to get `distutils.util`.
RUN if [ "$PY3_VERSION" != "10" ]; then apt install -y python3-distutils python3-apt; fi

# Install pip.
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.$PY3_VERSION

# Download bazel.
RUN wget https://github.com/bazelbuild/bazel/releases/download/6.2.1/bazel-6.2.1-linux-x86_64
RUN chmod +x /bazel-6.2.1-linux-x86_64
RUN mv /bazel-6.2.1-linux-x86_64 /usr/bin/bazel

# Add python alternatives.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.$PY3_VERSION 1

# Override gcc/g++.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# Install some basic things for all python versions.
RUN echo 1 | update-alternatives --config python3
# Get latest `pip`.
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN python3 -m pip install --no-cache --upgrade pip setuptools

ADD /envlogger/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache -r /tmp/requirements.txt grpcio-tools


# Add `python` so that `/usr/bin/env` finds it. This is used by `bazel`.
RUN ln -s /usr/bin/python3 /usr/bin/python

ADD . /envlogger/
WORKDIR /envlogger
