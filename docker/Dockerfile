FROM ubuntu:22.04

# ARG variable for Python 3 minor version.
# This can be set with `--build-arg PY3_VERSION=10` for Python 3.10.
ARG PY3_VERSION=11

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
RUN apt-get update && apt-get install -y git curl wget software-properties-common python3.$PY3_VERSION python3.$PY3_VERSION-dev libgmp-dev gcc g++ tmux vim

# Install distutils for Python 3.11 (distutils is removed in 3.12+).
RUN if [ "$PY3_VERSION" = "11" ]; then apt install -y python3.11-distutils; fi

# Install pip.
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.$PY3_VERSION

# Download bazel.
RUN wget https://github.com/bazelbuild/bazel/releases/download/7.1.0/bazel-7.1.0-linux-x86_64
RUN chmod +x /bazel-7.1.0-linux-x86_64
RUN mv /bazel-7.1.0-linux-x86_64 /usr/bin/bazel

# Add python alternatives.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.$PY3_VERSION 1



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
