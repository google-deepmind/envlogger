FROM ubuntu:20.04

# Set up timezone to avoid getting stuck at `tzdata` setup.
ENV TZ=America/Montreal
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update
RUN apt install -y tzdata

RUN apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update

# Install necessary packages.
RUN apt-get update && apt-get \
    install -y git \
    software-properties-common \
    curl \
    wget \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    libgmp-dev \
    gcc-9 \
    g++-9 \
    tmux \
    vim

# Download bazel.
RUN wget https://github.com/bazelbuild/bazel/releases/download/6.1.0/bazel-6.1.0-linux-x86_64
RUN chmod +x /bazel-6.1.0-linux-x86_64
RUN mv /bazel-6.1.0-linux-x86_64 /usr/bin/bazel

# Add python alternatives.
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Override gcc/g++.
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9

# Install some basic things for all python versions.
RUN echo 1 | update-alternatives --config python3
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
RUN python3 -m pip install --no-cache --upgrade pip setuptools

ADD /envlogger/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --no-cache -r /tmp/requirements.txt grpcio-tools

# Add `python` so that `/usr/bin/env` finds it. This is used by `bazel`.
RUN ln -s /usr/bin/python3 /usr/bin/python

ADD . /envlogger/
WORKDIR /envlogger