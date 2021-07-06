FROM python:3.8

# Install bazel.
RUN echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | \
    tee /etc/apt/sources.list.d/bazel.list && \
    curl https://bazel.build/bazel-release.pub.gpg | \
    apt-key add - && \
    apt-get update && apt-get install -y bazel=3.7.2

# For development purposes.
RUN apt-get install -y tmux vim

ADD . /envlogger/
WORKDIR /envlogger
