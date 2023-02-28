FROM mcr.microsoft.com/devcontainers/python:3.8

# Install Bazel
RUN sudo apt --no-install-recommends -y install wget

ARG architecture="$(dpkg --print-architecture)"
RUN wget https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-${architecture} && \
    chmod a+x bazelisk-linux-${architecture} && \
    mv bazelisk-linux-${architecture} /usr/bin/bazel

COPY welcome-message.txt /usr/local/etc/vscode-dev-containers/first-run-notice.txt
