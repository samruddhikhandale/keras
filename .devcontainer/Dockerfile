FROM mcr.microsoft.com/devcontainers/python:3.8

# Install Bazel
RUN sudo apt --no-install-recommends -y install wget
RUN architecture="$(dpkg --print-architecture)" && \
    wget https://github.com/bazelbuild/bazelisk/releases/download/v1.11.0/bazelisk-linux-${architecture} && \
    chmod a+x bazelisk-linux-${architecture} && \
    mv bazelisk-linux-${architecture} /usr/bin/bazel

COPY welcome-message.txt /usr/local/etc/vscode-dev-containers/first-run-notice.txt
