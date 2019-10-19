FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

# Usage
# $ sudo docker build -t="account/name:tag" .

# 以下のコマンドをbashにする.
SHELL ["/bin/bash", "-c"]

# aptの更新
RUN apt-get upgrade -y
RUN apt-get update

# install software
RUN apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    unzip \
    curl \
    wget \
    vim \
    tmux \
    htop \
    less \
    locate \
    ca-certificates \
    libsm6 \
    libxext6 \
    libxrender1 \
    libncurses5-dev \
    libncursesw5-dev \
    libglib2.0-0

# install Python
RUN apt-get install -y --no-install-recommends \
    python3.6 \
    python3.6-dev \
    python3.6-distutils

ENV PATH=$PATH:$HOME/.local/bin \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

RUN echo 'alias python=python3.6' >> ~/.bashrc
RUN source ~/.bashrc

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.6 get-pip.py && \
    rm get-pip.py