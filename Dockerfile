FROM nvidia/cuda:10.2-cudnn7-runtime-ubuntu18.04

# Set up cudnn
ENV CUDNN_VERSION 7.6.5.32
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libcudnn7=$CUDNN_VERSION-1+cuda10.2 \
    && apt-mark hold libcudnn7 \
    && rm -rf /var/lib/apt/lists/*

# Install Python and utilities
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.7 python3-dev python3-pip python3-wheel python3-setuptools \
        git vim ssh wget gcc cmake build-essential libblas3 libblas-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy package
WORKDIR /home/dgmvae/
COPY ./bin/ ./bin/
COPY ./dgmvae/ ./dgmvae/
COPY ./extra_metrics_configs/ ./extra_metrics_configs/
COPY ./src/ ./src/
COPY ./tests/ ./tests/
COPY ./setup.py ./setup.py

# Install package
RUN pip3 install -e . && rm -rf /root/.cache
