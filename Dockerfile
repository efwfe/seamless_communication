FROM nvidia/cuda:11.7.1-runtime-ubuntu22.04

RUN sed -i s/archive.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
RUN sed -i s/security.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
RUN sed -i s/ports.ubuntu.com/mirrors.aliyun.com/g /etc/apt/sources.list
# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN apt update && apt-get -y install git wget \
    python3.10 python3-pip \
    build-essential libgl-dev libglib2.0-0 wget

RUN pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
RUN pip config set global.trusted-host mirrors.cloud.tencent.com
# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
ADD condarc ~/.condarc
COPY ./requirements.txt /tmp/requirements.txt
RUN conda install -y -c conda-forge libsndfile
RUN pip install --no-cache-dir -r  /tmp/requirements.txt
RUN pip install --no-cache-dir uvicorn fastapi gradio
WORKDIR /home
COPY . .
RUN pip install .
ENV PYTHONPATH "${PYTHONPATH}:/home"