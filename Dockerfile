FROM nvidia/cuda:11.3.1-base-ubuntu20.04

# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list

ARG DEBIAN_FRONTEND=noninteractive
# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    sox \
    libsox-dev \
    libsox-fmt-all \
    build-essential \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    ffmpeg \
    libsm6 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app


USER root
RUN groupadd -g 1019 n.belousov

RUN adduser --disabled-password --uid 1018 --gid 1019 --gecos '' --shell /bin/bash n.belousov \
 && chown -R n.belousov:n.belousov /app
RUN echo "n.belousov ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-n.belousov
USER n.belousov

ENV HOME=/home/n.belousov



RUN mkdir $HOME/.cache $HOME/.config \
 && chmod -R 777 $HOME

# Set up the Conda environment
ENV CONDA_AUTO_UPDATE_CONDA=false \
    PATH=$HOME/miniconda/bin:$PATH
COPY environment.yml /app/environment.yml
RUN curl -sLo ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-py39_23.1.0-1-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda env update -n base -f /app/environment.yml \
 && rm /app/environment.yml \
 && conda clean -ya

RUN pip install notebook
RUN pip install jupyterlab
# RUN git clone https://github.com/nokiroki/NLP-Transactions.git
CMD jupyter notebook --allow-root --ip='0.0.0.0' --port=8890 --NotebookApp.token='' --NotebookApp.password=''