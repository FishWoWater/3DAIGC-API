# Multi-stage Dockerfile for 3DAIGC-API
FROM nvidia/cuda:12.1-devel-ubuntu20.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    curl \
    git \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libegl1 \
    libegl1-mesa \
    libgl1-mesa-dev \
    libgles2-mesa-dev \
    libosmesa6-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh
ENV PATH="/opt/conda/bin:${PATH}"

# Initialize conda
RUN conda init bash

# Create conda environment with Python 3.10
RUN conda create -n 3daigc-api python=3.10 -y

# Make conda environment activation persistent
SHELL ["conda", "run", "-n", "3daigc-api", "/bin/bash", "-c"]

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Set working directory
WORKDIR /app

# Copy the entire project
COPY . .

# Install TRELLIS dependencies
WORKDIR /app/thirdparty/TRELLIS
RUN bash setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
RUN pip install kaolin -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu121.html

# Install PartField dependencies
WORKDIR /app/thirdparty/PartField
RUN pip install lightning==2.2 h5py yacs trimesh scikit-image loguru boto3
RUN pip install mesh2sdf tetgen pymeshlab plyfile einops libigl polyscope potpourri3d simple_parsing arrgh open3d psutil
RUN pip install torch-scatter torch_cluster -f https://data.pyg.org/whl/torch-2.4.0+cu121.html

# Install Hunyuan3D 2.0 dependencies
WORKDIR /app/thirdparty/Hunyuan3D-2
RUN pip install -r requirements.txt
RUN pip install -e .

# Build custom rasterizer for Hunyuan3D 2.0
WORKDIR /app/thirdparty/Hunyuan3D-2/hy3dgen/texgen/custom_rasterizer
RUN python3 setup.py install

# Build differentiable renderer for Hunyuan3D 2.0
WORKDIR /app/thirdparty/Hunyuan3D-2/hy3dgen/texgen/differentiable_renderer
RUN python3 setup.py install

# Install Hunyuan3D 2.1 dependencies
WORKDIR /app/thirdparty/Hunyuan3D-2.1/hy3dpaint/custom_rasterizer
RUN pip install -e .

# Build differentiable renderer for Hunyuan3D 2.1
WORKDIR /app/thirdparty/Hunyuan3D-2.1/hy3dpaint/DifferentiableRenderer
RUN bash compile_mesh_painter.sh

# Install Hunyuan3D 2.1 requirements
WORKDIR /app/thirdparty/Hunyuan3D-2.1
RUN pip install -r requirements-inference.txt

# Install HoloPart dependencies
WORKDIR /app/thirdparty/HoloPart
RUN pip install -r requirements.txt

# Install UniRig dependencies
WORKDIR /app/thirdparty/UniRig
RUN pip install spconv-cu120 pyrender fast-simplification python-box timm

# Install PartPacker dependencies
WORKDIR /app/thirdparty/PartPacker
RUN pip install meshiki fpsample kiui pymcubes einops

# Install PartCrafter dependencies (if requirements exist)
WORKDIR /app/thirdparty/PartCrafter
RUN if [ -f "requirements.txt" ]; then pip install -r requirements.txt; fi

# Install main project dependencies
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install -r requirements-test.txt
RUN pip install huggingface_hub

# Create necessary directories
RUN mkdir -p /app/uploads /app/data

# Set environment variables for runtime
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV CONDA_DEFAULT_ENV=3daigc-api

# Expose port for FastAPI
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command to run the FastAPI server
CMD ["conda", "run", "-n", "3daigc-api", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"] 