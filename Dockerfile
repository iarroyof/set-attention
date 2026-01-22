# Base image with CUDA 12.4 runtime and cuDNN, good match for torch 2.5.1+cu124
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

ENV HF_HOME=/workspace/.hf \
    HF_DATASETS_CACHE=/workspace/.hf/datasets \
    HF_HUB_CACHE=/workspace/.hf/hub \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

# --- System dependencies ----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-venv python3-pip \
    git \
    build-essential \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make "python" point to python3.11 and upgrade pip
RUN ln -s /usr/bin/python3.11 /usr/bin/python && \
    python -m pip install --upgrade pip
RUN pip install --no-cache-dir wandb

WORKDIR /workspace

# --- Copy dependency descriptors first (better build cache) -----------------
COPY requirements.txt requirements-experiments.txt requirements-dev.txt ./ 
COPY scripts/update_requirements.py ./scripts/update_requirements.py

# Install Python deps (dev includes experiments and runtime)
RUN python -m pip install -r requirements-dev.txt

# --- Copy the rest of the project ------------------------------------------
COPY . .

# Install set-attention in editable mode
RUN python -m pip install -e .

# Optional: small sanity check at build time (will fail build if broken)
RUN python -c "import torch; import triton; \
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda, 'cuda?', torch.cuda.is_available()); \
print('Triton:', triton.__version__)"

# Default: drop into a shell; override CMD when running if you like
CMD ["/bin/bash"]
