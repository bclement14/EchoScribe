# Use NVIDIA’s official CUDA runtime image with cuDNN 8 on Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment to non-interactive (avoid prompts during installation)
ENV DEBIAN_FRONTEND=noninteractive

# Update apt and install system dependencies:
# - python3.10 and related packages,
# - ffmpeg for audio processing,
# - git and curl for fetching code, and optionally zsh if you need an interactive shell.
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3.10-dev python3-pip \
    ffmpeg git curl zsh && \
    rm -rf /var/lib/apt/lists/*

# Ensure that "python" points to python3.10
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip

RUN pip install huggingface_hub

# Create huggingface cache directory
RUN mkdir -p /root/.cache/huggingface/hub

# Pre-download the model manually (example for large-v3)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='openai/whisper-large-v3', cache_dir='/root/.cache/huggingface/hub')"

# Set Huggingface cache env var so whisperx knows where to look
ENV HUGGINGFACE_HUB_CACHE=/root/.cache/huggingface/hub

# Install WhisperX from GitHub (this will also install faster-whisper and other dependencies)
RUN pip install git+https://github.com/m-bain/whisperX.git@v3.3.1

# Set working directory inside the container
WORKDIR /app

# Optional: expose /app as a volume so your local files are accessible inside the container
VOLUME ["/app"]

# By default, run WhisperX help so you see available options; you can override this command when running the container.
CMD ["whisperx", "--help"]