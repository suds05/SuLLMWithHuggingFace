# Use a more recent base Python image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies with better error handling
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Install PyTorch and Transformers
# Ensure to install the correct PyTorch version for your system (e.g., CPU or GPU)
# For CPU-only:
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
# For CUDA-enabled GPU (example for CUDA 12.1):
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install packages for https://huggingface.co/openai/gpt-oss-20b
RUN pip install -U transformers kernels torch huggingface_hub accelerate rich aiohttp transformers[serving] bitsandbytes

# Print huggingface environment
RUN hf env

# Download the OSS model
RUN hf download openai/gpt-oss-20b

# List all cached models
# RUN huggingface-cli list --local

# Show cache directory
# RUN huggingface-cli list --cache-dir

# Copy your application code into the container
COPY test_oss_model.py .
COPY load_oss_model_with_disk_offload.py .

# Run the OSS model tests during container build to verify everything works
# RUN python test_oss_model.py

# Default command to run when the container starts
# CMD ["python", "test_oss_model.py"]