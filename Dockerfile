# Multi-stage Dockerfile for LCT (LLM Comparative Tool)
# Supports both CPU-only and GPU (CUDA) variants
# 
# Build variants:
#   CPU-only (~1.5GB):  docker build -t lct:latest .
#   GPU/CUDA (~4.5GB):  docker build --build-arg TORCH_VARIANT=cuda -t lct:gpu .

# ============================================================================
# Stage 1: Base Image with System Dependencies
# ============================================================================
FROM python:3.11-slim AS base

# Build argument to select PyTorch variant (cpu or cuda)
ARG TORCH_VARIANT=cpu
ENV TORCH_VARIANT=${TORCH_VARIANT}

LABEL maintainer="Adam Bouafia <a.bouafia@student.vu.nl>"
LABEL description="LLM Comparative Tool with Environmental Tracking"
LABEL version="2.0.0"
LABEL torch.variant="${TORCH_VARIANT}"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    git \
    # System monitoring tools
    htop \
    sysstat \
    # For energy profiling
    linux-cpupower \
    # Cleanup
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app user (non-root for security)
RUN useradd -m -u 1000 lct && \
    mkdir -p /app /data /experiments /models && \
    chown -R lct:lct /app /data /experiments /models

# Set working directory
WORKDIR /app

# ============================================================================
# Stage 2: Dependencies Installation
# ============================================================================
FROM base AS dependencies

# Copy requirements first (for layer caching)
COPY --chown=lct:lct config/requirements.txt /app/config/

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch based on variant (CPU or CUDA)
# CPU-only: ~1.5GB (default, fast CI/CD builds)
# CUDA: ~4.5GB (GPU support for production)
RUN if [ "$TORCH_VARIANT" = "cuda" ]; then \
        echo "Installing PyTorch with CUDA support..."; \
        pip install --no-cache-dir torch torchvision torchaudio; \
    else \
        echo "Installing PyTorch CPU-only..."; \
        pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu; \
    fi

# Install remaining Python dependencies (excluding torch to avoid reinstalling)
# Use grep to filter out torch lines from requirements.txt to prevent CUDA reinstall
RUN grep -v "^torch" /app/config/requirements.txt > /tmp/requirements_no_torch.txt && \
    pip install --no-cache-dir -r /tmp/requirements_no_torch.txt || \
    pip install --no-cache-dir \
    pandas psutil tabulate dill jsonpickle \
    transformers sentence-transformers huggingface_hub \
    nltk rouge-score bert-score evaluate datasets sacrebleu scikit-learn \
    codecarbon nvidia-ml-py \
    rich matplotlib seaborn numpy requests \
    jupyter ipython plotly

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies AS application

# Copy application code (only necessary files, .dockerignore handles exclusions)
# Copy app directory and config
COPY --chown=lct:lct app/ /app/app/
COPY --chown=lct:lct config/ /app/config/
COPY --chown=lct:lct README.md LICENSE /app/

# Create necessary directories
RUN mkdir -p \
    /app/data/huggingface \
    /app/data/models \
    /app/data/datasets \
    /app/data/cache \
    /app/experiments \
    /app/saved_configs \
    /app/logs

# Set permissions
RUN chmod +x /app/*.sh 2>/dev/null || true

# Switch to non-root user
USER lct

# Set environment variables for HuggingFace cache
ENV HF_HOME=/data/huggingface \
    TRANSFORMERS_CACHE=/data/huggingface/transformers \
    HF_DATASETS_CACHE=/data/huggingface/datasets \
    TORCH_HOME=/data/models

# Expose port for potential web interface
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command
CMD ["python", "app/src/ui/interactive_lct.py"]
