# Multi-stage Dockerfile for LCT (LLM Comparative Tool)
# Optimized for production deployment with environmental tracking

# ============================================================================
# Stage 1: Base Image with System Dependencies
# ============================================================================
FROM python:3.11-slim AS base

LABEL maintainer="Adam Bouafia <a.bouafia@student.vu.nl>"
LABEL description="LLM Comparative Tool with Environmental Tracking"
LABEL version="2.0.0"

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

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/config/requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# ============================================================================
# Stage 3: Application
# ============================================================================
FROM dependencies AS application

# Copy application code
COPY --chown=lct:lct . /app/

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
