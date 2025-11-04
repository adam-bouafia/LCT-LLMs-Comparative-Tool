#!/bin/bash

# Docker run script for LCT
# This script runs the LCT container with proper volume mounts and settings

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}    LCT Docker Run Script${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Error: Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if image exists
if ! docker image inspect lct:latest &> /dev/null; then
    echo -e "${YELLOW}Warning: lct:latest image not found${NC}"
    echo "Building image first..."
    ./docker-build.sh
fi

# Parse arguments
GPU_SUPPORT=false
PRIVILEGED=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu)
            GPU_SUPPORT=true
            shift
            ;;
        --privileged)
            PRIVILEGED=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--gpu] [--privileged]"
            exit 1
            ;;
    esac
done

# Create necessary directories if they don't exist
mkdir -p experiments saved_configs logs

echo -e "${GREEN}Starting LCT container...${NC}"
echo ""

# Build docker run command
DOCKER_CMD="docker run -it --rm"
DOCKER_CMD="$DOCKER_CMD --name lct-runner"

# Add volume mounts
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/experiments:/app/experiments"
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/saved_configs:/app/saved_configs"
DOCKER_CMD="$DOCKER_CMD -v $(pwd)/logs:/app/logs"
DOCKER_CMD="$DOCKER_CMD -v lct_huggingface_cache:/data/huggingface"
DOCKER_CMD="$DOCKER_CMD -v lct_models_cache:/data/models"

# Add environment variables
if [ -f .env ]; then
    echo "Loading environment from .env file..."
    DOCKER_CMD="$DOCKER_CMD --env-file .env"
fi

# Add GPU support if requested
if [ "$GPU_SUPPORT" = true ]; then
    echo "Enabling GPU support..."
    DOCKER_CMD="$DOCKER_CMD --gpus all"
fi

# Add privileged mode if requested (for energy profiling)
if [ "$PRIVILEGED" = true ]; then
    echo "Running in privileged mode (for energy profiling)..."
    DOCKER_CMD="$DOCKER_CMD --privileged"
fi

# Add image name
DOCKER_CMD="$DOCKER_CMD lct:latest"

echo ""
echo "Command: $DOCKER_CMD"
echo ""

# Run the container
eval $DOCKER_CMD

echo ""
echo -e "${GREEN}✓ Container stopped${NC}"
