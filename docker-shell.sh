#!/bin/bash

# Docker shell script for LCT
# This script opens a shell in a running or new LCT container

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}    LCT Docker Shell${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Error: Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if container is already running
CONTAINER_ID=$(docker ps -q -f name=lct-tool)

if [ -n "$CONTAINER_ID" ]; then
    echo -e "${GREEN}Found running container: $CONTAINER_ID${NC}"
    echo "Opening shell in existing container..."
    echo ""
    docker exec -it $CONTAINER_ID /bin/bash
else
    echo -e "${YELLOW}No running container found${NC}"
    echo "Starting new container with shell..."
    echo ""
    
    # Check if image exists
    if ! docker image inspect lct:latest &> /dev/null; then
        echo -e "${YELLOW}Image lct:latest not found${NC}"
        echo "Building image first..."
        ./docker-build.sh
    fi
    
    # Create necessary directories
    mkdir -p experiments saved_configs logs
    
    # Start container with shell
    docker run -it --rm \
        --name lct-shell \
        -v $(pwd)/experiments:/app/experiments \
        -v $(pwd)/saved_configs:/app/saved_configs \
        -v $(pwd)/logs:/app/logs \
        -v lct_huggingface_cache:/data/huggingface \
        -v lct_models_cache:/data/models \
        lct:latest \
        /bin/bash
fi

echo ""
echo -e "${GREEN}✓ Shell session ended${NC}"
