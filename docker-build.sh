#!/bin/bash

# Docker build script for LCT
# This script builds the Docker image with proper tags

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}    LCT Docker Build Script${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}Error: Docker is not installed${NC}"
    echo "Please install Docker first: https://docs.docker.com/get-docker/"
    exit 1
fi

# Get version from user or use default
VERSION=${1:-latest}

echo -e "${GREEN}Building LCT Docker image...${NC}"
echo "Version: $VERSION"
echo ""

# Build the image
docker build \
    -t lct:${VERSION} \
    -t lct:latest \
    --build-arg BUILDKIT_INLINE_CACHE=1 \
    .

echo ""
echo -e "${GREEN}✓ Build completed successfully!${NC}"
echo ""
echo "Image tags:"
echo "  - lct:${VERSION}"
echo "  - lct:latest"
echo ""
echo "Next steps:"
echo "  1. Run with Docker Compose: docker-compose up"
echo "  2. Run directly: docker run -it lct:latest"
echo "  3. Or use: ./docker-run.sh"
