#!/bin/bash

# Docker push script for LCT
# Pushes Docker images to container registries (Docker Hub or GitHub Container Registry)

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}    LCT Docker Push Script${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# Parse arguments
REGISTRY="docker.io"  # Default to Docker Hub
USERNAME=""
VERSION=""
LATEST=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --username)
            USERNAME="$2"
            shift 2
            ;;
        --version)
            VERSION="$2"
            shift 2
            ;;
        --no-latest)
            LATEST=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--registry REGISTRY] [--username USERNAME] [--version VERSION] [--no-latest]"
            echo ""
            echo "Options:"
            echo "  --registry    Container registry (docker.io, ghcr.io) [default: docker.io]"
            echo "  --username    Registry username/organization"
            echo "  --version     Version tag (e.g., 2.0.0)"
            echo "  --no-latest   Don't tag as latest"
            exit 1
            ;;
    esac
done

# Prompt for username if not provided
if [ -z "$USERNAME" ]; then
    echo -e "${YELLOW}Enter your registry username/organization:${NC}"
    read -p "> " USERNAME
fi

# Prompt for version if not provided
if [ -z "$VERSION" ]; then
    echo -e "${YELLOW}Enter version tag (e.g., 2.0.0):${NC}"
    read -p "> " VERSION
fi

# Validate inputs
if [ -z "$USERNAME" ] || [ -z "$VERSION" ]; then
    echo -e "${RED}Error: Username and version are required${NC}"
    exit 1
fi

# Determine registry-specific image naming
if [[ "$REGISTRY" == "ghcr.io" ]]; then
    IMAGE_NAME="${REGISTRY}/${USERNAME}/lct"
else
    IMAGE_NAME="${REGISTRY}/${USERNAME}/lct"
fi

echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Registry: $REGISTRY"
echo "  Username: $USERNAME"
echo "  Image: $IMAGE_NAME"
echo "  Version: $VERSION"
echo "  Tag latest: $LATEST"
echo ""

# Check if image exists locally
if ! docker image inspect lct:latest &> /dev/null; then
    echo -e "${YELLOW}Local image 'lct:latest' not found. Building...${NC}"
    ./docker-build.sh
fi

# Login to registry
echo -e "${BLUE}Logging into $REGISTRY...${NC}"
if [[ "$REGISTRY" == "ghcr.io" ]]; then
    echo -e "${YELLOW}Please ensure you have a GitHub Personal Access Token with 'write:packages' scope${NC}"
    echo -e "${YELLOW}Create one at: https://github.com/settings/tokens${NC}"
fi

docker login "$REGISTRY"

if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to login to registry${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}Tagging image...${NC}"

# Tag with version
docker tag lct:latest "${IMAGE_NAME}:${VERSION}"
echo "  ✓ Tagged as ${IMAGE_NAME}:${VERSION}"

# Tag as latest if requested
if [ "$LATEST" = true ]; then
    docker tag lct:latest "${IMAGE_NAME}:latest"
    echo "  ✓ Tagged as ${IMAGE_NAME}:latest"
fi

echo ""
echo -e "${GREEN}Pushing to registry...${NC}"

# Push version tag
docker push "${IMAGE_NAME}:${VERSION}"
echo "  ✓ Pushed ${IMAGE_NAME}:${VERSION}"

# Push latest tag if requested
if [ "$LATEST" = true ]; then
    docker push "${IMAGE_NAME}:latest"
    echo "  ✓ Pushed ${IMAGE_NAME}:latest"
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Successfully pushed to $REGISTRY!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${GREEN}Pull your image with:${NC}"
echo "  docker pull ${IMAGE_NAME}:${VERSION}"
if [ "$LATEST" = true ]; then
    echo "  docker pull ${IMAGE_NAME}:latest"
fi
echo ""
echo -e "${GREEN}Run your image with:${NC}"
echo "  docker run -it --rm ${IMAGE_NAME}:${VERSION}"
echo ""
