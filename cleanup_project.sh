#!/bin/bash
# LCT Project Cleanup Script
# Removes redundant and unused files to maintain a clean codebase

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           LCT Project Cleanup Script                        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get project directory
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

echo -e "${YELLOW}⚠️  This script will remove the following:${NC}"
echo ""
echo "1. Old experiment backup folders (3.4 MB)"
echo "2. Redundant test experiment configs"
echo "3. Nested duplicate experiment folders"
echo "4. Python cache directories (__pycache__)"
echo ""

# Count files to be removed
BACKUP_COUNT=$(find experiments -name "*backup*" -type d 2>/dev/null | wc -l)
PYCACHE_COUNT=$(find . -name "__pycache__" -type d 2>/dev/null | wc -l)

echo -e "${GREEN}Files to be removed:${NC}"
echo "  - $BACKUP_COUNT backup folders"
echo "  - $PYCACHE_COUNT __pycache__ directories"
echo "  - Redundant experiment configs"
echo ""

# Ask for confirmation
read -p "Do you want to proceed? (yes/no): " CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
    echo -e "${RED}Cleanup cancelled.${NC}"
    exit 0
fi

echo ""
echo -e "${BLUE}Starting cleanup...${NC}"
echo ""

# 1. Remove backup folders
echo -e "${GREEN}[1/5] Removing backup folders...${NC}"
find experiments -name "*backup*" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Removed $BACKUP_COUNT backup folders"

# 2. Remove redundant experiment configs (keep only saved_configs/)
echo -e "${GREEN}[2/5] Removing redundant experiment configs...${NC}"
# Remove root level test experiments
rm -f experiments/RunnerConfig.py 2>/dev/null || true
rm -f experiments/experiment_info.json 2>/dev/null || true
echo "  ✓ Removed root experiment configs"

# Remove nested duplicate folders
rm -rf experiments/beta/beta 2>/dev/null || true
rm -rf experiments/omega/omega 2>/dev/null || true
rm -rf experiments/Omega/Omega 2>/dev/null || true
echo "  ✓ Removed nested duplicate folders"

# Remove uppercase Omega (keep lowercase omega)
rm -rf experiments/Omega 2>/dev/null || true
echo "  ✓ Removed duplicate Omega folder"

# 3. Remove testing experiment (use saved_configs instead)
echo -e "${GREEN}[3/5] Removing testing experiment folder...${NC}"
rm -rf experiments/testing 2>/dev/null || true
echo "  ✓ Removed testing experiment folder"

# 4. Clean Python cache
echo -e "${GREEN}[4/5] Cleaning Python cache...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true
find . -type f -name "*.pyo" -delete 2>/dev/null || true
echo "  ✓ Removed Python cache files"

# 5. Clean pytest cache
echo -e "${GREEN}[5/5] Cleaning pytest cache...${NC}"
rm -rf .pytest_cache 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
echo "  ✓ Removed pytest cache"

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Cleanup completed successfully!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Show disk space saved
echo -e "${YELLOW}Checking disk space saved...${NC}"
echo ""

# Show remaining experiment structure
echo -e "${GREEN}Current experiment structure:${NC}"
tree -L 2 experiments 2>/dev/null || find experiments -maxdepth 2 -type d

echo ""
echo -e "${YELLOW}💡 Recommendations:${NC}"
echo "  1. Use saved_configs/ for your experiment presets"
echo "  2. Use the CLI's Quick Presets instead of manual RunnerConfig.py files"
echo "  3. Run this script periodically to keep the project clean"
echo ""
