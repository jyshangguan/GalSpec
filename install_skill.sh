#!/bin/bash
# Installation/Update script for SAGAN Claude Code skill
# This script copies skill files to ~/.claude/skills so they are properly loaded

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Package directory
PACKAGE_DIR="/Users/shangguan/Softwares/my_modules/SAGAN"

# Source skills directory
SOURCE_SKILLS_DIR="$PACKAGE_DIR/skills"

# Skill installation directory
SKILL_DIR="$HOME/.claude/skills/sagan-spectral-fitting"

echo "============================================================"
echo "SAGAN Spectral Fitting - Claude Code Skill Installer/Updater"
echo "============================================================"
echo ""

# Check if package directory exists
if [ ! -d "$PACKAGE_DIR" ]; then
    echo -e "${RED}Error: Package directory not found: $PACKAGE_DIR${NC}"
    echo "Please ensure the package is installed in the correct location."
    exit 1
fi

# Check if source skills directory exists
if [ ! -d "$SOURCE_SKILLS_DIR" ]; then
    echo -e "${RED}Error: Skills directory not found: $SOURCE_SKILLS_DIR${NC}"
    exit 1
fi

# Detect if this is an update or fresh installation
if [ -e "$SKILL_DIR" ]; then
    echo -e "${BLUE}Mode: UPDATE (removing existing installation)${NC}"
    echo -e "${YELLOW}Removing existing installation at: $SKILL_DIR${NC}"

    # Check if it's a symlink or directory
    if [ -L "$SKILL_DIR" ]; then
        echo "  (existing installation is a symbolic link)"
    elif [ -d "$SKILL_DIR" ]; then
        echo "  (existing installation is a directory)"
    fi

    rm -rf "$SKILL_DIR"
    echo -e "${GREEN}✓ Existing installation removed${NC}"
else
    echo -e "${BLUE}Mode: FRESH INSTALLATION${NC}"
fi

echo ""

# Create the skill directory by copying files
echo -e "${YELLOW}Copying skill files from source...${NC}"
echo "  Source: $SOURCE_SKILLS_DIR"
echo "  Target: $SKILL_DIR"
echo ""

# Create target directory
mkdir -p "$SKILL_DIR"

# Copy all files from source to target
# -a: archive mode (preserves permissions, times, etc.)
# -v: verbose
cp -av "$SOURCE_SKILLS_DIR"/. "$SKILL_DIR/"

echo ""
echo -e "${GREEN}✓ Skill files copied successfully${NC}"
echo ""

# Verify installation
if [ -d "$SKILL_DIR" ]; then
    echo "============================================================"
    echo -e "${GREEN}Installation/Update Complete!${NC}"
    echo "============================================================"
    echo ""
    echo "Skill location: $SKILL_DIR"
    echo "Source directory: $SOURCE_SKILLS_DIR"
    echo ""
    echo "Installed files:"
    echo "  - SKILL.md (main skill definition)"
    echo "  - fitting_strategies/ (strategy guides)"
    if [ -d "$SKILL_DIR/fitting_strategies" ]; then
        ls -1 "$SKILL_DIR/fitting_strategies/" | sed 's/^/      • /'
    fi
    echo "  - function_reference/ (function documentation)"
    if [ -d "$SKILL_DIR/function_reference" ]; then
        ls -1 "$SKILL_DIR/function_reference/" | sed 's/^/      • /'
    fi
    echo "  - typical_bugs.md"
    echo ""
    echo -e "${YELLOW}IMPORTANT: Restart Claude Code completely (quit and reopen)${NC}"
    echo ""
    echo "To test the skill:"
    echo "1. Restart Claude Code"
    echo "2. In a new session, ask something like:"
    echo ""
    echo "   'Help me fit a Type 1 AGN spectrum with Hα and Hβ lines'"
    echo "   'Show me how to use SAGAN to fit [S II] doublet for narrow line template'"
    echo "   'Fit the [S II] 6716/6731 doublet and derive narrow line template'"
    echo ""
    echo "To update the skill in the future:"
    echo "  Simply run this script again - it will automatically update"
    echo "  the installation with the latest files from $SOURCE_SKILLS_DIR"
    echo ""
else
    echo -e "${RED}✗ Installation failed${NC}"
    exit 1
fi

echo "============================================================"
