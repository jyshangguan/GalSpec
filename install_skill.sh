#!/bin/bash
# Installation script for SAGAN Claude Code skill

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Package directory
PACKAGE_DIR="/Users/shangguan/Softwares/my_modules/SAGAN"

# Skill installation directory
SKILL_DIR="$HOME/.claude/skills/sagan-spectral-fitting"

echo "============================================================"
echo "SAGAN Spectral Fitting - Claude Code Skill Installer"
echo "============================================================"
echo ""

# Check if package directory exists
if [ ! -d "$PACKAGE_DIR" ]; then
    echo -e "${RED}Error: Package directory not found: $PACKAGE_DIR${NC}"
    echo "Please ensure the package is installed in the correct location."
    exit 1
fi

# Create skill directory if it doesn't exist
echo -e "${YELLOW}Creating skill directory...${NC}"
mkdir -p "$SKILL_DIR"

# Copy skill file
echo -e "${YELLOW}Installing skill file...${NC}"
cp "$PACKAGE_DIR/skills/sagan-spectral-fitting/SKILL.md" "$SKILL_DIR/"

# Verify installation
if [ -f "$SKILL_DIR/SKILL.md" ]; then
    echo -e "${GREEN}✓ Skill file installed successfully${NC}"
    echo ""
    echo "Skill location: $SKILL_DIR/SKILL.md"
    echo ""
    echo "Documentation files:"
    echo "  - Main guide: $PACKAGE_DIR/skills/sagan_spectral_fitting.md"
    echo "  - Type 1 AGN strategy: $PACKAGE_DIR/skills/strategy_types/type1_agn.md"
    echo "  - Function reference: $PACKAGE_DIR/skills/function_reference.md"
    echo ""
    echo -e "${YELLOW}IMPORTANT: Restart Claude Code completely (quit and reopen)${NC}"
    echo ""
    echo "To test the skill:"
    echo "1. Restart Claude Code"
    echo "2. Open a new terminal"
    echo "3. Ask: 'Help me fit a Type 1 AGN spectrum with Hα and Hβ lines'"
    echo "4. Or: 'Show me how to use SAGAN to fit BAL troughs'"
    echo ""
    echo "Example queries:"
    echo "  - 'Fit an emission line spectrum with broad and narrow components'"
    echo "  - 'How do I model iron templates in SAGAN?'"
    echo "  - 'Set up MCMC fitting for a Type 1 AGN'"
    echo "  - 'Tie absorption parameters across Hα and Hβ lines'"
    echo "  - 'Measure black hole mass from broad line width'"
    echo ""
else
    echo -e "${RED}✗ Installation failed${NC}"
    exit 1
fi

echo "============================================================"
