#!/bin/bash
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_SKILLS_DIR="$SCRIPT_DIR/skills"
TARGET_ROOT="$HOME/.claude/skills"
SKILL_NAME="sagan"
TARGET_DIR="$TARGET_ROOT/$SKILL_NAME"

echo "Installing Claude skill: $SKILL_NAME"
echo "Source: $SOURCE_SKILLS_DIR"
echo "Target: $TARGET_DIR"

if [ ! -d "$SOURCE_SKILLS_DIR" ]; then
  echo -e "${RED}Error: source skills directory not found:${NC} $SOURCE_SKILLS_DIR"
  exit 1
fi

mkdir -p "$TARGET_ROOT"

# Remove old target if it exists
if [ -e "$TARGET_DIR" ] || [ -L "$TARGET_DIR" ]; then
  rm -rf "$TARGET_DIR"
fi

# Symlink instead of copying
ln -s "$SOURCE_SKILLS_DIR" "$TARGET_DIR"

echo -e "${GREEN}Done.${NC}"
echo "Claude skill installed at: $TARGET_DIR"
echo
echo "You can test with:"
echo "  /skills"
echo "  Use SAGAN to fit the spectrum"