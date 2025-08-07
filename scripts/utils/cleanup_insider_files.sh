# Color definitions
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Removing insider files...${NC}"

# Remove any file in .github/workflows/ that does not end with .yaml
echo -e "${YELLOW}Cleaning .github/workflows/ directory...${NC}"
find .github/workflows/ -type f ! -name "*.yaml" -exec git rm --cached -r {} \;

# Remove VSCode configuration
echo -e "${YELLOW}Removing .vscode directory...${NC}"
git rm --cached -r .vscode

# Remove scripts
echo -e "${YELLOW}Removing scripts directory...${NC}"
git rm --cached -r scripts/dev
git rm --cached -r scripts/doc

# Remove Makefile
echo -e "${YELLOW}Removing Makefile...${NC}"
git rm --cached -r Makefile

# Add more files/directories to remove below as needed
# Example:
# echo "Removing development configs..."
# git rm --cached -r .devcontainer
# git rm --cached -r .gitpod.yml
# git rm --cached -r docs/_build/

echo -e "${GREEN}Insider files cleanup completed!${NC}"
