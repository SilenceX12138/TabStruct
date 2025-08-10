# Color definitions
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to prompt yes/no questions
prompt_yes_no() {
    local question="$1"
    local response
    while true; do
        echo -n -e "${BLUE}$question (y/n)${NC}: "
        read response
        case $response in
        [Yy]*) return 0 ;;
        [Nn]*) return 1 ;;
        *) echo -e "${RED}Please answer yes (y) or no (n)${NC}" ;;
        esac
    done
}

echo -e "${YELLOW}=== Pre-Release Checklist ===${NC}"
echo ""

# Check 1: pyproject.toml version
if ! prompt_yes_no "Have you updated the version in pyproject.toml?"; then
    echo -e "${RED}Please update the version in pyproject.toml before proceeding with the release.${NC}"
    exit 1
fi

# Check 2: tutorials
if ! prompt_yes_no "Have you updated the tutorials?"; then
    echo -e "${RED}Please update the tutorials before proceeding with the release.${NC}"
    exit 1
fi

# Check 3: README
if ! prompt_yes_no "Have you updated the README?"; then
    echo -e "${RED}Please update the README before proceeding with the release.${NC}"
    exit 1
fi

# Check 4: Branch
if ! prompt_yes_no "Have you updated the master branch?"; then
    echo -e "${RED}Please update the master branch before proceeding with the release.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All pre-release checks completed!${NC}"
echo ""

# Prompt user for version input
echo -n -e "${YELLOW}Enter the version for the release (e.g., 1.0.0)${NC}: "
read version

# Check if version was provided
if [ -z "$version" ]; then
    echo -e "${RED}Error: Version cannot be empty${NC}"
    exit 1
fi

release_tag="release-v$version"
release_branch="release-v$version-pack"

# Check if the release tag already exists
if git tag -l | grep -q "^$release_tag$"; then
    echo -e "${RED}Error: Release tag '$release_tag' already exists!${NC}"
    echo -e "${YELLOW}Please choose a different version number.${NC}"
    exit 1
fi

echo -e "${GREEN}Creating release for version: v$version${NC}"

git tag $release_tag HEAD

git checkout -b $release_branch public/master
git pull public master
git diff --binary $release_branch master >./release.patch
git apply ./release.patch --allow-empty
rm ./release.patch
git add .

# Remove insider files using separate script
bash ./scripts/utils/cleanup_insider_files.sh

git commit -m "release: v$version"
git push public $release_branch
git add .
git checkout master
git branch -D $release_branch

echo -e "${GREEN}Release v$version completed successfully!${NC}"
