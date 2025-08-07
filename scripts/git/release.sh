# Color definitions
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

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
git diff $release_branch master >./release.patch
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
