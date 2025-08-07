#!/bin/bash

# ===================================================================
# Tetra Project Initialization Script
#
# This script initializes a new Tetra project by:
# 1. Collecting project configuration (including target directory)
# 2. Setting up git repositories (insider and public)
# 3. Updating configuration files with project details
# 4. Creating initial commits and branches
#
# Usage:
#   ./init.sh
#
# The script will prompt for:
#   - Target directory (defaults to current directory)
#   - Project name (defaults to "Project Name")
#   - Insider git repository URL
#   - Public git repository URL
# ===================================================================

set -e # Exit on any error

# Color definitions for better output formatting
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ===================================================================
# UTILITY FUNCTIONS
# ===================================================================

# Function to print colored status messages
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1" >&2
}

# Function to prompt for input with a default value
prompt_input() {
    local prompt="$1"
    local default="$2"
    local result

    if [ -n "$default" ]; then
        echo -n -e "${BLUE}$prompt${NC} ${GREEN}[$default]${NC}: " >&2
        read result
        echo "${result:-$default}"
    else
        echo -n -e "${BLUE}$prompt${NC}: " >&2
        read result
        echo "$result"
    fi
}

# Function to prompt for required input (no default)
prompt_required() {
    local prompt="$1"
    local result

    while [ -z "$result" ]; do
        echo -n -e "${YELLOW}$prompt${NC}: " >&2
        read result
        if [ -z "$result" ]; then
            print_error "This field is required. Please provide a value."
        fi
    done
    echo "$result"
}

# Cross-platform sed function
sed_update() {
    local file="$1"
    local pattern="$2"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "$pattern" "$file"
    else
        sed -i "$pattern" "$file"
    fi
}

# Safe sed update function that handles URLs with forward slashes
sed_update_safe() {
    local file="$1"
    local search_pattern="$2"
    local replacement="$3"

    if [[ "$OSTYPE" == "darwin"* ]]; then
        sed -i '' "s|${search_pattern}|${replacement}|g" "$file"
    else
        sed -i "s|${search_pattern}|${replacement}|g" "$file"
    fi
}

# Function to validate and create directory
validate_directory() {
    local dir_path="$1"

    # Check if we can create/access the directory
    if ! mkdir -p "$dir_path" 2>/dev/null; then
        print_error "Cannot create or access directory: $dir_path"
        print_error "Please check permissions and try again."
        exit 1
    fi

    # Check if we can write to the directory
    if [ ! -w "$dir_path" ]; then
        print_error "Cannot write to directory: $dir_path"
        print_error "Please check permissions and try again."
        exit 1
    fi

    return 0
}

# ===================================================================
# PROJECT CONFIGURATION
# ===================================================================

collect_project_info() {
    print_info "Collecting project information..."

    # Get target directory for initialization
    local current_dir=$(pwd)
    TARGET_DIR=$(prompt_input "Enter directory to initialize project" "$current_dir")

    # Expand tilde and resolve relative paths
    TARGET_DIR=$(eval echo "$TARGET_DIR")

    # Validate and create directory if needed
    validate_directory "$TARGET_DIR"

    # Get absolute path
    TARGET_DIR=$(cd "$TARGET_DIR" && pwd)

    # Get project name
    PROJECT_NAME=$(prompt_input "Enter project name" "Project Name")

    # Display configuration summary first
    echo
    print_info "Project configuration:"
    echo -e "${GREEN}Target Directory:${NC} $TARGET_DIR"
    echo -e "${GREEN}Project Name:${NC} $PROJECT_NAME"

    # Change to target directory
    cd "$TARGET_DIR"
    print_status "Changed to target directory: $TARGET_DIR"

    # Check if insider remote already exists
    if [ -d .git ] && git remote get-url origin >/dev/null 2>&1; then
        INSIDER_GIT=$(git remote get-url origin)
        print_status "Found existing insider remote: $INSIDER_GIT"
        print_warning "Reusing existing insider repository configuration."
    else
        INSIDER_GIT=$(prompt_required "Enter insider git repository URL")
    fi

    # Get public repository URL
    PUBLIC_GIT=$(prompt_required "Enter public git repository URL")

    # Display final configuration summary
    echo
    print_info "Final project configuration:"
    echo -e "${GREEN}Target Directory:${NC} $TARGET_DIR"
    echo -e "${GREEN}Project Name:${NC} $PROJECT_NAME"
    echo -e "${GREEN}Insider Git:${NC} $INSIDER_GIT"
    echo -e "${GREEN}Public Git:${NC} $PUBLIC_GIT"
    echo
}

# ===================================================================
# GIT REPOSITORY SETUP
# ===================================================================

setup_git_repository() {
    print_info "Setting up git repository..."

    # Initialize git repository if not already present
    if [ ! -d .git ]; then
        git init
        git remote add origin "$INSIDER_GIT"
        print_status "Initialized git repository with insider remote"
    else
        print_status "Git repository already exists"
    fi
}

clone_template_if_needed() {
    print_info "Checking if template needs to be cloned..."

    # If current folder is empty, pull from template remote
    if [ -z "$(ls -A . 2>/dev/null)" ]; then
        print_info "Current folder is empty. Cloning Tetra-insider template..."
        git clone https://github.com/SilenceX12138/Tetra-insider.git
        rm -rf Tetra-insider/.git
        mv Tetra-insider/* . 2>/dev/null || true
        mv Tetra-insider/.* . 2>/dev/null || true
        rm -rf Tetra-insider
        print_status "Cloned Tetra-insider template into current folder"
    else
        print_status "Current folder is not empty. Skipping template clone"
    fi
}

# ===================================================================
# CONFIGURATION FILE UPDATES
# ===================================================================

update_pyproject_toml() {
    if [ ! -f "pyproject.toml" ]; then
        print_warning "pyproject.toml not found. Skipping update."
        return
    fi

    print_info "Updating pyproject.toml..."

    # Convert project name to lowercase and replace spaces with hyphens for package name
    local package_name=$(echo "$PROJECT_NAME" | tr '[:upper:]' '[:lower:]' | sed 's/ /-/g')

    sed_update "pyproject.toml" "s/^name = .*/name = \"$package_name\"/"
    print_status "Updated pyproject.toml with package name: $package_name"
}

update_sphinx_config() {
    if [ ! -f "docs/wiki/conf.py" ]; then
        print_warning "docs/wiki/conf.py not found. Skipping update."
        return
    fi

    print_info "Updating Sphinx configuration..."

    # Update project name
    sed_update "docs/wiki/conf.py" "s/project = .*/project = \"$PROJECT_NAME\"/"

    # Update HTML title
    sed_update "docs/wiki/conf.py" "s/html_title = .*/html_title = \"$PROJECT_NAME Documentation\"/"

    # Update repository URL using safe sed function
    sed_update_safe "docs/wiki/conf.py" "\"repository_url\": .*," "\"repository_url\": \"$PUBLIC_GIT\","

    print_status "Updated docs/wiki/conf.py with project configuration"
}

update_sphinx_index() {
    if [ ! -f "docs/wiki/index.rst" ]; then
        print_warning "docs/wiki/index.rst not found. Skipping update."
        return
    fi

    print_info "Updating Sphinx index..."

    # Create new index.rst with project name
    cat >docs/wiki/index.rst <<EOF
$PROJECT_NAME Documentation
$(echo "$PROJECT_NAME Documentation" | sed 's/./=/g')
EOF

    print_status "Updated docs/wiki/index.rst with project name"
}

update_manifest() {
    if [ ! -f "MANIFEST.in" ]; then
        print_warning "MANIFEST.in not found. Skipping update."
        return
    fi

    print_info "Updating MANIFEST.in..."

    sed_update "MANIFEST.in" "s/Tetra/$PROJECT_NAME/g"
    print_status "Updated MANIFEST.in with project name"
}

update_readme() {
    print_info "Creating README.md..."
    echo "# $PROJECT_NAME" >README.md
    print_status "Created README.md with project name"
}

update_all_config_files() {
    print_info "Updating configuration files..."

    update_pyproject_toml
    update_sphinx_config
    update_sphinx_index
    update_manifest
    update_readme

    print_status "All configuration files updated"
}

# ===================================================================
# CLEANUP AND FINALIZATION
# ===================================================================

cleanup_insider_files() {
    print_info "Cleaning up insider-specific files..."

    # Make cleanup script executable and run it
    if [ -f "./scripts/utils/cleanup_insider_files.sh" ]; then
        chmod +x ./scripts/utils/*.sh
        bash ./scripts/utils/cleanup_insider_files.sh
        print_status "Cleaned up insider-specific files"
    else
        print_warning "Cleanup script not found. Skipping insider file cleanup."
    fi
}

create_initial_commits() {
    print_info "Creating initial commits and branches..."

    # Stage and commit configuration changes
    git add .
    # Clean up any existing insider files
    cleanup_insider_files
    git commit -m "Update project name and configuration files"
    print_status "Created initial commit with configuration updates"

    # Create and manage release tag
    git tag -d release-v0 2>/dev/null || true
    git tag release-v0 HEAD
    print_status "Created release-v0 tag"

    # Add public remote
    if ! git remote get-url public >/dev/null 2>&1; then
        git remote add public "$PUBLIC_GIT"
        print_status "Added public remote"
    else
        print_status "Public remote already exists"
    fi

    print_info "Current remotes:"
    git remote -v
}

create_public_branch() {
    print_info "Creating and pushing public branch..."

    # Create temporary branch for public release
    git checkout -b release-v0-pack

    # Push to public repository
    git push public release-v0-pack:master
    print_status "Pushed initial version to public repository"

    # Return to master and cleanup
    git checkout master
    git branch -D release-v0-pack
    print_status "Cleaned up temporary branch"
}

# ===================================================================
# MAIN EXECUTION
# ===================================================================

main() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}    Tetra Project Initialization${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo

    # Step 1: Collect project information and set working directory
    collect_project_info

    # Ensure we're in the correct directory for all subsequent operations
    if [ ! -d "$TARGET_DIR" ] || [ "$(pwd)" != "$TARGET_DIR" ]; then
        print_error "Failed to change to target directory: $TARGET_DIR"
        exit 1
    fi

    # Step 2: Clone template if needed
    clone_template_if_needed

    # Step 3: Setup git repository
    setup_git_repository

    # Step 4: Update configuration files
    update_all_config_files

    # Step 5: Create commits and tags
    create_initial_commits

    # Step 6: Create and push public branch
    create_public_branch

    # Step 7: Track the insider files in master
    git add .
    git commit -m "Track insider files in insider repository"
    git push origin master

    echo
    print_status "Project initialization completed successfully!"
    echo -e "${GREEN}Your project '$PROJECT_NAME' is ready at: $TARGET_DIR${NC}"
    echo
}

# Run main function
main
