# Python Code Style Check Script
# Converted from GitHub Actions workflow: .github/workflows/style_check.yaml

set -e # Exit on any error

# Environment variables (can be overridden)
IGNORE_PATHS_PYLINT=""
IGNORE_PATHS_FLAKE8=""

echo "=== Python Code Style Check ==="
echo "Starting style check for Python code..."

# Check if we're in the project root (look for pyproject.toml or setup.py)
if [[ ! -f "pyproject.toml" && ! -f "setup.py" ]]; then
    echo "Error: Please run this script from the project root directory"
    exit 1
fi

# Check if src/ directory exists
if [[ ! -d "src" ]]; then
    echo "Error: src/ directory not found"
    exit 1
fi

# Run Pylint
echo "Running Pylint..."
echo "----------------------------------------"

pylint src/ \
    --errors-only \
    --disable=E0401,E1101,E0602,E1102 \
    --extension-pkg-whitelist=PyQt5 \
    --generated-members='numpy.*,torch.*,cv2.*' \
    --ignore-paths="$IGNORE_PATHS_PYLINT" \
    --recursive yes

echo "Pylint check completed successfully!"
echo ""

# Run Flake8
echo "Running Flake8..."
echo "----------------------------------------"

flake8 src/ \
    --ignore=E501,F401,E402,E221,E203,E251,E128,E502,E266,E271,E111,E117,W504,W503 \
    --exclude="$IGNORE_PATHS_FLAKE8"

echo "Flake8 check completed successfully!"
echo ""
echo "=== All style checks passed! ==="
