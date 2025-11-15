.PHONY: all dev test lint format type check clean build publish help

# Define variables
# point to the system-level python
PYTHON := python

# Default target: Run all checks
all: check

# Run development mode (e.g., Jupyter)
dev:
    @echo "Starting development environment (Jupyter Lab)..."
    @$(PYTHON) -m jupyter lab --port=8888 --no-browser --ip=0.0.0.0 --allow-root

# Run tests
test:
    @echo "Running tests..."
    @$(PYTHON) -m pytest

# Run linting and formatting (Ruff)
lint:
    @echo "Running Ruff linting..."
    @$(PYTHON) -m ruff check .
format:
    @echo "Running Ruff formatting..."
    @$(PYTHON) -m ruff format .

# Run static type checking (MyPy)
type:
    @echo "Running MyPy type checking..."
    @$(PYTHON) -m mypy src/ tests/ notebooks/ # Adjust paths as needed

# Run all checks (lint, format, type, test)
check: lint format type test

# Clean up build artifacts and temporary files
clean:
    @echo "Cleaning up..."
    @rm -rf build/ dist/ *.egg-info/
    @find . -type d -name "__pycache__" -exec rm -rf {} +
    @find . -type f -name "*.pyc" -delete
    @rm -rf .pytest_cache/
    @rm -rf .mypy_cache/
    @rm -f .coverage
    @echo "Cleanup complete."

# Build the package
build:
    @echo "Building package..."
    @$(PYTHON) -m build

# Publish to PyPI (requires setup and credentials)
publish: build
    @echo "Publishing package to PyPI..."
    @$(PYTHON) -m twine upload dist/*

# Help message
help:
    @echo "Available commands:"
    @echo "  make dev     - Start Jupyter Lab for development."
    @echo "  make test    - Run all project tests."
    @echo "  make lint    - Run Ruff linter."
    @echo "  make format  - Run Ruff formatter."
    @echo "  make type    - Run MyPy static type checker."
    @echo "  make check   - Run lint, format, type, and tests."
    @echo "  make clean   - Clean up build artifacts and caches."
    @echo "  make build   - Build the distributable package."
    @echo "  make publish - Publish the package to PyPI (requires credentials)."
    @echo "  make help    - Display this help message."
