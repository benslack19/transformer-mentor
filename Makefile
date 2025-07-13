.PHONY: all setup install dev test lint format type check clean build publish help

# Define variables
VENV_DIR := .venv
PYTHON := $(VENV_DIR)/bin/python
# HATCH := $(VENV_DIR)/bin/hatch # We won't rely on Hatch being in the venv for initial setup

# Default target
all: setup dev

# Setup the virtual environment and install dependencies
setup:
	@echo "Setting up virtual environment and installing dependencies..."
	# --- FIX STARTS HERE ---
	# Ensure the parent directory exists for the virtual environment
	@mkdir -p $(VENV_DIR)
	# Create the virtual environment using the system's 'python3' command first
	@python3 -m venv $(VENV_DIR)
	# --- FIX ENDS HERE ---
	@echo "Upgrading pip and installing project dependencies..."
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -e .[dev]
	@echo "Virtual environment setup complete. Installing pre-commit hooks..."
	@$(PYTHON) -m pre_commit install
	@echo "Pre-commit hooks installed."

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
	@rm -rf $(VENV_DIR) # Remove virtual environment
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
	@echo "  make setup   - Set up the virtual environment and install dependencies (run once initially)."
	@echo "  make dev     - Start Jupyter Lab for development."
	@echo "  make test    - Run all project tests."
	@echo "  make lint    - Run Ruff linter."
	@echo "  make format  - Run Ruff formatter."
	@echo "  make type    - Run MyPy static type checker."
	@echo "  make check   - Run lint, format, type, and tests."
	@echo "  make clean   - Clean up build artifacts, caches, and virtual environment."
	@echo "  make build   - Build the distributable package."
	@echo "  make publish - Publish the package to PyPI (requires credentials)."
	@echo "  make help    - Display this help message."