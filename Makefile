# Author: Martin Zůbek
# Date: 2026-03-26
# Description: Makefile of this Bachelors thesis project.

# Directory with the source code of the project
SRC_DIR = src
# Directory with the code of the extractors
EXTERNAL_DIR = external
# Directory for the external code of TruFor
TRUEFOR_DIR = $(EXTERNAL_DIR)/TruFor
# Directory for the external code of IML-ViT
IML_VIT_DIR = $(EXTERNAL_DIR)/IML_VIT
# Directory for the external code of MMFusion
MMFUSION_DIR = $(EXTERNAL_DIR)/MMFusion
# Directory for the external code of SwinT
SWIN_DIR = $(EXTERNAL_DIR)/SwinT
# Directory for the scripts for setup of the project for training and evaluation and for downloading the extractors
SCRIPTS_DIR = scripts
# Directory of the environment of the project
ENV_DIR = .venv
# Directory fot the meta data of the project
META_DIR = meta_data
# Directory for the tests of the project
TESTS_DIR = tests
# Directory for the cache of the pytest
PYTEST_CACHE_DIR = .pytest_cache
# Directory for the cache of the mypy
MYPY_CACHE_DIR = .mypy_cache
# Directory for the cache of the ruff
RUFF_CACHE_DIR = .ruff_cache
# Directory with checkpoint for the application
CHECKPOINT_DIR = checkpoint
# Makefile file
MAKEFILE = Makefile
# README file
README = README.md
# License file
LICENSE = LICENSE
# Pyproject file
PYPROJECT = pyproject.toml
# Pip executable for the project
PIP = $(ENV_DIR)/bin/pip
# Python executable for the project
PYTHON = $(ENV_DIR)/bin/python3.12
# Ruff executable for the project
RUFF = $(ENV_DIR)/bin/ruff
# Mypy executable for the project
MYPY = $(ENV_DIR)/bin/mypy

# Build of the project
build:
	@echo "Setting up extractors and environment of the project."

	@if [ ! -d "$(ENV_DIR)" ]; then \
		echo "Creating virtual environment."; \
		python3.12 -m venv "$(ENV_DIR)"; \
		"$(ENV_DIR)/bin/pip" install --upgrade pip; \
	else \
		echo "Virtual environment already exists."; \
	fi

	@echo "Installing project dependencies."
	@mkdir -p "$(META_DIR)"
	@"$(PIP)" install -e .[dev]

	@if [ ! -d "$(EXTERNAL_DIR)" ]; then \
		echo "Downloading extractors."; \
		"$(PYTHON)" "$(SCRIPTS_DIR)/download_extractors.py"; \
	else \
		echo "Extractors already exist."; \
	fi

	@if [ -f "$(TRUEFOR_DIR)/requirements.txt" ]; then \
		echo "Installing TruFor dependencies."; \
		"$(PIP)" install --no-cache-dir -r "$(TRUEFOR_DIR)/requirements.txt"; \
	fi

	@if [ -f "$(IML_VIT_DIR)/requirements.txt" ]; then \
		echo "Installing IML-ViT dependencies."; \
		"$(PIP)" install --no-cache-dir -r "$(IML_VIT_DIR)/requirements.txt"; \
	fi

	@if [ -f "$(MMFUSION_DIR)/requirements.txt" ]; then \
		echo "Installing MMFusion dependencies."; \
		"$(PIP)" install --no-cache-dir -r "$(MMFUSION_DIR)/requirements.txt"; \
	fi

# Build of the application for running the GUI
build_app:
	@echo "Setting up application environment."

	@if [ ! -d "$(ENV_DIR)" ]; then \
		echo "Creating virtual environment."; \
		python3.12 -m venv "$(ENV_DIR)"; \
		"$(PIP)" install --upgrade pip; \
	else \
		echo "Virtual environment already exists."; \
	fi

	@echo "Installing application dependencies."
	@mkdir -p "$(META_DIR)"
	@"$(PIP)" install -e .
	
	
	@echo "Setting up extractor for the application."
	@PYTHONPATH="$(PWD)" "$(PYTHON)" "$(SCRIPTS_DIR)/build_app.py"
	@if [ -f "$(MMFUSION_DIR)/requirements.txt" ]; then \
		echo "Installing MMFusion dependencies."; \
		"$(PIP)" install --no-cache-dir -r "$(MMFUSION_DIR)/requirements.txt"; \
	fi

	@echo "Build complete. Run make run_app to start the application."

# Run the application
run_app:
	@echo "Running the application."
	@PYTHONPATH="$(PWD)" "$(PYTHON)" "$(SCRIPTS_DIR)/run_app.py"

# Running all the tests of the project
test_all:
	@$(MAKE) build

	@echo "Running all the tests of the project."
	@"$(PYTHON)" -m pytest -v $(TESTS_DIR)

# Running the tests for the data processing
test_data_processing:
	@echo "Testing the data processing pipeline."
	@echo "Setting up environment for testing."

	@if [ ! -d "$(ENV_DIR)" ]; then \
		echo "Creating virtual environment."; \
		python3.12 -m venv "$(ENV_DIR)"; \
		"$(PIP)" install --upgrade pip; \
	else \
		echo "Virtual environment already exists."; \
	fi

	@echo "Installing project dependencies."
	@mkdir -p "$(META_DIR)"
	@"$(PIP)" install -e .

	@"$(PYTHON)" -m pytest -v $(TESTS_DIR)/data_processing/unit
	@"$(PYTHON)" -m pytest -v $(TESTS_DIR)/data_processing/integrations

# Running the tests for the extractors
test_extractors:
	@echo "Testing the extractors."
	@echo "Setting up environment for testing."
	@$(MAKE) build
	@"$(PYTHON)" -m pytest -v $(TESTS_DIR)/extractors/

# Running the tests for the heads
test_heads:
	@echo "Testing the heads."
	@echo "Setting up environment for testing."
	@$(MAKE) build
	@"$(PYTHON)" -m pytest -v $(TESTS_DIR)/heads/

# Running the tests for the training
test_training:
	@echo "Testing the training."
	@echo "Setting up environment for testing."
	@$(MAKE) build
	@"$(PYTHON)" -m pytest -v $(TESTS_DIR)/training/

# Ruff code quality check
ruff_check:
	@echo "Running ruff for code quality check."
	@echo "Setting up environment for ruff check."

	@if [ ! -d "$(ENV_DIR)" ]; then \
		echo "Creating virtual environment."; \
		python3.12 -m venv "$(ENV_DIR)"; \
		"$(PIP)" install --upgrade pip; \
	else \
		echo "Virtual environment already exists."; \
	fi

	@echo "Installing project dependencies."
	@mkdir -p "$(META_DIR)"
	@"$(PIP)" install -e .

	@echo "Running ruff for code quality check."
	@"$(RUFF)" check "$(SRC_DIR)" "$(TESTS_DIR)" "$(SCRIPTS_DIR)"

# Ruff code quality check with fixing issues
ruff_fix:
	@echo "Running ruff for fixing issues."
	@echo "Setting up environment for ruff fix."

	@if [ ! -d "$(ENV_DIR)" ]; then \
		echo "Creating virtual environment."; \
		python3.12 -m venv "$(ENV_DIR)"; \
		"$(PIP)" install --upgrade pip; \
	else \
		echo "Virtual environment already exists."; \
	fi

	@echo "Installing project dependencies."
	@mkdir -p "$(META_DIR)"
	@"$(PIP)" install -e .

	@echo "Running ruff for fixing issues."
	@"$(RUFF)" check --fix "$(SRC_DIR)" "$(SCRIPTS_DIR)" "$(TESTS_DIR)"

# Mypy type check
mypy_check:
	@$(MAKE) build
	@echo "Running mypy type check."
	@"$(MYPY)" "$(SRC_DIR)" "$(SCRIPTS_DIR)" "$(TESTS_DIR)"

# Zipping the project for submission
zip:
	@echo "Zipping the project for submission."
	@zip -r x253206.zip "$(SRC_DIR)" "$(SCRIPTS_DIR)" "$(TESTS_DIR)" "$(CHECKPOINT_DIR)" "$(MAKEFILE)" "$(PYPROJECT)" "$(README)" "$(LICENSE)"

# Cleaning the project
clean:
	@echo "Cleaning the project."
	@rm -rf "$(META_DIR)"
	@rm -rf "$(ENV_DIR)"
	@rm -rf "$(EXTERNAL_DIR)"
	@rm -rf "$(PYTEST_CACHE_DIR)"
	@rm -rf "$(MYPY_CACHE_DIR)"
	@rm -rf "$(RUFF_CACHE_DIR)"
	@find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: build build_app zip clean test_data_processing test_all test_extractors test_heads test_training ruff_check ruff_fix mypy_check