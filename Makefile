# Author: Martin Zůbek
# Date: 2026-03-26
# Description: Makefile of this Bachelors thesis project.

# Directory with the source code of the project
SRC_DIR = src
# Directory with the code of the extractors
EXTERNAL_DIR = external
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
	@"$(ENV_DIR)/bin/pip" install -e .

	@if [ ! -d "$(EXTERNAL_DIR)" ]; then \
		echo "Downloading extractors."; \
		python3.12 "$(SCRIPTS_DIR)/download_extractors.py"; \
	else \
		echo "Extractors already exist."; \
	fi

	@if [ -f "$(EXTERNAL_DIR)/TruFor/requirements.txt" ]; then \
		echo "Installing TruFor dependencies."; \
		"$(ENV_DIR)/bin/pip" install --no-cache-dir -r "$(EXTERNAL_DIR)/TruFor/requirements.txt"; \
	fi

# Running the tests for the data processing
test_data_processing:
	@echo "Testing the data processing pipeline."
	@echo "Setting up environment for testing."

	@if [ ! -d "$(ENV_DIR)" ]; then \
		echo "Creating virtual environment."; \
		python3.12 -m venv "$(ENV_DIR)"; \
		"$(ENV_DIR)/bin/pip" install --upgrade pip; \
	else \
		echo "Virtual environment already exists."; \
	fi

	@echo "Installing project dependencies."
	@mkdir -p "$(META_DIR)"
	@"$(ENV_DIR)/bin/pip" install -e .

	@"$(ENV_DIR)/bin/python" -m pytest -v $(TESTS_DIR)/data_processing/unit
	@"$(ENV_DIR)/bin/python" -m pytest -v $(TESTS_DIR)/data_processing/integrations

# Ruff code quality check
ruff_check:
	@echo "Running ruff for code quality check."
	@echo "Setting up environment for ruff check."

	@if [ ! -d "$(ENV_DIR)" ]; then \
		echo "Creating virtual environment."; \
		python3.12 -m venv "$(ENV_DIR)"; \
		"$(ENV_DIR)/bin/pip" install --upgrade pip; \
	else \
		echo "Virtual environment already exists."; \
	fi

	@echo "Installing project dependencies."
	@mkdir -p "$(META_DIR)"
	@"$(ENV_DIR)/bin/pip" install -e .

	@echo "Running ruff for code quality check."
	@"$(ENV_DIR)/bin/ruff" check "$(SRC_DIR)" "$(TESTS_DIR)" "$(SCRIPTS_DIR)"

# Ruff code quality check with fixing issues
ruff_fix:
	@echo "Running ruff for fixing issues."
	@echo "Setting up environment for ruff fix."

	@if [ ! -d "$(ENV_DIR)" ]; then \
		echo "Creating virtual environment."; \
		python3.12 -m venv "$(ENV_DIR)"; \
		"$(ENV_DIR)/bin/pip" install --upgrade pip; \
	else \
		echo "Virtual environment already exists."; \
	fi

	@echo "Installing project dependencies."
	@mkdir -p "$(META_DIR)"
	@"$(ENV_DIR)/bin/pip" install -e .

	@echo "Running ruff for fixing issues."
	@"$(ENV_DIR)/bin/ruff" check --fix "$(SRC_DIR)" "$(SCRIPTS_DIR)" "$(TESTS_DIR)"

# Mypy type check
mypy_check:
	@$(MAKE) build
	@echo "Running mypy type check."
	@"$(ENV_DIR)/bin/mypy" "$(SRC_DIR)" "$(SCRIPTS_DIR)" "$(TESTS_DIR)"

# Zipping the project for submission
zip:
	@echo "Zipping the project for submission."
	@zip -r x253206.zip "$(SRC_DIR)" "$(SCRIPTS_DIR)" "$(TESTS_DIR)" Makefile pyproject.toml README.md

# Cleaning the project
clean:
	@echo "Cleaning the project."
	@rm -rf "$(META_DIR)"
	@rm -rf "$(ENV_DIR)"
	@rm -rf "$(EXTERNAL_DIR)"
	@rm -rf "$(PYTEST_CACHE_DIR)"
	@find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: build zip clean test_data_processing ruff_check ruff_fix mypy_check