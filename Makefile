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

# Zipping the Bachelors thesis for submission
zip:
	@echo "Zipping the project for submission."
	@zip -r submission.zip "$(SRC_DIR)" "$(SCRIPTS_DIR)" Makefile pyproject.toml README.md

# Cleaning the project
clean:
	@echo "Cleaning the project."
	@rm -rf "$(META_DIR)"
	@rm -rf "$(ENV_DIR)"
	@rm -rf "$(EXTERNAL_DIR)"

.PHONY: build zip clean 