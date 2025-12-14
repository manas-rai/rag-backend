.PHONY: help setup venv env install install-all clean run check-python setup-pyenv fix-pyenv

# Python command (will be auto-detected at runtime, defaults to python3)
PYTHON ?= python
VENV := .venv
VENV_BIN := $(VENV)/bin
PIP := $(VENV_BIN)/pip
PYTHON_VENV := $(VENV_BIN)/python

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Available targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-15s$(NC) %s\n", $$1, $$2}'

check-python: ## Check Python installation and version
	@echo "$(GREEN)Checking Python installation...$(NC)"
	@PYTHON_CMD=$$(command -v python3 2> /dev/null || command -v python 2> /dev/null || echo ""); \
	if [ -z "$$PYTHON_CMD" ]; then \
		echo "$(RED)✗ No Python found in PATH$(NC)"; \
		echo "$(YELLOW)If using pyenv, try:$(NC)"; \
		echo "  pyenv install 3.11.0"; \
		echo "  pyenv local 3.11.0"; \
		echo "  pyenv rehash"; \
		exit 1; \
	else \
		echo "$(GREEN)✓ Found Python: $(YELLOW)$$PYTHON_CMD$(NC)"; \
		$$PYTHON_CMD --version; \
		if $$PYTHON_CMD -m venv --help > /dev/null 2>&1; then \
			echo "$(GREEN)✓ venv module is available$(NC)"; \
		else \
			echo "$(RED)✗ venv module is not available$(NC)"; \
			exit 1; \
		fi; \
	fi

setup-pyenv: ## Diagnose and help fix pyenv setup
	@echo "$(GREEN)Diagnosing pyenv setup...$(NC)"
	@echo ""
	@if ! command -v pyenv > /dev/null 2>&1; then \
		echo "$(RED)✗ pyenv is not in PATH$(NC)"; \
		echo "$(YELLOW)Make sure pyenv is initialized in your shell. Add to ~/.zshrc:$(NC)"; \
		echo "  export PYENV_ROOT=\"\$$HOME/.pyenv\""; \
		echo "  export PATH=\"\$$PYENV_ROOT/bin:\$$PATH\""; \
		echo "  eval \"\$$(pyenv init - zsh)\""; \
		echo ""; \
		echo "$(YELLOW)Then run: source ~/.zshrc$(NC)"; \
		exit 1; \
	else \
		echo "$(GREEN)✓ pyenv is installed$(NC)"; \
		pyenv --version; \
	fi
	@echo ""
	@echo "$(GREEN)Checking installed Python versions...$(NC)"
	@if pyenv versions --bare 2>/dev/null | grep -q .; then \
		echo "$(GREEN)Installed versions:$(NC)"; \
		pyenv versions; \
	else \
		echo "$(YELLOW)No Python versions installed via pyenv$(NC)"; \
		echo "$(YELLOW)Install a version with: pyenv install 3.11.0$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Checking current Python version...$(NC)"
	@CURRENT_VERSION=$$(pyenv version-name 2>/dev/null || echo "system"); \
	echo "Current version: $(YELLOW)$$CURRENT_VERSION$(NC)"; \
	@echo ""
	@echo "$(GREEN)Checking python3 availability...$(NC)"
	@if command -v python3 > /dev/null 2>&1; then \
		echo "$(GREEN)✓ python3 is available: $(YELLOW)$$(command -v python3)$(NC)"; \
		python3 --version; \
	else \
		echo "$(RED)✗ python3 is not available$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Checking python availability...$(NC)"
	@if command -v python > /dev/null 2>&1; then \
		echo "$(GREEN)✓ python is available: $(YELLOW)$$(command -v python)$(NC)"; \
		python --version; \
	else \
		echo "$(YELLOW)⚠ python is not available (this is OK, python3 works)$(NC)"; \
	fi
	@echo ""
	@echo "$(YELLOW)Recommendations:$(NC)"
	@if ! pyenv versions --bare 2>/dev/null | grep -q .; then \
		echo "  1. Install a Python version: $(YELLOW)pyenv install 3.11.0$(NC)"; \
	fi
	@CURRENT_GLOBAL=$$(pyenv global 2>/dev/null || echo ""); \
	if [ -z "$$CURRENT_GLOBAL" ] || [ "$$CURRENT_GLOBAL" = "system" ]; then \
		INSTALLED=$$(pyenv versions --bare 2>/dev/null | head -1); \
		if [ -n "$$INSTALLED" ]; then \
			echo "  2. Set global version: $(YELLOW)pyenv global $$INSTALLED$(NC)"; \
		fi; \
	fi
	@echo "  3. Set local version for this project: $(YELLOW)pyenv local <version>$(NC)"
	@echo "  4. Update shims: $(YELLOW)pyenv rehash$(NC)"
	@echo ""
	@echo "$(GREEN)Note: The Makefile uses python3, so you don't need 'python' command.$(NC)"

fix-pyenv: ## Attempt to fix common pyenv issues
	@echo "$(GREEN)Attempting to fix pyenv setup...$(NC)"
	@echo ""
	@if ! command -v pyenv > /dev/null 2>&1; then \
		echo "$(RED)Error: pyenv is not in PATH$(NC)"; \
		echo "$(YELLOW)Please initialize pyenv in your shell first.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)1. Checking for installed Python versions...$(NC)"
	@INSTALLED_VERSIONS=$$(pyenv versions --bare 2>/dev/null | grep -v system || echo ""); \
	if [ -z "$$INSTALLED_VERSIONS" ]; then \
		echo "$(YELLOW)No Python versions installed. Installing 3.11.0...$(NC)"; \
		pyenv install 3.11.0 || { \
			echo "$(RED)Failed to install Python 3.11.0$(NC)"; \
			echo "$(YELLOW)You may need to install dependencies first.$(NC)"; \
			exit 1; \
		}; \
		INSTALLED_VERSIONS="3.11.0"; \
	else \
		echo "$(GREEN)✓ Found installed versions$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)2. Setting global Python version...$(NC)"
	@LATEST_VERSION=$$(pyenv versions --bare 2>/dev/null | grep -v system | head -1); \
	if [ -n "$$LATEST_VERSION" ]; then \
		echo "$(YELLOW)Setting global version to: $$LATEST_VERSION$(NC)"; \
		pyenv global $$LATEST_VERSION || echo "$(YELLOW)Warning: Could not set global version$(NC)"; \
		echo "$(GREEN)✓ Global version set$(NC)"; \
	else \
		echo "$(YELLOW)No versions available to set as global$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)3. Setting local Python version for this project...$(NC)"
	@LATEST_VERSION=$$(pyenv versions --bare 2>/dev/null | grep -v system | head -1); \
	if [ -n "$$LATEST_VERSION" ]; then \
		echo "$(YELLOW)Setting local version to: $$LATEST_VERSION$(NC)"; \
		pyenv local $$LATEST_VERSION || echo "$(YELLOW)Warning: Could not set local version$(NC)"; \
		echo "$(GREEN)✓ Local version set$(NC)"; \
	else \
		echo "$(YELLOW)No versions available to set as local$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)4. Updating pyenv shims...$(NC)"
	@pyenv rehash
	@echo "$(GREEN)✓ Shims updated$(NC)"
	@echo ""
	@echo "$(GREEN)5. Verifying python3 is available...$(NC)"
	@if command -v python3 > /dev/null 2>&1; then \
		echo "$(GREEN)✓ python3 is available: $(YELLOW)$$(command -v python3)$(NC)"; \
		python3 --version; \
		echo ""
		echo "$(GREEN)✓ pyenv setup complete!$(NC)"; \
		echo "$(YELLOW)You can now run: make setup$(NC)"; \
	else \
		echo "$(RED)✗ python3 is still not available$(NC)"; \
		echo "$(YELLOW)Try restarting your terminal or running: source ~/.zshrc$(NC)"; \
		exit 1; \
	fi

setup: venv env install-all ## Complete setup: create venv, .env file, and install all dependencies
	@echo "$(GREEN)✓ Setup complete!$(NC)"
	@echo "$(YELLOW)Next steps:$(NC)"
	@echo "  1. Edit .env file with your API keys and configuration"
	@echo "  2. Activate virtual environment: source $(VENV)/bin/activate"
	@echo "  3. Run the application: make run"

venv: ## Create Python virtual environment
	@if [ -d "$(VENV)" ]; then \
		echo "$(YELLOW)Virtual environment already exists at $(VENV)$(NC)"; \
	else \
		echo "$(GREEN)Creating virtual environment...$(NC)"; \
		PYTHON_CMD=$$(command -v python3 2> /dev/null || command -v python 2> /dev/null || echo ""); \
		if [ -z "$$PYTHON_CMD" ]; then \
			echo "$(RED)Error: Python not found. Please install Python 3 or ensure it's in your PATH.$(NC)"; \
			echo "$(YELLOW)If using pyenv, try:$(NC)"; \
			echo "  pyenv install 3.11.0"; \
			echo "  pyenv local 3.11.0"; \
			echo "  pyenv rehash"; \
			exit 1; \
		fi; \
		if ! $$PYTHON_CMD -m venv --help > /dev/null 2>&1; then \
			echo "$(RED)Error: $$PYTHON_CMD does not support venv module.$(NC)"; \
			exit 1; \
		fi; \
		$$PYTHON_CMD -m venv $(VENV); \
		echo "$(GREEN)✓ Virtual environment created using $$PYTHON_CMD$(NC)"; \
	fi

env: ## Create .env file with default configuration
	@if [ -f ".env" ]; then \
		echo "$(YELLOW).env file already exists. Skipping creation.$(NC)"; \
		echo "$(YELLOW)If you want to recreate it, delete .env and run 'make env' again.$(NC)"; \
	else \
		echo "$(GREEN)Creating .env file...$(NC)"; \
		echo "# RAG Backend Configuration" > .env; \
		echo "# Generated by Makefile" >> .env; \
		echo "" >> .env; \
		echo "# LLM Provider Settings" >> .env; \
		echo "LLM_PROVIDER=groq" >> .env; \
		echo "EMBEDDING_PROVIDER=sentence_transformers" >> .env; \
		echo "" >> .env; \
		echo "# Vector Store Settings" >> .env; \
		echo "VECTOR_PROVIDER=pinecone" >> .env; \
		echo "VECTOR_STORE_PATH=./data/vector_store" >> .env; \
		echo "VECTOR_STORE_DIMENSION=768" >> .env; \
		echo "VECTOR_STORE_METRIC=cosine" >> .env; \
		echo "" >> .env; \
		echo "# Text Processing Settings" >> .env; \
		echo "TEXT_SPLITTER=langchain" >> .env; \
		echo "CHUNK_SIZE=1000" >> .env; \
		echo "CHUNK_OVERLAP=200" >> .env; \
		echo "" >> .env; \
		echo "# Azure Settings (Optional)" >> .env; \
		echo "# AZURE_API_KEY=your_azure_api_key" >> .env; \
		echo "# AZURE_API_BASE=your_azure_api_base" >> .env; \
		echo "AZURE_API_VERSION=2024-02-15-preview" >> .env; \
		echo "AZURE_DEPLOYMENT_NAME=gpt-35-turbo" >> .env; \
		echo "AZURE_EMBEDDING_DEPLOYMENT_NAME=text-embedding-ada-002" >> .env; \
		echo "" >> .env; \
		echo "# OpenAI Settings (Optional)" >> .env; \
		echo "# OPENAI_API_KEY=your_openai_api_key" >> .env; \
		echo "OPENAI_MODEL=gpt-3.5-turbo" >> .env; \
		echo "OPENAI_EMBEDDING_MODEL=text-embedding-ada-002" >> .env; \
		echo "" >> .env; \
		echo "# GCP Settings (Optional)" >> .env; \
		echo "# GCP_PROJECT_ID=your_gcp_project_id" >> .env; \
		echo "# GCP_LOCATION=your_gcp_location" >> .env; \
		echo "GCP_MODEL=gemini-pro" >> .env; \
		echo "GCP_EMBEDDING_MODEL=textembedding-gecko" >> .env; \
		echo "" >> .env; \
		echo "# AWS Settings (Optional)" >> .env; \
		echo "# AWS_ACCESS_KEY_ID=your_aws_access_key_id" >> .env; \
		echo "# AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key" >> .env; \
		echo "# AWS_REGION=your_aws_region" >> .env; \
		echo "AWS_MODEL=anthropic.claude-3-sonnet-20240229-v1:0" >> .env; \
		echo "AWS_EMBEDDING_MODEL=amazon.titan-embed-text-v1" >> .env; \
		echo "" >> .env; \
		echo "# Groq Settings" >> .env; \
		echo "GROQ_API_KEY=your_groq_api_key" >> .env; \
		echo "GROQ_MODEL=llama3-8b-8192" >> .env; \
		echo "GROQ_EMBEDDING_MODEL=llama3-8b-8192" >> .env; \
		echo "" >> .env; \
		echo "# Sentence Transformers Settings" >> .env; \
		echo "SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2" >> .env; \
		echo "SENTENCE_TRANSFORMER_DEVICE=cpu" >> .env; \
		echo "" >> .env; \
		echo "# Pinecone Settings (Optional)" >> .env; \
		echo "# PINECONE_API_KEY=your_pinecone_api_key" >> .env; \
		echo "PINECONE_INDEX_NAME=ragindex1" >> .env; \
		echo "PINECONE_ENVIRONMENT=gcp-starter" >> .env; \
		echo "$(GREEN)✓ .env file created$(NC)"; \
		echo "$(YELLOW)Please edit .env file and add your API keys!$(NC)"; \
	fi

install: venv ## Install base dependencies only
	@echo "$(GREEN)Installing base dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements-base.txt
	@echo "$(GREEN)✓ Base dependencies installed$(NC)"

install-all: venv ## Install all dependencies (base + ML + providers)
	@echo "$(GREEN)Installing all dependencies...$(NC)"
	@$(PIP) install --upgrade pip
	@echo "$(GREEN)Installing base dependencies...$(NC)"
	@$(PIP) install -r requirements-base.txt
	@echo "$(GREEN)Installing ML dependencies...$(NC)"
	@$(PIP) install -r requirements-ml.txt
	@echo "$(GREEN)Installing provider dependencies...$(NC)"
	@$(PIP) install -r requirements-providers.txt
	@echo "$(GREEN)✓ All dependencies installed$(NC)"

clean: ## Remove virtual environment and .env file
	@echo "$(YELLOW)Cleaning up...$(NC)"
	@if [ -d "$(VENV)" ]; then \
		rm -rf $(VENV); \
		echo "$(GREEN)✓ Virtual environment removed$(NC)"; \
	fi
	@if [ -f ".env" ]; then \
		read -p "$(YELLOW)Delete .env file? [y/N]: $(NC)" confirm; \
		if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
			rm -f .env; \
			echo "$(GREEN)✓ .env file removed$(NC)"; \
		fi; \
	fi

run: ## Run the FastAPI application
	@if [ ! -d "$(VENV)" ]; then \
		echo "$(RED)Error: Virtual environment not found. Run 'make setup' first.$(NC)"; \
		exit 1; \
	fi
	@echo "$(GREEN)Starting FastAPI server...$(NC)"
	@$(VENV_BIN)/uvicorn app.main:app --reload

test: venv ## Run tests (if available)
	@echo "$(GREEN)Running tests...$(NC)"
	@if [ -d "tests" ]; then \
		$(VENV_BIN)/pytest; \
	else \
		echo "$(YELLOW)No tests directory found$(NC)"; \
	fi

