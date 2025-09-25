.PHONY: help install clean test lint format train train-linear train-cnn train-both demo api docker-build docker-up docker-down

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
DOCKER_COMPOSE := docker-compose
OUTPUT_DIR := outputs

# Colors for terminal output
BOLD := \033[1m
GREEN := \033[32m
YELLOW := \033[33m
BLUE := \033[34m
RESET := \033[0m

help: ## Show this help message
	@echo "$(BOLD)MNIST Classifier - Available Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-20s$(RESET) %s\n", $$1, $$2}'

install: ## Install dependencies
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev: ## Install development dependencies
	@echo "$(GREEN)Installing development dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

clean: ## Clean build artifacts and cache
	@echo "$(YELLOW)Cleaning build artifacts...$(RESET)"
	rm -rf build/ dist/ *.egg-info
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache/ .coverage coverage.xml htmlcov/
	rm -rf .mypy_cache/

test: ## Run unit tests
	@echo "$(GREEN)Running tests...$(RESET)"
	pytest tests/ -v --cov=src/mnist_classifier --cov-report=term-missing

test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(RESET)"
	pytest tests/ -v --cov=src/mnist_classifier --cov-report=html --cov-report=term
	@echo "$(BLUE)Coverage report generated in htmlcov/index.html$(RESET)"

lint: ## Run linting checks
	@echo "$(GREEN)Running linters...$(RESET)"
	flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503
	mypy src/ --ignore-missing-imports
	black --check src/ tests/
	isort --check-only src/ tests/

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(RESET)"
	black src/ tests/ *.py
	isort src/ tests/ *.py

train: train-both ## Train both models (alias)

train-linear: ## Train linear model
	@echo "$(GREEN)Training Linear model...$(RESET)"
	$(PYTHON) train.py --model linear --epochs 10 --output-dir $(OUTPUT_DIR)

train-cnn: ## Train CNN model
	@echo "$(GREEN)Training CNN model...$(RESET)"
	$(PYTHON) train.py --model cnn --epochs 10 --output-dir $(OUTPUT_DIR)

train-both: ## Train both models and compare
	@echo "$(GREEN)Training both models...$(RESET)"
	$(PYTHON) train.py --model both --epochs 10 --output-dir $(OUTPUT_DIR)

train-quick: ## Quick training for testing (2 epochs)
	@echo "$(GREEN)Quick training (2 epochs)...$(RESET)"
	$(PYTHON) train.py --model both --epochs 2 --output-dir $(OUTPUT_DIR)

demo: ## Launch Gradio web demo
	@echo "$(GREEN)Launching Gradio demo...$(RESET)"
	$(PYTHON) app.py

api: ## Launch FastAPI service
	@echo "$(GREEN)Launching FastAPI service...$(RESET)"
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

docker-build: ## Build Docker images
	@echo "$(GREEN)Building Docker images...$(RESET)"
	$(DOCKER_COMPOSE) build

docker-up: ## Start Docker services
	@echo "$(GREEN)Starting Docker services...$(RESET)"
	$(DOCKER_COMPOSE) up -d
	@echo "$(BLUE)Services available at:$(RESET)"
	@echo "  - API: http://localhost:8000"
	@echo "  - Web Demo: http://localhost:7860"

docker-down: ## Stop Docker services
	@echo "$(YELLOW)Stopping Docker services...$(RESET)"
	$(DOCKER_COMPOSE) down

docker-logs: ## Show Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-train: ## Train models in Docker
	@echo "$(GREEN)Training models in Docker...$(RESET)"
	$(DOCKER_COMPOSE) --profile training up train

notebook: ## Convert notebook to Python script
	@echo "$(GREEN)Converting notebook to Python script...$(RESET)"
	jupyter nbconvert --to script mnist_linear_classifier.ipynb

download-data: ## Download MNIST dataset
	@echo "$(GREEN)Downloading MNIST dataset...$(RESET)"
	$(PYTHON) -c "from torchvision import datasets; datasets.MNIST(root='./data', download=True)"

benchmark: ## Run performance benchmark
	@echo "$(GREEN)Running performance benchmark...$(RESET)"
	$(PYTHON) -c "from src.mnist_classifier.models import *; import time; import torch; \
		models = [LinearClassifier(), CNNClassifier()]; \
		x = torch.randn(100, 1, 28, 28); \
		for m in models: \
			m.eval(); \
			start = time.time(); \
			with torch.no_grad(): _ = m(x); \
			elapsed = time.time() - start; \
			print(f'{m.__class__.__name__}: {elapsed*1000:.2f}ms for 100 samples')"

check: lint test ## Run all checks (lint + test)
	@echo "$(GREEN)All checks passed!$(RESET)"

setup: install download-data ## Complete setup (install + download data)
	@echo "$(GREEN)Setup complete!$(RESET)"

all: clean setup train test ## Full pipeline (clean, setup, train, test)
	@echo "$(GREEN)Full pipeline complete!$(RESET)"