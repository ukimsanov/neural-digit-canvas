.PHONY: help install clean train api frontend run

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip

# Colors for terminal output
GREEN := \033[32m
BLUE := \033[34m
RESET := \033[0m

help: ## Show available commands
	@echo "$(GREEN)MNIST Classifier - Available Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "$(BLUE)%-15s$(RESET) %s\n", $$1, $$2}'

install: ## Install Python dependencies
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

clean: ## Clean cache files
	@echo "$(GREEN)Cleaning cache...$(RESET)"
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/

train: ## Train both models
	@echo "$(GREEN)Training models...$(RESET)"
	$(PYTHON) train.py --model both --epochs 10

api: ## Start API server
	@echo "$(GREEN)Starting API server...$(RESET)"
	uvicorn api:app --reload --host 0.0.0.0 --port 8000

frontend: ## Start frontend dev server
	@echo "$(GREEN)Starting frontend...$(RESET)"
	cd frontend && npm run dev

run: ## Start both API and frontend
	@echo "$(GREEN)Starting full application...$(RESET)"
	./run.sh