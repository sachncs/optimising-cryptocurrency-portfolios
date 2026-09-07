.PHONY: help install dev test lint format typecheck clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

dev: ## Install with dev dependencies
	pip install -e ".[dev]"
	pip install pre-commit mypy ruff
	pre-commit install

test: ## Run tests
	PYTHONPATH=src pytest -q

test-cov: ## Run tests with coverage report
	PYTHONPATH=src python -m coverage run -m pytest -q
	python -m coverage report -m

lint: ## Run linting with ruff
	PYTHONPATH=src ruff check src/ tests/

lint-fix: ## Run linting and auto-fix issues
	PYTHONPATH=src ruff check --fix src/ tests/

format: ## Format code with ruff
	PYTHONPATH=src ruff format src/ tests/

typecheck: ## Run type checking with mypy
	PYTHONPATH=src mypy src/cps

check: lint typecheck test ## Run all checks (lint, typecheck, test)

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info .pytest_cache .mypy_cache .ruff_cache htmlcov/ .coverage coverage.xml
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
