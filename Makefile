# Makefile for Eigen3 development

.PHONY: help install install-dev install-evorl test test-unit test-integration clean format lint type-check

help:
	@echo "Available commands:"
	@echo "  make install          - Install eigen3 package"
	@echo "  make install-dev      - Install with development dependencies"
	@echo "  make install-evorl    - Install EvoRL framework"
	@echo "  make test             - Run all tests"
	@echo "  make test-unit        - Run unit tests only"
	@echo "  make test-integration - Run integration tests only"
	@echo "  make format           - Format code with black and isort"
	@echo "  make lint             - Run flake8 linter"
	@echo "  make type-check       - Run mypy type checker"
	@echo "  make clean            - Clean build artifacts and cache"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

install-evorl:
	cd evorl && pip install -e .

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v --tb=short

format:
	black eigen3/ tests/ scripts/
	isort eigen3/ tests/ scripts/

lint:
	flake8 eigen3/ tests/ scripts/ --max-line-length=100 --ignore=E203,W503

type-check:
	mypy eigen3/ --ignore-missing-imports

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ htmlcov/ .coverage
