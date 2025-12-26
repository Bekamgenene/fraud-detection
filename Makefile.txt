.PHONY: help install install-dev test test-unit test-integration lint format type-check clean build

help:
	@echo "Available commands:"
	@echo "  install       Install production dependencies"
	@echo "  install-dev   Install development dependencies"
	@echo "  test          Run all tests"
	@echo "  test-unit     Run unit tests only"
	@echo "  test-integration Run integration tests only"
	@echo "  lint          Run linting checks"
	@echo "  format        Format code with black and isort"
	@echo "  type-check    Run type checking with mypy"
	@echo "  clean         Clean build artifacts"
	@echo "  build         Build package"

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt
	pip install -e .[dev]
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov=utils --cov-report=html --cov-report=term-missing

test-unit:
	pytest tests/unit/ -v --cov=src --cov=utils

test-integration:
	pytest tests/integration/ -v

lint:
	flake8 src utils tests
	black --check src utils tests
	isort --check-only src utils tests

format:
	black src utils tests
	isort src utils tests

type-check:
	mypy src utils --ignore-missing-imports

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

build:
	python -m build
