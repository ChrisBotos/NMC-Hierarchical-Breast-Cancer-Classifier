.PHONY: test lint format install env-update clean

install:
	pip install -e ".[dev]" && pre-commit install

test:
	pytest tests/ -v

lint:
	ruff check code/ tests/ presentation/ model/
	black --check code/ tests/ presentation/ model/

format:
	black code/ tests/ presentation/ model/
	ruff check --fix code/ tests/ presentation/ model/

ci-check: lint test
	@echo "All CI checks passed."

env-update:
	conda env update -f environment.yml --prune && pip install -e ".[dev]"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
