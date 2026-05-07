.PHONY: install test status validate-status ci-check lint format env-update clean

install:
	pip install -e ".[dev]" && pre-commit install

test:
	pytest tests/ -v

status:
	@echo "===== STATUS.md ====="
	@cat STATUS.md
	@echo ""
	@echo "===== TODO.md ====="
	@cat TODO.md

validate-status:
	@echo "Validating STATUS.md and TODO.md consistency..."
	@test -f STATUS.md || (echo "ERROR: STATUS.md not found" && exit 1)
	@test -f TODO.md || (echo "ERROR: TODO.md not found" && exit 1)
	@echo "STATUS.md and TODO.md both present."
	@echo "Validation passed."

ci-check: validate-status lint test
	@echo "All CI checks passed."

lint:
	ruff check code/ tests/ scripts/
	black --check code/ tests/ scripts/

format:
	black code/ tests/ scripts/
	ruff check --fix code/ tests/ scripts/

env-update:
	conda env update -f environment.yml --prune && pip install -e ".[dev]"

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
