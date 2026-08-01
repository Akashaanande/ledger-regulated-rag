.PHONY: install lint test eval up down

install:
	pip install -e ".[dev]"

lint:
	ruff check src tests
	mypy src

test:
	pytest --cov=ledger --cov-report=term-missing

up:
	docker compose up -d

down:
	docker compose down

eval:
	python -m ledger.eval.run --config all --out RESULTS.md
