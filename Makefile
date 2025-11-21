lint:
	uv run ruff check .

type:
	uv run ty check src tests

test:
	uv run pytest -v -s --cov=src tests

publish:
	uv build -f wheel
	uv publish

dev:
	uv run mcp dev src/yfmcp/server.py

.PHONY: lint test publish
