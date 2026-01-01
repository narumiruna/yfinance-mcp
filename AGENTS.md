# Repository Guidelines

## Project Structure & Module Organization
- `src/yfmcp/` holds the MCP server implementation, charting utilities, and type definitions.
- `tests/` contains async pytest coverage for server behavior and charting.
- `demo.py` is a Chainlit demo client for local experimentation.
- `pyproject.toml` defines dependencies, tooling, and formatting rules.

## Architecture Overview
- `yfmcp.server` exposes FastMCP tools and wraps blocking `yfinance` calls with `asyncio.to_thread()`.
- `yfmcp.chart` generates WebP charts for MCP responses; the demo converts them for display.
- `yfmcp.types` centralizes Literal enums for tool inputs and validation.

## Build, Test, and Development Commands
- `uv sync`: install runtime dependencies.
- `uv sync --extra dev`: install demo and dev tooling (Chainlit, pytest, ruff, ty).
- `uv run yfmcp`: run the MCP server locally.
- `uv run chainlit run demo.py`: start the demo chatbot at `http://localhost:8000`.
- `uv run ruff check .`: run linting.
- `uv run ty check src tests`: run type checks.
- `uv run pytest -v -s --cov=src tests`: run pytest with coverage.

## Coding Style & Naming Conventions
- Python 3.12+ only; keep imports one-per-line (ruff isort config).
- Line length is 120; use `ruff format .` for formatting.
- Tool functions are async; tool names in `@mcp.tool` are prefixed `yfinance_` (see `src/yfmcp/server.py`).
- Use `_error()` helper for consistent JSON error responses.

## Testing Guidelines
- Frameworks: `pytest` + `pytest-asyncio` (async tests).
- Naming: files `tests/test_*.py`, test functions `test_*`.
- Run all tests with `uv run pytest` or `make test`; target coverage with `--cov=src`.

## Commit & Pull Request Guidelines
- Use short, imperative commit subjects with no trailing period (e.g., "Fix asyncio mock", "Add test cases").
- If helpful, include a brief body explaining why the change is needed.
- Keep commits focused; include tests or reasoning when behavior changes.
- PRs should describe intent, note testing done, and include screenshots for chart/demo changes.

## Configuration & Secrets
- Demo uses `.env` for `OPENAI_API_KEY` or LiteLLM settings (see `README.md`).
- Do not commit API keys or generated artifacts; prefer local `.env` files.
