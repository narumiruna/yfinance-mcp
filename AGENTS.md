# Repository Guidelines

## Project Structure & Module Organization
- `src/yfmcp/server.py`: FastMCP server, tool registration, and async wrappers around `yfinance` calls.
- `src/yfmcp/chart.py`: chart generation (`price_volume`, `vwap`, `volume_profile`) and WebP image encoding.
- `src/yfmcp/types.py`: shared Literal types (`SearchType`, `TopType`, `Period`, `Interval`, `ChartType`, `ErrorCode`).
- `src/yfmcp/utils.py`: JSON helpers, including `create_error_response()`.
- `tests/`: async pytest suite for server tools, charts, and type behavior.
- `README.md`: end-user setup and usage guide. Demo chatbot link points to the dedicated `yfinance-mcp-demo` repository.

## Architecture Overview
- MCP tools are exposed from `yfmcp.server` with `yfinance_`-prefixed names:
  `yfinance_get_ticker_info`, `yfinance_get_ticker_news`, `yfinance_search`, `yfinance_get_top`, `yfinance_get_price_history`.
- All blocking `yfinance` operations MUST be wrapped with `asyncio.to_thread()`.
- Errors MUST be returned via `create_error_response()` with structured JSON (`error`, `error_code`, optional `details`).
- Chart responses are returned as base64-encoded WebP `ImageContent`; tabular history uses Markdown tables.

## Build, Test, and Development Commands
- `uv sync`: install runtime dependencies.
- `uv sync --extra dev`: install development dependencies.
- `uv run yfmcp`: run the MCP server.
- `uv run ruff check .` and `uv run ruff format .`: lint/format.
- `uv run ty check src tests`: type check.
- `uv run pytest -v -s --cov=src tests`: full test run with coverage.

## Coding Style & Testing Guidelines
- Python `>=3.12`, line length `120`, one import per line (ruff isort).
- Test files: `tests/test_*.py`; test functions: `test_*`.
- Use `pytest` + `pytest-asyncio` for async behavior.
- If `.pre-commit-config.yaml` exists and code/config changed, run `prek run -a` (fallback: `pre-commit run -a`).

## Commit & Pull Request Guidelines
- Prefer short, imperative commit subjects (for example: `Fix import order`, `Update README`).
- Keep commits focused; include tests when behavior changes.
- PRs should summarize intent and list the validation commands run.

## Configuration & Secrets
- Never commit secrets or generated artifacts.
