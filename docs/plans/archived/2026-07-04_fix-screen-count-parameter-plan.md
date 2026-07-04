## Goal

Fix `yfinance_screen` so the public `count` parameter correctly limits predefined screener results, while keeping the existing MCP interface and validation behavior stable.

Success means `yfinance_screen(query="day_gainers", query_type="predefined", count=1)` returns one quote through the MCP tool, and unit tests prove the wrapper sends the upstream parameter that currently works.

## Context

Live checks showed that `yfinance.screen("day_gainers", count=1)` and `count=5` both return 25 quotes, while `yfinance.screen("day_gainers", size=1)` and `size=5` return the requested number of quotes. The wrapper currently rejects `size` for predefined queries and passes `count=count` to `yf.screen`.

## Non-Goals

- Do not change the MCP public schema to require `size` for predefined screeners.
- Do not change custom equity or fund screener behavior, where `size` remains the supported row-limit parameter.
- Do not add new dependencies.

## Assumptions

- Keeping `count` as the predefined-query public parameter avoids a breaking interface change.
- Mapping predefined `count` to upstream `size` is acceptable because current direct `yfinance` behavior verifies `size` is the effective limiter.

## Plan

- [x] Add a focused failing unit test in `tests/test_server_unit.py` for predefined `screen(..., count=10)` that asserts `asyncio.to_thread` calls `yf.screen` with `size=10` and `count=None`; verified failure before implementation with `uv run pytest tests/test_server_unit.py -k screen_predefined -q` showing `assert None == 10`.
- [x] Update `src/yfmcp/server.py` so the predefined branch passes `size=count` and `count=None` to `yf.screen`, while custom `equity` and `fund` branches keep passing `size=size` and `count=None`; verified with `uv run pytest tests/test_server_unit.py -k screen -q` showing 11 passed.
- [x] Preserve validation that rejects `size` for `query_type="predefined"` and rejects `count` for `query_type in {"equity", "fund"}`; verified existing invalid-parameter tests still pass with `uv run pytest tests/test_server_unit.py -k "screen and rejects" -q` showing 2 passed.
- [x] Run formatting and lint checks for the touched Python files; verified with `uv run ruff format tests/test_server_unit.py src/yfmcp/server.py` showing 2 files left unchanged and `uv run ruff check tests/test_server_unit.py src/yfmcp/server.py` showing all checks passed.
- [x] Run the project type check and relevant full test slice; verified with `uv run ty check src tests` showing all checks passed and `uv run pytest -v -s --cov=src tests` showing 99 passed.
- [x] Manually recheck the MCP-visible behavior by calling `yfinance_screen` with `query="day_gainers"`, `query_type="predefined"`, and `count=1`; verified by a fresh local stdio MCP server call showing `count 1`, `quotes_len 1`, and `total 404`.

## Risks

- If a future `yfinance` release makes `count` work differently, the wrapper will still use `size` for predefined queries. This is acceptable for now because `size` is verified against current upstream behavior and preserves the MCP API.
- The live screener result depends on Yahoo Finance availability, so the unit test must assert wrapper parameter translation rather than depend on live market data.

## Completion Checklist

- [x] `yfinance_screen` predefined queries honor public `count` by returning the requested number of quotes, verified by a fresh local stdio MCP call with `count=1` showing `count 1` and `quotes_len 1`.
- [x] Unit tests cover the upstream `size=count` translation for predefined screeners, verified by `uv run pytest tests/test_server_unit.py -k screen_predefined -q` showing 4 passed after implementation.
- [x] Custom screener behavior and validation remain unchanged, verified by `uv run pytest tests/test_server_unit.py -k screen -q` showing 11 passed.
- [x] Formatting, linting, type checking, and the full pytest suite pass, verified by `uv run ruff format tests/test_server_unit.py src/yfmcp/server.py`, `uv run ruff check tests/test_server_unit.py src/yfmcp/server.py`, `uv run ty check src tests`, `uv run pytest -v -s --cov=src tests`, and `prek run -a`.
