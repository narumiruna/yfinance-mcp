# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Yahoo Finance MCP Server (`yfmcp`) - An MCP server providing tools to fetch stock data, news, and financial charts using yfinance.

## Commands

```bash
# Install dependencies
uv sync

# Run MCP server
uv run yfmcp

# Run tests
uv run pytest

# Run single test
uv run pytest tests/test_server.py::test_get_ticker_info -v

# Lint and format
uv run ruff check .
uv run ruff format .

# Type check
uv run ty src/
```

## Architecture

- `src/yfmcp/server.py` - Main FastMCP server with all tool implementations (`get_ticker_info`, `get_ticker_news`, `search`, `get_top`, `get_price_history`, `get_chart`)
- `src/yfmcp/types.py` - Literal type definitions for enums (Sector, TopType, Period, Interval, ChartType, SearchType)
- Entry point: `yfmcp.server:main` via `yfmcp` command

## Key Dependencies

- `mcp[cli]` - FastMCP framework for MCP server
- `yfinance` - Yahoo Finance data API
- `mplfinance` - Financial charting library
