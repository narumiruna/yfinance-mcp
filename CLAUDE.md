# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Yahoo Finance MCP Server (`yfmcp`) - An MCP server providing tools to fetch stock data, news, and financial charts using yfinance.

## Commands

```bash
# Install dependencies
uv sync

# Install with dev dependencies (for demo)
uv sync --extra dev

# Run MCP server
uv run yfmcp

# Run demo chatbot
uv run chainlit run demo.py

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

### MCP Server
- `src/yfmcp/server.py` - Main FastMCP server with all tool implementations (`get_ticker_info`, `get_ticker_news`, `search`, `get_top`, `get_price_history`)
- `src/yfmcp/types.py` - Literal type definitions for enums (Sector, TopType, Period, Interval, ChartType, SearchType)
- Entry point: `yfmcp.server:main` via `yfmcp` command

### Demo Chatbot
- `demo.py` - Chainlit-based chatbot demo that connects to the MCP server
  - Supports both OpenAI and LiteLLM backends
  - Handles tool calling with automatic image/chart display
  - Manages persistent MCP session throughout chat lifecycle
  - Key functions:
    - `get_mcp_client()` - Creates MCP client connection
    - `extract_tool_result()` - Extracts text and images from tool results
    - `convert_mcp_tools_to_openai_format()` - Converts MCP tools to OpenAI format
    - `handle_error()` - Unified error handling and logging

## Key Dependencies

### MCP Server
- `mcp[cli]` - FastMCP framework for MCP server
- `yfinance` - Yahoo Finance data API
- `mplfinance` - Financial charting library

### Demo Chatbot (dev dependencies)
- `chainlit` - Conversational AI framework for building chat interfaces
- `openai` - OpenAI Python client (for OpenAI backend)
- `litellm` - Unified LLM API wrapper (for LiteLLM backend)
- `python-dotenv` - Environment variable management
- `loguru` - Logging library
- `Pillow` - Image processing (for WebP to PNG conversion)
