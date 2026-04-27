# Yahoo Finance MCP Server

[![PyPI version](https://img.shields.io/pypi/v/yfmcp)](https://pypi.org/project/yfmcp/)
[![Python](https://img.shields.io/pypi/pyversions/yfmcp.svg)](https://pypi.org/project/yfmcp/)
[![CI](https://github.com/narumiruna/yfinance-mcp/actions/workflows/python.yml/badge.svg)](https://github.com/narumiruna/yfinance-mcp/actions/workflows/python.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that provides AI assistants with access to Yahoo Finance data via [yfinance](https://github.com/ranaroussi/yfinance). Query stock information, financial news, sector rankings, and generate professional financial charts — all from your AI chat.

<a href="https://glama.ai/mcp/servers/@narumiruna/yfinance-mcp">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@narumiruna/yfinance-mcp/badge" />
</a>

## Features

- **Stock Data** — Company info, financials, valuation metrics, dividends, and trading data
- **Financial Statements** — Income statement and balance sheet with historical data (EBIT, Invested Capital, etc.)
- **Financial News** — Recent news articles and press releases for any ticker
- **Search** — Find stocks, ETFs, and news across Yahoo Finance
- **Sector Rankings** — Top ETFs, mutual funds, companies, growth leaders, and top performers by sector
- **Price History** — Historical OHLCV data as markdown tables or professional charts
- **Chart Generation** — Candlestick, VWAP, and volume profile charts returned as WebP images
- **Options Data** — Option chains with calls, puts, strike prices, IV, and expiration dates

## Tools

### `yfinance_get_ticker_info`

Retrieve comprehensive stock data including company info, financials, trading metrics, and governance data.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock ticker symbol (e.g. `AAPL`, `GOOGL`, `MSFT`) |

**Returns:** JSON object with company details, price data, valuation metrics, trading info, dividends, financials, and performance indicators.

### `yfinance_get_ticker_news`

Fetch recent news articles and press releases for a specific stock.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock ticker symbol |

**Returns:** JSON array of news items with title, summary, publication date, provider, URL, and thumbnail.

### `yfinance_search`

Search Yahoo Finance for stocks, ETFs, and news articles.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string | Yes | Search query — company name, ticker symbol, or keywords |
| `search_type` | string | Yes | `"all"` (quotes + news), `"quotes"` (stocks/ETFs only), or `"news"` (articles only) |

**Returns:** Matching quotes and/or news results depending on `search_type`.

### `yfinance_get_top`

Get top-ranked financial entities within a market sector.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `sector` | string | Yes | Market sector (see [supported sectors](#supported-sectors) below) |
| `top_type` | string | Yes | `"top_etfs"`, `"top_mutual_funds"`, `"top_companies"`, `"top_growth_companies"`, or `"top_performing_companies"` |
| `top_n` | number | No | Number of results to return (default: `10`, max: `100`) |

**Returns:** JSON array of top entities with relevant metrics.

#### Supported Sectors

`Basic Materials`, `Communication Services`, `Consumer Cyclical`, `Consumer Defensive`, `Energy`, `Financial Services`, `Healthcare`, `Industrials`, `Real Estate`, `Technology`, `Utilities`

### `yfinance_get_price_history`

Fetch historical price data and optionally generate technical analysis charts.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock ticker symbol |
| `period` | string | No | Time range — `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max` (default: `1mo`) |
| `interval` | string | No | Data granularity — `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo` (default: `1d`) |
| `chart_type` | string | No | Chart to generate (omit for tabular data) |

**Chart types:**

| Value | Description |
|-------|-------------|
| `"price_volume"` | Candlestick chart with volume bars |
| `"vwap"` | Price chart with Volume Weighted Average Price overlay |
| `"volume_profile"` | Candlestick chart with volume distribution by price level |

**Returns:**
- Without `chart_type`: Markdown table with Date, Open, High, Low, Close, Volume, Dividends, and Stock Splits columns.
- With `chart_type`: Base64-encoded WebP image for efficient token usage.

### `yfinance_get_financials`

Fetch financial statements (income statement, balance sheet, and cash flow) with historical data.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock ticker symbol |
| `frequency` | string | No | `"annual"` (yearly), `"quarterly"` (quarterly), or `"ttm"` (trailing twelve months). Default: `"annual"` |

**Returns:** JSON object with income statement, balance sheet, and cash flow data for each reporting period.

- **Income Statement fields**: EBIT, Net Income, Tax Provision, Pretax Income, Interest Expense, Total Revenue, Operating Income, EBITDA, Normalized Income
- **Balance Sheet fields**: Stockholders Equity, Total Debt, Cash And Cash Equivalents, Invested Capital, Net Debt, Total Assets, Total Liabilities Net Minority Interest, Net Tangible Assets, Tangible Book Value
- **Cash Flow fields**: Operating Cash Flow, Free Cash Flow, Capital Expenditure, Net Income From Continuing Operations, Depreciation And Amortization, Change In Working Capital, Cash Dividends Paid

### `yfinance_get_option_dates`

Fetch available option expiration dates for a stock.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock ticker symbol (e.g. `AAPL`, `MSFT`) |

**Returns:** JSON array of expiration dates in YYYY-MM-DD format.

### `yfinance_get_option_chain`

Fetch option chain data (calls and puts) for a stock with available strike prices.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock ticker symbol |
| `expiration_date` | string | No | Option expiration date in YYYY-MM-DD format. Omit to fetch all dates. |
| `option_type` | string | No | `"calls"`, `"puts"`, or `"all"` (default: `"all"`) |

**Returns:** JSON object keyed by expiration date, with calls and/or puts data including:
- `contractSymbol`: Option contract identifier
- `strike`: Strike price
- `lastPrice`: Last traded price
- `bid`/`ask`: Bid and ask prices
- `volume`: Trading volume
- `openInterest`: Open interest
- `impliedVolatility`: IV
- `inTheMoney`: Whether option is ITM
- `contractSize`: Contract size (REGULAR)
- `currency`: Currency (USD)

## Usage

### Via uv (recommended)

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Add the following to your MCP client configuration:

```json
{
  "mcpServers": {
    "yfmcp": {
      "command": "uvx",
      "args": ["yfmcp@latest"]
    }
  }
}
```

### Via Docker

```json
{
  "mcpServers": {
    "yfmcp": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "narumi/yfinance-mcp"]
    }
  }
}
```

### From Source

1. Clone the repository and install dependencies:

```bash
git clone https://github.com/narumiruna/yfinance-mcp.git
cd yfinance-mcp
uv sync
```

2. Add the following to your MCP client configuration:

```json
{
  "mcpServers": {
    "yfmcp": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/yfinance-mcp",
        "yfmcp"
      ]
    }
  }
}
```

Replace `/path/to/yfinance-mcp` with the actual path to your cloned repository.

## Development

### Prerequisites

- Python ≥ 3.12
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
uv sync --extra dev
```

### Lint & Format

```bash
uv run ruff check .
uv run ruff format .
```

### Type Check

```bash
uv run ty check src tests
```

### Test

```bash
uv run pytest -v -s --cov=src tests
```

## Demo Chatbot

See the demo chatbot in its dedicated repository: [yfinance-mcp-demo](https://github.com/narumiruna/yfinance-mcp-demo)

## Contributors

<a href="https://github.com/narumiruna/yfinance-mcp/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=narumiruna/yfinance-mcp" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## License

This project is licensed under the [MIT License](LICENSE).
