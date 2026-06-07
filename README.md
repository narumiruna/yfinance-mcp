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
- **Ownership Data** — Major holders, institutional investors, mutual fund holders, and insider transactions

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

### `yfinance_screen`

Run Yahoo Finance screeners using either predefined screener keys or custom query trees.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `query` | string/object | Yes | For `query_type="predefined"`: screener key such as `"day_gainers"`. For `query_type="equity"` or `"fund"`: custom query tree with `{operator, operands}` nodes |
| `query_type` | string | No | `"predefined"` (default), `"equity"`, or `"fund"` |
| `offset` | number | No | Result offset |
| `size` | number | No | Rows for custom queries; Yahoo maximum is `250` |
| `count` | number | No | Rows for predefined queries; Yahoo maximum is `250` |
| `sort_field` | string | No | Sort field, for example `"percentchange"` |
| `sort_asc` | boolean | No | Sort ascending if `true`, descending if `false` |
| `user_id` | string | No | Optional Yahoo user identifier |
| `user_id_type` | string | No | Optional Yahoo user ID type, commonly `"guid"` |

**Returns:** JSON screener response from Yahoo Finance, typically including quote rows and metadata.

Custom equity screener example:

```json
{
  "query_type": "equity",
  "query": {
    "operator": "and",
    "operands": [
      { "operator": "gt", "operands": ["percentchange", 3] },
      { "operator": "eq", "operands": ["region", "us"] },
      { "operator": "gte", "operands": ["intradayprice", 5] },
      { "operator": "gt", "operands": ["dayvolume", 500000] }
    ]
  },
  "sort_field": "percentchange",
  "sort_asc": false,
  "size": 50
}
```

### `yfinance_screen_gappers`

Run a purpose-built custom screener for opening-session bullish gappers.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `min_percent_change` | number | No | Minimum percent gap/change from prior close (default: `3.0`) |
| `min_price` | number | No | Minimum intraday price (default: `5.0`) |
| `min_volume` | number | No | Minimum day volume (default: `500000`) |
| `min_market_cap` | number | No | Minimum intraday market cap in USD (default: `2000000000`) |
| `region` | string | No | Yahoo region code (default: `"us"`) |
| `size` | number | No | Number of results (default: `50`, max: `250`) |
| `offset` | number | No | Result offset for pagination (default: `0`) |
| `sort_asc` | boolean | No | Sort by `percentchange` ascending (`true`) or descending (`false`, default) |

**Returns:** JSON screener response from Yahoo Finance.

### `yfinance_get_price_history`

Fetch historical price data and optionally generate technical analysis charts.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock ticker symbol |
| `period` | string | No | Time range — `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max` (default: `1mo`) |
| `interval` | string | No | Data granularity — `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`, `3mo` (default: `1d`) |
| `chart_type` | string | No | Chart to generate (omit for tabular data) |
| `prepost` | boolean | No | Include pre-market and post-market data when available (default: `false`; useful with intraday requests like `period="1d"`, `interval="1m"`) |

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

### `yfinance_get_holders`

Fetch major holders, institutional holders, mutual fund holders, and insider data.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `symbol` | string | Yes | Stock ticker symbol (e.g. `AAPL`, `MSFT`) |
| `max_rows` | number | No | Maximum rows returned per holder section. Default: `10`. Use `0` to return all rows |

**Returns:** JSON object with:
- **`major_holders`** — Aggregated breakdown where each row has an `index` label (e.g. `insidersPercentHeld`, `institutionsPercentHeld`, `institutionsFloatPercentHeld`, `institutionsCount`) and a `Value`
- **`institutional_holders`** — Institutional investors; records typically include fields such as `Date Reported`, `Holder`, `Shares`, `Value`, `pctChange`, `pctHeld`
- **`mutualfund_holders`** — Mutual fund holders; records typically include fields similar to institutional holders
- **`insider_transactions`** — Recent insider trades; records typically include fields such as `Shares`, `Value`, `Insider`, `Position`, `Transaction`, `Start Date`, `Ownership`
- **`insider_purchases`** — Six-month summary where each row describes a category (Purchases, Sales, Net Shares, etc.); records typically include fields such as `Insider Purchases Last 6m`, `Shares`, `Trans`
- **`insider_roster`** — Known insiders; records typically include fields such as `Name`, `Position`, `Shares Owned Directly`, `Most Recent Transaction`, `Latest Transaction Date`
- **`_metadata`** — Row limit metadata with `max_rows` and per-section `total_rows`, `returned_rows`, and `truncated`

Holder sections are limited to 10 rows by default to keep responses concise. Pass `max_rows: 0` when you need the complete holder datasets. Field names for holder-related datasets are provided by `yfinance` and may vary by ticker, data availability, and `yfinance` version.

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

### Testing with Codex CLI

This repository includes `.codex/config.toml`, which registers the local `yfmcp` MCP server for Codex CLI using `uv run yfmcp`. After cloning the repository and running `uv sync`, open Codex CLI from the repository root and try prompts such as:

```text
Show VOO ticker info
Show VOO price history for the last 5 days
Find the ticker symbol for Toyota
Get AAPL option expiration dates
```

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
