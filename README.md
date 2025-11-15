# Yahoo Finance MCP Server

A simple MCP server for Yahoo Finance using [yfinance](https://github.com/ranaroussi/yfinance). This server provides a set of tools to fetch stock data, news, and other financial information.

<a href="https://glama.ai/mcp/servers/@narumiruna/yfinance-mcp">
  <img width="380" height="200" src="https://glama.ai/mcp/servers/@narumiruna/yfinance-mcp/badge" />
</a>

## Tools

- **get_ticker_info**

  - Retrieve stock data including company info, financials, trading metrics and governance data.
  - Inputs:
    - `symbol` (string): The stock symbol.

- **get_ticker_news**

  - Fetches recent news articles related to a specific stock symbol with title, content, and source details.
  - Inputs:
    - `symbol` (string): The stock symbol.

- **search**

  - Fetches and organizes search results from Yahoo Finance, including stock quotes and news articles.
  - Inputs:
    - `query` (string): The search query (ticker symbol or company name).
    - `search_type` (string): Type of search results to retrieve (options: "all", "quotes", "news").

- **get_top**

  - Get top entities (ETFs, mutual funds, companies, growth companies, or performing companies) in a sector.
  - Inputs:
    - `sector` (string): The sector to get.
    - `top_type` (string): Type of top companies to retrieve (options: "top_etfs", "top_mutual_funds", "top_companies", "top_growth_companies", "top_performing_companies").
    - `top_n` (number, optional): Number of top entities to retrieve (default 10).

- **get_price_history**

  - Fetch historical price data for a given stock symbol over a specified period and interval.
  - Inputs:
    - `symbol` (string): The stock symbol.
    - `period` (string, optional): Time period to retrieve data for (e.g. '1d', '1mo', '1y'). Default is '1mo'.
    - `interval` (string, optional): Data interval frequency (e.g. '1d', '1h', '1m'). Default is '1d'.

- **get_chart**

  - Generate a financial chart using mplfinance showing candlestick price data with volume bars, optionally with VWAP (Volume Weighted Average Price) overlay or volume profile. Returns a base64-encoded WebP image for efficient token usage.
  - Inputs:
    - `symbol` (string): The stock symbol.
    - `chart_type` (string, optional): Type of chart to generate. Options:
      - "price_volume" (default): Candlestick chart with volume bars
      - "vwap": Volume Weighted Average Price chart with VWAP overlay
      - "volume_profile": Candlestick chart with volume profile showing volume distribution by price level (displayed as a histogram on the right side)
    - `period` (string, optional): Time period to retrieve data for (e.g. '1d', '5d', '1mo'). For intraday charts, use '1d' or '5d'. Default is '1d'.
    - `interval` (string, optional): Data interval frequency. For day charts, use '1m', '2m', '5m', '15m', '30m', '60m', or '1h'. Default is '5m'.
  - Output: JSON object containing:
    - `image_base64`: Base64-encoded WebP image string
    - `symbol`: The stock symbol
    - `chart_type`: The chart type used
    - `period`: The time period
    - `interval`: The data interval
    - `data_points`: Number of data points in the chart
    - `format`: Image format (always "webp")

## Usage

You can use this MCP server either via uv (Python package installer) or Docker.

### Via uv

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
2. Add the following configuration to your MCP server configuration file:

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

Add the following configuration to your MCP server configuration file:

```json
{
  "mcpServers": {
    "yfmcp": {
      "command": "docker",
      "args": ["run", "-i", "--rm", "narumi/yfinance-mcp"]
    }
  }
}
