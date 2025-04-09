# Yahoo Finance MCP Server

A simple MCP server for Yahoo Finance using yfinance. This server provides a set of tools to fetch stock data, news, and other financial information.

## Tools

- **get_ticker_info**

  - Retrieve stock data including company info, financials, trading metrics and governance data.
  - Inputs:
    - `symbol` (string): The stock symbol.

- **get_ticker_news**

  - Fetches recent news articles related to a specific stock symbol with title, content, and source details.
  - Inputs:
    - `symbol` (string): The stock symbol.

- **search_quote**

  - Search for stock quotes with company name, exchange, sector and industry information by keyword.
  - Inputs:
    - `query` (string): The search query.
    - `max_results` (number, optional): Maximum number of results (default 8).

- **search_news**

  - Search for financial news articles matching a keyword with title, source and publication details.
  - Inputs:
    - `query` (string): The search query.
    - `news_count` (number, optional): Number of news articles (default 8).

- **get_top**

  - Get top entities (ETFs, mutual funds, companies, growth companies, or performing companies) in a sector.
  - Inputs:
    - `sector` (string): The sector to get.
    - `top_type` (string): Type of top companies to retrieve (options: "top_etfs", "top_mutual_funds", "top_companies", "top_growth_companies", "top_performing_companies").
    - `top_n` (number, optional): Number of top entities to retrieve (default 10).

## Usage

1. [Install uv.](https://docs.astral.sh/uv/getting-started/installation/)
2. Add the following configuration to your MCP server configuration file.

```json
{
  "mcpServers": {
    "yfmcp": {
      "command": "uvx",
      "args": ["yfmcp"]
    }
  }
}
```

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
