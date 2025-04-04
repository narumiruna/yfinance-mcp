# Yahoo Finance MCP Server

A simple MCP server for Yahoo Finance using yfinance. This server provides a set of tools to fetch stock data, news, and other financial information.

## Tools

- **get_ticker_info**

  - Retrieve information about a specific stock symbol using Yahoo Finance API.
  - Inputs:
    - `symbol` (string): The stock symbol.

- **get_ticker_news**

  - Fetch news articles for a given stock ticker symbol.
  - Inputs:
    - `symbol` (string): The stock symbol.

- **search_quote**

  - Search for quotes using a query string.
  - Inputs:
    - `query` (string): The search query.
    - `max_results` (number, optional): Maximum number of results (default 8).

- **search_news**

  - Search for news articles using a query string.
  - Inputs:
    - `query` (string): The search query.
    - `news_count` (number, optional): Number of news articles (default 8).

- **get_top_etfs**

  - Retrieve the top ETFs in a specific sector.
  - Inputs:
    - `sector` (string): The sector to get ETFs for.

- **get_top_mutual_funds**

  - Retrieve the top mutual funds in a specific sector.
  - Inputs:
    - `sector` (string): The sector to get mutual funds for.

- **get_top_companies**

  - Retrieve the top companies in a specific sector.
  - Inputs:
    - `sector` (string): The sector to get companies for.
    - `top_n` (number): Number of top companies to retrieve.

- **get_top_growth_companies**

  - Retrieve the top growth companies in a specific sector.
  - Inputs:
    - `sector` (string): The sector to get companies for.
    - `top_n` (number): Number of top growth companies to retrieve.

- **get_top_performing_companies**

  - Retrieve the top performing companies in a specific sector.
  - Inputs:
    - `sector` (string): The sector to get companies for.
    - `top_n` (number): Number of top performing companies to retrieve.

- **analyze_sentiment**
  - Analyze sentiment for a given stock symbol based on provided rationale.
  - Inputs:
    - `symbol` (string): The stock symbol.
    - `reasoning` (string): The rationale behind the sentiment analysis.
    - `sentiment` (string): The sentiment label ('positive', 'negative', or 'neutral').
    - `score` (number): The sentiment score ranging from -1 to 1.

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
