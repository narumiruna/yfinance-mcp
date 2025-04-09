import json
from typing import Annotated

import yfinance as yf
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from yfinance.const import SECTOR_INDUSTY_MAPPING

from yfmcp.types import Sector
from yfmcp.types import TopType

# https://github.com/jlowin/fastmcp/issues/81#issuecomment-2714245145
mcp = FastMCP("Yahoo Finance MCP Server", log_level="ERROR")


@mcp.tool()
def get_ticker_info(symbol: Annotated[str, Field(description="The stock symbol")]) -> str:
    """Retrieve stock data including company info, financials, trading metrics and governance data."""
    ticker = yf.Ticker(symbol)
    return json.dumps(ticker.info, ensure_ascii=False)


@mcp.tool()
def get_ticker_news(symbol: Annotated[str, Field(description="The stock symbol")]) -> str:
    """Fetches recent news articles related to a specific stock symbol with title, content, and source details."""
    ticker = yf.Ticker(symbol)
    news = ticker.get_news()
    return str(news)


@mcp.tool()
def search_quote(
    query: Annotated[str, Field(description="The search query")],
    max_results: Annotated[int, Field(description="The maximum number of results")] = 8,
) -> str:
    """Search for stock quotes with company name, exchange, sector and industry information by keyword."""
    search = yf.Search(query, max_results=max_results)
    return str(search.quotes)


@mcp.tool()
def search_news(
    query: Annotated[str, Field(description="The search query")],
    news_count: Annotated[int, Field(description="The number of news articles")] = 8,
) -> str:
    """Search for financial news articles matching a keyword with title, source and publication details."""
    search = yf.Search(query, news_count=news_count)
    assert len(search.news) == news_count, f"Expected {news_count} news articles, but got {len(search.news)}"
    return str(search.news)


def get_top_etfs(
    sector: Annotated[Sector, Field(description="The sector to get")],
    top_n: Annotated[int, Field(description="Number of top ETFs to retrieve")],
) -> str:
    """Retrieve popular ETFs for a sector, returned as a list in 'SYMBOL: ETF Name' format."""
    if top_n < 1:
        return "top_n must be greater than 0"

    s = yf.Sector(sector)

    result = [f"{symbol}: {name}" for symbol, name in s.top_etfs.items()]

    return "\n".join(result[:top_n])


def get_top_mutual_funds(
    sector: Annotated[Sector, Field(description="The sector to get")],
    top_n: Annotated[int, Field(description="Number of top mutual funds to retrieve")],
) -> str:
    """Retrieve popular mutual funds for a sector, returned as a list in 'SYMBOL: Fund Name' format."""
    if top_n < 1:
        return "top_n must be greater than 0"

    s = yf.Sector(sector)
    return "\n".join(f"{symbol}: {name}" for symbol, name in s.top_mutual_funds.items())


def get_top_companies(
    sector: Annotated[Sector, Field(description="The sector to get")],
    top_n: Annotated[int, Field(description="Number of top companies to retrieve")],
) -> str:
    """Get top companies in a sector with name, analyst rating, and market weight as JSON array."""
    if top_n < 1:
        return "top_n must be greater than 0"

    s = yf.Sector(sector)
    df = s.top_companies
    if df is None:
        return f"No top companies available for {sector} sector."

    return df.iloc[:top_n].to_json(orient="records")


def get_top_growth_companies(
    sector: Annotated[Sector, Field(description="The sector to get")],
    top_n: Annotated[int, Field(description="Number of top growth companies to retrieve")],
) -> str:
    """Get top growth companies grouped by industry within a sector as JSON array with growth metrics."""
    if top_n < 1:
        return "top_n must be greater than 0"

    results = []

    for industry_name in SECTOR_INDUSTY_MAPPING[sector]:
        industry = yf.Industry(industry_name)

        df = industry.top_growth_companies
        if df is None:
            continue

        results.append(
            {
                "industry": industry_name,
                "top_growth_companies": df.iloc[:top_n].to_json(orient="records"),
            }
        )
    return json.dumps(results, ensure_ascii=False)


def get_top_performing_companies(
    sector: Annotated[Sector, Field(description="The sector to get")],
    top_n: Annotated[int, Field(description="Number of top performing companies to retrieve")],
) -> str:
    """Get top performing companies grouped by industry within a sector as JSON array with performance metrics."""
    if top_n < 1:
        return "top_n must be greater than 0"

    results = []

    for industry_name in SECTOR_INDUSTY_MAPPING[sector]:
        industry = yf.Industry(industry_name)

        df = industry.top_performing_companies
        if df is None:
            continue

        results.append(
            {
                "industry": industry_name,
                "top_performing_companies": df.iloc[:top_n].to_json(orient="records"),
            }
        )
    return json.dumps(results, ensure_ascii=False)


@mcp.tool()
def get_top(
    sector: Annotated[Sector, Field(description="The sector to get")],
    top_type: Annotated[TopType, Field(description="Type of top companies to retrieve")],
    top_n: Annotated[int, Field(description="Number of top entities to retrieve (limit the results)")] = 10,
) -> str:
    """Get top entities (ETFs, mutual funds, companies, growth companies, or performing companies) in a sector."""
    match top_type:
        case "top_etfs":
            return get_top_etfs(sector, top_n)
        case "top_mutual_funds":
            return get_top_mutual_funds(sector, top_n)
        case "top_companies":
            return get_top_companies(sector, top_n)
        case "top_growth_companies":
            return get_top_growth_companies(sector, top_n)
        case "top_performing_companies":
            return get_top_performing_companies(sector, top_n)
        case _:
            return "Invalid top_type"


def main():
    mcp.run()
