import json
from datetime import datetime
from typing import Annotated

import yfinance as yf
from loguru import logger
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from pydantic import Field
from yfinance.const import SECTOR_INDUSTY_MAPPING

from yfmcp.chart import generate_chart
from yfmcp.types import ChartType
from yfmcp.types import Interval
from yfmcp.types import Period
from yfmcp.types import SearchType
from yfmcp.types import Sector
from yfmcp.types import TopType

# https://github.com/jlowin/fastmcp/issues/81#issuecomment-2714245145
mcp = FastMCP("Yahoo Finance MCP Server", log_level="ERROR")


@mcp.tool()
def get_ticker_info(symbol: Annotated[str, Field(description="The stock symbol")]) -> str:
    """Retrieve stock data including company info, financials, trading metrics and governance data."""
    ticker = yf.Ticker(symbol)

    # Convert timestamps to human-readable format
    info = ticker.info
    for key, value in info.items():
        if not isinstance(key, str):
            continue

        if key.lower().endswith(("date", "start", "end", "timestamp", "time", "quarter")):
            try:
                info[key] = datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logger.error("Unable to convert {}: {} to datetime, got error: {}", key, value, e)
                continue

    return json.dumps(info, ensure_ascii=False)


@mcp.tool()
def get_ticker_news(symbol: Annotated[str, Field(description="The stock symbol")]) -> str:
    """Fetches recent news articles related to a specific stock symbol with title, content, and source details."""
    ticker = yf.Ticker(symbol)
    news = ticker.get_news()
    return str(news)


@mcp.tool()
def search(
    query: Annotated[str, Field(description="The search query (ticker symbol or company name)")],
    search_type: Annotated[SearchType, Field(description="Type of search results to retrieve")],
) -> str:
    """Fetches and organizes search results from Yahoo Finance, including stock quotes and news articles."""
    s = yf.Search(query)
    match search_type.lower():
        case "all":
            return json.dumps(s.all, ensure_ascii=False)
        case "quotes":
            return json.dumps(s.quotes, ensure_ascii=False)
        case "news":
            return json.dumps(s.news, ensure_ascii=False)
        case _:
            return "Invalid output_type. Use 'all', 'quotes', or 'news'."


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

    try:
        s = yf.Sector(sector)
        df = s.top_companies
    except Exception as e:
        return json.dumps({"error": f"Failed to get top companies for sector '{sector}': {e}"})
    if df is None:
        return json.dumps({"error": f"No top companies available for {sector} sector."})
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


@mcp.tool()
def get_price_history(
    symbol: Annotated[str, Field(description="The stock symbol")],
    period: Annotated[Period, Field(description="Time period to retrieve data for (e.g. '1d', '1mo', '1y').")] = "1mo",
    interval: Annotated[Interval, Field(description="Data interval frequency (e.g. '1d', '1h', '1m')")] = "1d",
    chart_type: Annotated[
        ChartType | None,
        Field(
            description=(
                "Type of chart: 'price_volume' for candlestick with volume bars, "
                "'vwap' for Volume Weighted Average Price, or 'volume_profile' "
                "for volume distribution by price level"
            )
        ),
    ] = None,
) -> str | ImageContent:
    """Fetch historical price data for a given stock symbol over a specified period and interval."""
    ticker = yf.Ticker(symbol)

    df = ticker.history(
        period=period,
        interval=interval,
        rounding=True,
    )

    if df.empty:
        return f"No data available for symbol {symbol} with period {period} and interval {interval}"

    if chart_type is None:
        return df.to_markdown()

    return generate_chart(symbol=symbol, df=df, chart_type=chart_type)


def main() -> None:
    mcp.run()
