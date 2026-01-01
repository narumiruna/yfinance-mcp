import asyncio
import json
from datetime import datetime
from typing import Annotated

import yfinance as yf
from loguru import logger
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from mcp.types import ToolAnnotations
from pydantic import Field
from yfinance.const import SECTOR_INDUSTY_MAPPING

from yfmcp.chart import generate_chart
from yfmcp.types import ChartType
from yfmcp.types import ErrorCode
from yfmcp.types import Interval
from yfmcp.types import Period
from yfmcp.types import SearchType
from yfmcp.types import Sector
from yfmcp.types import TopType

# https://github.com/jlowin/fastmcp/issues/81#issuecomment-2714245145
mcp = FastMCP("yfinance_mcp", log_level="ERROR")


def _dump_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _error(message: str, error_code: ErrorCode = "UNKNOWN_ERROR", details: dict | None = None) -> str:
    """Create a structured error response.

    Args:
        message: Human-readable error message
        error_code: Machine-readable error code for client handling
        details: Optional additional error details

    Returns:
        JSON string with error information
    """
    error_obj = {"error": message, "error_code": error_code}
    if details:
        error_obj["details"] = details
    return _dump_json(error_obj)


@mcp.tool(
    name="yfinance_get_ticker_info",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_ticker_info(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
) -> str:
    """Retrieve comprehensive stock data including company information, financials, trading metrics and governance.

    Returns JSON object with fields including:
    - Company: symbol, longName, sector, industry, longBusinessSummary, website, city, country
    - Price: currentPrice, previousClose, open, dayHigh, dayLow, fiftyTwoWeekHigh, fiftyTwoWeekLow
    - Valuation: marketCap, enterpriseValue, trailingPE, forwardPE, priceToBook, pegRatio
    - Trading: volume, averageVolume, averageVolume10days, bid, ask, bidSize, askSize
    - Dividends: dividendRate, dividendYield, exDividendDate, payoutRatio
    - Financials: totalRevenue, revenueGrowth, earningsGrowth, profitMargins, operatingMargins
    - Performance: beta, fiftyDayAverage, twoHundredDayAverage, trailingEps, forwardEps

    Note: Available fields vary by security type. Timestamps are converted to readable dates.
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        info = await asyncio.to_thread(lambda: ticker.info)
    except (ConnectionError, TimeoutError, OSError) as exc:
        return _error(
            f"Network error while fetching ticker info for '{symbol}'. Check your internet connection and try again.",
            error_code="NETWORK_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )
    except Exception as exc:
        return _error(
            f"Failed to fetch ticker info for '{symbol}'. Verify the symbol is correct and try again.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    if not info:
        return _error(
            f"No information available for symbol '{symbol}'. "
            "The symbol may be invalid or delisted. Try searching for the company "
            "name using the 'yfinance_search' tool to find the correct symbol.",
            error_code="INVALID_SYMBOL",
            details={"symbol": symbol},
        )

    # Convert timestamps to human-readable format when they look numeric.
    for key, value in list(info.items()):
        if not isinstance(key, str):
            continue

        if not isinstance(value, int | float):
            continue

        if key.lower().endswith(("date", "start", "end", "timestamp", "time", "quarter")):
            try:
                info[key] = datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as exc:
                logger.error("Unable to convert {}: {} to datetime: {}", key, value, exc)

    return _dump_json(info)


@mcp.tool(
    name="yfinance_get_ticker_news",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_ticker_news(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
) -> str:
    """Fetch recent news articles and press releases for a specific stock.

    Returns JSON array where each news item has:
    - id: Unique article identifier
    - content: Object containing:
        - title: Article headline
        - summary: Brief article summary
        - pubDate: Publication date (ISO 8601 format)
        - provider: Object with displayName (e.g., "Yahoo Finance") and url
        - canonicalUrl: Object with article url, site, region, lang
        - thumbnail: Object with image URLs and resolutions
        - contentType: Type of content (e.g., "STORY", "VIDEO")

    Use this to track company announcements, market sentiment, and breaking news.
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        news = await asyncio.to_thread(ticker.get_news)
    except (ConnectionError, TimeoutError, OSError) as exc:
        return _error(
            f"Network error while fetching news for '{symbol}'. Check your internet connection and try again.",
            error_code="NETWORK_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )
    except Exception as exc:
        return _error(
            f"Failed to fetch news for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    if not news:
        return _error(
            f"No news articles available for '{symbol}'. "
            "This may indicate an invalid symbol or no recent news coverage.",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    return _dump_json(news)


@mcp.tool(
    name="yfinance_search",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def search(
    query: Annotated[str, Field(description="Search query - company name, ticker symbol, or keywords")],
    search_type: Annotated[
        SearchType,
        Field(
            description="Filter results: 'all' (quotes + news), 'quotes' (stocks/ETFs only), or 'news' (articles only)"
        ),
    ],
) -> str:
    """Search Yahoo Finance for stocks, ETFs, and news articles.

    Returns JSON with search results based on search_type:

    - 'quotes': Array of securities with:
        - symbol: Ticker symbol
        - shortname/longname: Company name
        - quoteType: Security type (EQUITY, ETF, MUTUALFUND, etc.)
        - exchange: Exchange code
        - sector: Business sector
        - industry: Industry classification
        - score: Search relevance score

    - 'news': Array of articles with:
        - uuid: Article identifier
        - title: Headline
        - publisher: News source
        - link: Article URL
        - providerPublishTime: Unix timestamp
        - relatedTickers: Array of related symbols
        - thumbnail: Image URLs

    - 'all': Object with both 'quotes' and 'news' arrays

    Use this to find ticker symbols, discover related securities, or search financial news.
    """
    try:
        s = await asyncio.to_thread(yf.Search, query)
    except (ConnectionError, TimeoutError, OSError) as exc:
        return _error(
            f"Network error during search for '{query}'. Check your internet connection and try again.",
            error_code="NETWORK_ERROR",
            details={"query": query, "exception": str(exc)},
        )
    except Exception as exc:
        return _error(
            f"Search failed for '{query}'. Try simplifying your query or using different keywords.",
            error_code="API_ERROR",
            details={"query": query, "exception": str(exc)},
        )

    match search_type.lower():
        case "all":
            return _dump_json(s.all)
        case "quotes":
            return _dump_json(s.quotes)
        case "news":
            return _dump_json(s.news)
        case _:
            return _error(
                f"Invalid search_type '{search_type}'. Valid options: 'all', 'quotes', 'news'.",
                error_code="INVALID_PARAMS",
                details={"search_type": search_type, "valid_options": ["all", "quotes", "news"]},
            )


async def get_top_etfs(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top ETFs to retrieve", ge=1)],
) -> str:
    """Get the most popular ETFs for a specific sector.

    Returns JSON array where each ETF has:
    - symbol: ETF ticker symbol
    - name: Full ETF name
    """
    try:
        s = await asyncio.to_thread(yf.Sector, sector)
        etfs = await asyncio.to_thread(lambda: s.top_etfs)
    except (ConnectionError, TimeoutError, OSError) as exc:
        return _error(
            f"Network error while fetching top ETFs for '{sector}'. Check your internet connection and try again.",
            error_code="NETWORK_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )
    except Exception as exc:
        return _error(
            f"Failed to fetch top ETFs for '{sector}'. Verify the sector name is valid.",
            error_code="API_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )

    if not etfs:
        return _error(
            f"No ETF data available for sector '{sector}'.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    result = [{"symbol": symbol, "name": name} for symbol, name in list(etfs.items())[:top_n]]
    return _dump_json(result)


async def get_top_mutual_funds(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top mutual funds to retrieve", ge=1)],
) -> str:
    """Get the most popular mutual funds for a specific sector.

    Returns JSON array where each mutual fund has:
    - symbol: Fund ticker symbol
    - name: Full fund name
    """
    try:
        s = await asyncio.to_thread(yf.Sector, sector)
        funds = await asyncio.to_thread(lambda: s.top_mutual_funds)
    except (ConnectionError, TimeoutError, OSError) as exc:
        return _error(
            f"Network error while fetching top mutual funds for '{sector}'. "
            "Check your internet connection and try again.",
            error_code="NETWORK_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )
    except Exception as exc:
        return _error(
            f"Failed to fetch top mutual funds for '{sector}'. Verify the sector name is valid.",
            error_code="API_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )

    if not funds:
        return _error(
            f"No mutual fund data available for sector '{sector}'.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    result = [{"symbol": symbol, "name": name} for symbol, name in list(funds.items())[:top_n]]
    return _dump_json(result)


async def get_top_companies(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top companies to retrieve", ge=1)],
) -> str:
    """Get top companies in a sector by market capitalization.

    Returns JSON array with company data from Yahoo Finance sector data.
    Typically includes company identifiers, market metrics, and analyst information.
    """
    try:
        s = await asyncio.to_thread(yf.Sector, sector)
        df = await asyncio.to_thread(lambda: s.top_companies)
    except (ConnectionError, TimeoutError, OSError) as exc:
        return _error(
            f"Network error while fetching top companies for '{sector}'. Check your internet connection and try again.",
            error_code="NETWORK_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )
    except Exception as exc:
        return _error(
            f"Failed to fetch top companies for '{sector}'. Verify the sector name is valid.",
            error_code="API_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )

    if df is None or df.empty:
        return _error(
            f"No company data available for '{sector}'. This sector may not have enough listed companies.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    return _dump_json(df.head(top_n).to_dict(orient="records"))


async def get_top_growth_companies(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top growth companies per industry", ge=1)],
) -> str:
    """Get fastest-growing companies organized by industry within a sector.

    Returns JSON array grouped by industry. Each industry entry contains company data
    with growth-related metrics from Yahoo Finance.

    Results are organized by industry to show growth leaders across the sector.
    """
    try:
        industries = SECTOR_INDUSTY_MAPPING[sector]
    except KeyError:
        return _error(
            f"Unknown sector '{sector}'. Valid sectors: {', '.join(SECTOR_INDUSTY_MAPPING.keys())}",
            error_code="INVALID_PARAMS",
            details={"sector": sector, "valid_sectors": list(SECTOR_INDUSTY_MAPPING.keys())},
        )

    results = []
    for industry_name in industries:
        try:
            industry = await asyncio.to_thread(yf.Industry, industry_name)
        except Exception as exc:
            logger.warning("Failed to load industry {}: {}", industry_name, exc)
            continue

        df = await asyncio.to_thread(lambda i=industry: i.top_growth_companies)
        if df is None or df.empty:
            continue

        results.append(
            {
                "industry": industry_name,
                "top_growth_companies": df.head(top_n).to_dict(orient="records"),
            }
        )

    if not results:
        return _error(
            f"No growth company data available for '{sector}'. Try a different sector or check back later.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    return _dump_json(results)


async def get_top_performing_companies(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top performing companies per industry", ge=1)],
) -> str:
    """Get best-performing companies by stock price performance, organized by industry.

    Returns JSON array grouped by industry. Each industry entry contains company data
    with performance-related metrics from Yahoo Finance.

    Results are organized by industry to show top performers across the sector.
    """
    try:
        industries = SECTOR_INDUSTY_MAPPING[sector]
    except KeyError:
        return _error(
            f"Unknown sector '{sector}'. Valid sectors: {', '.join(SECTOR_INDUSTY_MAPPING.keys())}",
            error_code="INVALID_PARAMS",
            details={"sector": sector, "valid_sectors": list(SECTOR_INDUSTY_MAPPING.keys())},
        )

    results = []
    for industry_name in industries:
        try:
            industry = await asyncio.to_thread(yf.Industry, industry_name)
        except Exception as exc:
            logger.warning("Failed to load industry {}: {}", industry_name, exc)
            continue

        df = await asyncio.to_thread(lambda i=industry: i.top_performing_companies)
        if df is None or df.empty:
            continue

        results.append(
            {
                "industry": industry_name,
                "top_performing_companies": df.head(top_n).to_dict(orient="records"),
            }
        )

    if not results:
        return _error(
            f"No performance data available for '{sector}'. Try a different sector or check back later.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    return _dump_json(results)


@mcp.tool(
    name="yfinance_get_top",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_top(
    sector: Annotated[
        Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare', 'Financial Services')")
    ],
    top_type: Annotated[
        TopType,
        Field(
            description=(
                "Type of entities to retrieve: "
                "'top_etfs' (sector ETFs), "
                "'top_mutual_funds' (sector mutual funds), "
                "'top_companies' (largest by market cap), "
                "'top_growth_companies' (fastest revenue/earnings growth), "
                "'top_performing_companies' (best stock price performance)"
            )
        ),
    ],
    top_n: Annotated[
        int,
        Field(
            description="Number of top entities to retrieve per category/industry",
            ge=1,
            le=100,
        ),
    ] = 10,
) -> str:
    """Get top-ranked financial entities within a sector.

    This unified tool provides access to various rankings:
    - ETFs and mutual funds focused on the sector
    - Largest companies by market capitalization
    - Fastest-growing companies by revenue/earnings
    - Best-performing stocks by price appreciation

    Returns JSON data with relevant metrics for each entity type.
    """
    match top_type:
        case "top_etfs":
            return await get_top_etfs(sector, top_n)
        case "top_mutual_funds":
            return await get_top_mutual_funds(sector, top_n)
        case "top_companies":
            return await get_top_companies(sector, top_n)
        case "top_growth_companies":
            return await get_top_growth_companies(sector, top_n)
        case "top_performing_companies":
            return await get_top_performing_companies(sector, top_n)
        case _:
            return _error(
                f"Invalid top_type '{top_type}'. "
                "Valid options: 'top_etfs', 'top_mutual_funds', 'top_companies', "
                "'top_growth_companies', 'top_performing_companies'.",
                error_code="INVALID_PARAMS",
                details={
                    "top_type": top_type,
                    "valid_options": [
                        "top_etfs",
                        "top_mutual_funds",
                        "top_companies",
                        "top_growth_companies",
                        "top_performing_companies",
                    ],
                },
            )


@mcp.tool(
    name="yfinance_get_price_history",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_price_history(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    period: Annotated[
        Period,
        Field(
            description=(
                "Time range: '1d'/'5d' (days), '1mo'/'3mo'/'6mo' (months), "
                "'1y'/'2y'/'5y'/'10y' (years), 'ytd' (year-to-date), 'max' (all available data)"
            )
        ),
    ] = "1mo",
    interval: Annotated[
        Interval,
        Field(
            description=(
                "Data granularity: '1m'/'5m'/'15m'/'30m' (minutes), '1h' (hour), "
                "'1d'/'5d' (days), '1wk' (week), '1mo'/'3mo' (months). "
                "Short intervals require short periods (e.g., '1m' interval only works with '1d'/'5d' period)"
            )
        ),
    ] = "1d",
    chart_type: Annotated[
        ChartType | None,
        Field(
            description=(
                "Optional visualization: "
                "'price_volume' (candlestick chart with volume bars), "
                "'vwap' (Volume Weighted Average Price overlay), "
                "'volume_profile' (volume distribution by price level). "
                "Omit for tabular data"
            )
        ),
    ] = None,
) -> str | ImageContent:
    """Fetch historical price data and optionally generate technical analysis charts.

    When chart_type is None, returns Markdown table with columns:
    - Date: Trading date (index)
    - Open: Opening price
    - High: Highest price
    - Low: Lowest price
    - Close: Closing price
    - Volume: Trading volume
    - Dividends: Dividend payments (if any)
    - Stock Splits: Split events (if any)

    When chart_type is specified, returns a chart image:
    - 'price_volume': Candlestick chart with volume bars
    - 'vwap': Price with Volume Weighted Average Price overlay
    - 'volume_profile': Volume distribution by price level

    Note: Not all period/interval combinations are valid. Minute intervals (1m, 5m, etc.)
    only work with short periods (1d, 5d).
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        df = await asyncio.to_thread(
            ticker.history,
            period=period,
            interval=interval,
            rounding=True,
        )
    except (ConnectionError, TimeoutError, OSError) as exc:
        return _error(
            f"Network error while fetching price history for '{symbol}'. Check your internet connection and try again.",
            error_code="NETWORK_ERROR",
            details={
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "exception": str(exc),
            },
        )
    except Exception as exc:
        return _error(
            f"Failed to fetch price history for '{symbol}'. "
            "Verify the symbol is correct and the period/interval combination is valid.",
            error_code="API_ERROR",
            details={
                "symbol": symbol,
                "period": period,
                "interval": interval,
                "exception": str(exc),
            },
        )

    if df.empty:
        return _error(
            f"No price data available for '{symbol}' with period='{period}' and interval='{interval}'. "
            "Common issues: (1) Invalid symbol, (2) Incompatible period/interval combination "
            "(e.g., '1m' interval requires '1d' or '5d' period), (3) Market holidays or insufficient history. "
            "Try a longer period or daily interval.",
            error_code="NO_DATA",
            details={"symbol": symbol, "period": period, "interval": interval},
        )

    if chart_type is None:
        return df.to_markdown()

    return generate_chart(symbol=symbol, df=df, chart_type=chart_type)


def main() -> None:
    mcp.run()
