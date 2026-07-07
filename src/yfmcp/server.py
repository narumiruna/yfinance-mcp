import asyncio
from datetime import datetime
from typing import Annotated
from typing import Any

import yfinance as yf
from loguru import logger
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from mcp.types import ToolAnnotations
from pydantic import Field
from yfinance.const import SECTOR_INDUSTY_MAPPING
from yfinance.exceptions import YFInvalidPeriodError
from yfinance.exceptions import YFPricesMissingError
from yfinance.exceptions import YFRateLimitError
from yfinance.exceptions import YFTzMissingError

from yfmcp.chart import generate_chart
from yfmcp.screener import build_screener_query
from yfmcp.types import ChartType
from yfmcp.types import Interval
from yfmcp.types import OptionChainType
from yfmcp.types import Period
from yfmcp.types import ScreenerQueryType
from yfmcp.types import SearchType
from yfmcp.types import Sector
from yfmcp.types import TopType
from yfmcp.utils import create_error_response
from yfmcp.utils import dump_json

# https://github.com/jlowin/fastmcp/issues/81#issuecomment-2714245145
mcp = FastMCP("yfinance_mcp", log_level="ERROR")


_RETRYABLE_YFINANCE_EXCEPTIONS: tuple[type[Exception], ...] = (
    ConnectionError,
    TimeoutError,
    OSError,
    YFRateLimitError,
)


def _is_retryable_yfinance_error(exc: BaseException) -> bool:
    return isinstance(exc, _RETRYABLE_YFINANCE_EXCEPTIONS)


def _is_rate_limit_error(exc: BaseException) -> bool:
    return isinstance(exc, YFRateLimitError)


def _create_retryable_error_response(action: str, exc: BaseException, details: dict[str, Any]) -> str:
    if _is_rate_limit_error(exc):
        message = f"Rate limit reached while {action}. Try again later."
    else:
        message = f"Temporary network issue while {action}. Try again later."

    return create_error_response(message, error_code="NETWORK_ERROR", details={**details, "exception": str(exc)})


def _price_history_details(
    symbol: str,
    period: Period,
    interval: Interval,
    prepost: bool,
    exc: BaseException | None = None,
) -> dict[str, Any]:
    details: dict[str, Any] = {"symbol": symbol, "period": period, "interval": interval, "prepost": prepost}
    if exc is not None:
        details["exception"] = str(exc)
    return details


def _create_price_history_no_data_response(
    symbol: str,
    period: Period,
    interval: Interval,
    prepost: bool,
    exc: BaseException | None = None,
) -> str:
    return create_error_response(
        f"No price data available for '{symbol}' with period='{period}' and interval='{interval}'. "
        "Common issues: (1) Invalid symbol, (2) Incompatible period/interval combination "
        "(e.g., '1m' interval requires '1d' or '5d' period), (3) Market holidays or insufficient history. "
        "Try a longer period or daily interval.",
        error_code="NO_DATA",
        details=_price_history_details(symbol, period, interval, prepost, exc),
    )


def _create_price_history_api_error_response(
    symbol: str,
    period: Period,
    interval: Interval,
    prepost: bool,
    exc: BaseException,
) -> str:
    return create_error_response(
        f"Failed to fetch price history for '{symbol}'. "
        "Verify the symbol is correct and the period/interval combination is valid.",
        error_code="API_ERROR",
        details=_price_history_details(symbol, period, interval, prepost, exc),
    )


def _is_price_history_rate_limit_message(message: str) -> bool:
    normalized = message.lower()
    return any(
        indicator in normalized
        for indicator in (
            "too many requests",
            "rate limit",
            "rate-limit",
            "ratelimit",
            "status_code = 429",
            "status_code=429",
        )
    )


def _is_price_history_no_data_prices_missing_error(exc: YFPricesMissingError) -> bool:
    debug_info = str(getattr(exc, "debug_info", "") or "").lower()

    if "yahoo status_code" in debug_info:
        return False

    if "yahoo error" in debug_info:
        return any(
            indicator in debug_info
            for indicator in (
                "no data found",
                "symbol may be delisted",
                "possibly delisted",
            )
        )

    return True


def _create_price_history_prices_missing_error_response(
    symbol: str,
    period: Period,
    interval: Interval,
    prepost: bool,
    exc: YFPricesMissingError,
) -> str:
    rate_limit_context = f"{getattr(exc, 'debug_info', '')} {getattr(exc, 'rationale', '')}"
    if _is_price_history_rate_limit_message(rate_limit_context):
        return create_error_response(
            f"Rate limit reached while fetching price history for '{symbol}'. Try again later.",
            error_code="NETWORK_ERROR",
            details=_price_history_details(symbol, period, interval, prepost, exc),
        )

    if _is_price_history_no_data_prices_missing_error(exc):
        return _create_price_history_no_data_response(symbol, period, interval, prepost, exc)

    return _create_price_history_api_error_response(symbol, period, interval, prepost, exc)


def _select_retryable_exception(exceptions: list[Exception]) -> BaseException:
    rate_limit_exception = next((exc for exc in exceptions if _is_rate_limit_error(exc)), None)
    return rate_limit_exception or exceptions[0]


def _create_option_dates_fetch_error(symbol: str, exc: Exception, api_message: str) -> str:
    if _is_retryable_yfinance_error(exc):
        return _create_retryable_error_response(f"fetching option dates for '{symbol}'", exc, {"symbol": symbol})

    return create_error_response(
        api_message,
        error_code="API_ERROR",
        details={"symbol": symbol, "exception": str(exc)},
    )


def _create_option_chain_fetch_error(
    symbol: str,
    dates_to_fetch: list[str],
    fetch_errors: list[tuple[str, Exception]],
) -> str:
    failed_dates = [date for date, _ in fetch_errors]

    if len(dates_to_fetch) == 1:
        failed_date, exc = fetch_errors[0]
        if _is_retryable_yfinance_error(exc):
            return _create_retryable_error_response(
                f"fetching option chain for '{symbol}' on '{failed_date}'",
                exc,
                {"symbol": symbol, "expiration_date": failed_date},
            )

        return create_error_response(
            f"Failed to fetch option chain for '{symbol}' on '{failed_date}'.",
            error_code="API_ERROR",
            details={"symbol": symbol, "expiration_date": failed_date, "exception": str(exc)},
        )

    retryable_exceptions = [exc for _, exc in fetch_errors if _is_retryable_yfinance_error(exc)]

    if retryable_exceptions:
        return _create_retryable_error_response(
            f"fetching option chain for '{symbol}'",
            _select_retryable_exception(retryable_exceptions),
            {
                "symbol": symbol,
                "dates_requested": dates_to_fetch,
                "failed_dates": failed_dates,
            },
        )

    representative_exception = fetch_errors[0][1]
    return create_error_response(
        f"Failed to fetch option chain for '{symbol}' for all requested dates.",
        error_code="API_ERROR",
        details={
            "symbol": symbol,
            "dates_requested": dates_to_fetch,
            "failed_dates": failed_dates,
            "exception": str(representative_exception),
        },
    )


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
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching ticker info for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch ticker info for '{symbol}'. Verify the symbol is correct and try again.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    if not info:
        return create_error_response(
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

    return dump_json(info)


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
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching news for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch news for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    if not news:
        return create_error_response(
            f"No news articles available for '{symbol}'. "
            "This may indicate an invalid symbol or no recent news coverage.",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    return dump_json(news)


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
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"searching for '{query}'", exc, {"query": query})
    except Exception as exc:
        return create_error_response(
            f"Search failed for '{query}'. Try simplifying your query or using different keywords.",
            error_code="API_ERROR",
            details={"query": query, "exception": str(exc)},
        )

    match search_type.lower():
        case "all":
            return dump_json(s.all)
        case "quotes":
            return dump_json(s.quotes)
        case "news":
            return dump_json(s.news)
        case _:
            return create_error_response(
                f"Invalid search_type '{search_type}'. Valid options: 'all', 'quotes', 'news'.",
                error_code="INVALID_PARAMS",
                details={"search_type": search_type, "valid_options": ["all", "quotes", "news"]},
            )


@mcp.tool(
    name="yfinance_screen",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def screen(
    query: Annotated[
        str | dict[str, Any],
        Field(
            description=(
                "Screener query. For query_type='predefined': string key like 'day_gainers'. "
                "For query_type='equity' or 'fund': query tree object with {operator, operands} nodes."
            )
        ),
    ],
    query_type: Annotated[
        ScreenerQueryType,
        Field(description="Query mode: 'predefined', 'equity', or 'fund'."),
    ] = "predefined",
    offset: Annotated[int | None, Field(description="Result offset.", ge=0)] = None,
    size: Annotated[
        int | None,
        Field(description="Rows to return for custom queries. Yahoo maximum is 250.", ge=1, le=250),
    ] = None,
    count: Annotated[
        int | None,
        Field(description="Rows to return for predefined queries. Yahoo maximum is 250.", ge=1, le=250),
    ] = None,
    sort_field: Annotated[str | None, Field(description="Sort field, for example 'percentchange'.")] = None,
    sort_asc: Annotated[bool | None, Field(description="Sort ascending if true, descending if false.")] = None,
    user_id: Annotated[str | None, Field(description="Optional Yahoo user id.")] = None,
    user_id_type: Annotated[str | None, Field(description="Optional Yahoo user id type, commonly 'guid'.")] = None,
) -> str:
    """Run a Yahoo Finance screener query.

    Supports predefined Yahoo screener keys and custom equity or fund query trees.
    """
    try:
        if query_type == "predefined" and size is not None:
            return create_error_response(
                "For query_type='predefined', use count instead of size.",
                error_code="INVALID_PARAMS",
                details={"query_type": query_type, "invalid_parameter": "size", "expected_parameter": "count"},
            )
        if query_type in {"equity", "fund"} and count is not None:
            return create_error_response(
                "For query_type='equity' or 'fund', use size instead of count.",
                error_code="INVALID_PARAMS",
                details={"query_type": query_type, "invalid_parameter": "count", "expected_parameter": "size"},
            )

        if query_type == "predefined":
            if not isinstance(query, str):
                return create_error_response(
                    "For query_type='predefined', query must be a string screener key.",
                    error_code="INVALID_PARAMS",
                    details={"query_type": query_type, "expected_query_type": "string"},
                )

            predefined = getattr(yf, "PREDEFINED_SCREENER_QUERIES", {})
            if query not in predefined:
                return create_error_response(
                    f"Unknown predefined screener '{query}'.",
                    error_code="INVALID_PARAMS",
                    details={
                        "query": query,
                        "query_type": query_type,
                        "valid_predefined_queries": sorted(predefined.keys()),
                    },
                )

            resolved_query: str | Any = query
        else:
            if not isinstance(query, dict):
                return create_error_response(
                    "For query_type='equity' or 'fund', query must be an object with 'operator' and 'operands'.",
                    error_code="INVALID_PARAMS",
                    details={"query_type": query_type, "expected_query_type": "object"},
                )

            resolved_query = build_screener_query(query_type=query_type, query=query)

        upstream_size = count if query_type == "predefined" else size
        result = await asyncio.to_thread(
            yf.screen,
            resolved_query,
            offset=offset,
            size=upstream_size,
            count=None,
            sortField=sort_field,
            sortAsc=sort_asc,
            userId=user_id,
            userIdType=user_id_type,
        )
    except (TypeError, ValueError) as exc:
        return create_error_response(
            "Invalid screener query. Check operators, operands, and field values for the selected query_type.",
            error_code="INVALID_PARAMS",
            details={"query_type": query_type, "exception": str(exc)},
        )
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response("running screener query", exc, {"query_type": query_type})
    except Exception as exc:
        return create_error_response(
            "Failed to run screener query.",
            error_code="API_ERROR",
            details={"query_type": query_type, "exception": str(exc)},
        )

    return dump_json(result)


@mcp.tool(
    name="yfinance_screen_gappers",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def screen_gappers(
    min_percent_change: Annotated[
        float,
        Field(description="Minimum percent change from prior close, for example 3.0 for +3%.", ge=0),
    ] = 3.0,
    min_price: Annotated[
        float,
        Field(description="Minimum current intraday price.", ge=0),
    ] = 5.0,
    min_volume: Annotated[
        int,
        Field(description="Minimum intraday trading volume.", ge=0),
    ] = 500000,
    min_market_cap: Annotated[
        int,
        Field(description="Minimum intraday market cap in USD.", ge=0),
    ] = 2000000000,
    region: Annotated[
        str,
        Field(description="Yahoo screener region code, for example 'us'."),
    ] = "us",
    size: Annotated[
        int,
        Field(description="Rows to return. Yahoo maximum is 250.", ge=1, le=250),
    ] = 50,
    offset: Annotated[
        int,
        Field(description="Result offset for pagination.", ge=0),
    ] = 0,
    sort_asc: Annotated[
        bool,
        Field(description="Sort by percentchange ascending if true, descending if false."),
    ] = False,
) -> str:
    """Run a custom equity screener tuned for opening-session stock gappers."""
    query = {
        "operator": "and",
        "operands": [
            {"operator": "gte", "operands": ["percentchange", min_percent_change]},
            {"operator": "eq", "operands": ["region", region]},
            {"operator": "gte", "operands": ["intradaymarketcap", min_market_cap]},
            {"operator": "gte", "operands": ["intradayprice", min_price]},
            {"operator": "gte", "operands": ["dayvolume", min_volume]},
        ],
    }

    try:
        resolved_query = build_screener_query(query_type="equity", query=query)
        result = await asyncio.to_thread(
            yf.screen,
            resolved_query,
            offset=offset,
            size=size,
            sortField="percentchange",
            sortAsc=sort_asc,
        )
    except (TypeError, ValueError) as exc:
        return create_error_response(
            "Invalid gappers screener parameters.",
            error_code="INVALID_PARAMS",
            details={"exception": str(exc)},
        )
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response("running gappers screener", exc, {})
    except Exception as exc:
        return create_error_response(
            "Failed to run gappers screener.",
            error_code="API_ERROR",
            details={"exception": str(exc)},
        )

    return dump_json(result)


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
        s = await asyncio.to_thread(yf.Sector, _sector_key(sector))
        etfs = await asyncio.to_thread(lambda: s.top_etfs)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching top ETFs for '{sector}'", exc, {"sector": sector})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch top ETFs for '{sector}'. Verify the sector name is valid.",
            error_code="API_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )

    if not etfs:
        return create_error_response(
            f"No ETF data available for sector '{sector}'.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    result = [{"symbol": symbol, "name": name} for symbol, name in list(etfs.items())[:top_n]]
    return dump_json(result)


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
        s = await asyncio.to_thread(yf.Sector, _sector_key(sector))
        funds = await asyncio.to_thread(lambda: s.top_mutual_funds)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(
            f"fetching top mutual funds for '{sector}'",
            exc,
            {"sector": sector},
        )
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch top mutual funds for '{sector}'. Verify the sector name is valid.",
            error_code="API_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )

    if not funds:
        return create_error_response(
            f"No mutual fund data available for sector '{sector}'.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    result = [{"symbol": symbol, "name": name} for symbol, name in list(funds.items())[:top_n]]
    return dump_json(result)


async def get_top_companies(
    sector: Annotated[Sector, Field(description="Market sector (e.g., 'Technology', 'Healthcare')")],
    top_n: Annotated[int, Field(description="Number of top companies to retrieve", ge=1)],
) -> str:
    """Get top companies in a sector by market capitalization.

    Returns JSON array with company data from Yahoo Finance sector data.
    Typically includes company identifiers, market metrics, and analyst information.
    """
    try:
        s = await asyncio.to_thread(yf.Sector, _sector_key(sector))
        df = await asyncio.to_thread(lambda: s.top_companies)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching top companies for '{sector}'", exc, {"sector": sector})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch top companies for '{sector}'. Verify the sector name is valid.",
            error_code="API_ERROR",
            details={"sector": sector, "exception": str(exc)},
        )

    if df is None or df.empty:
        return create_error_response(
            f"No company data available for '{sector}'. This sector may not have enough listed companies.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    return dump_json(df.head(top_n).to_dict(orient="records"))


def _sector_key(name: str) -> str:
    """Convert human-readable sector name to Yahoo Finance API key format."""
    return name.lower().replace(" ", "-")


def _industry_key(name: str) -> str:
    """Convert human-readable industry name to Yahoo Finance API key format.

    SECTOR_INDUSTY_MAPPING uses em dashes (—) and title case,
    but the API expects lowercase with regular hyphens.
    """
    return name.lower().replace("& ", "").replace("- ", "").replace(", ", " ").replace("—", "-").replace(" ", "-")


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
        return create_error_response(
            f"Unknown sector '{sector}'. Valid sectors: {', '.join(SECTOR_INDUSTY_MAPPING.keys())}",
            error_code="INVALID_PARAMS",
            details={"sector": sector, "valid_sectors": list(SECTOR_INDUSTY_MAPPING.keys())},
        )

    results = []
    for industry_name in industries:
        try:
            industry = await asyncio.to_thread(yf.Industry, _industry_key(industry_name))
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
        return create_error_response(
            f"No growth company data available for '{sector}'. Try a different sector or check back later.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    return dump_json(results)


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
        return create_error_response(
            f"Unknown sector '{sector}'. Valid sectors: {', '.join(SECTOR_INDUSTY_MAPPING.keys())}",
            error_code="INVALID_PARAMS",
            details={"sector": sector, "valid_sectors": list(SECTOR_INDUSTY_MAPPING.keys())},
        )

    results = []
    for industry_name in industries:
        try:
            industry = await asyncio.to_thread(yf.Industry, _industry_key(industry_name))
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
        return create_error_response(
            f"No performance data available for '{sector}'. Try a different sector or check back later.",
            error_code="NO_DATA",
            details={"sector": sector},
        )

    return dump_json(results)


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
            return create_error_response(
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
    prepost: Annotated[
        bool,
        Field(description="Include pre-market and post-market data when available"),
    ] = False,
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

    Set prepost=True to include pre-market and post-market data when available.

    Note: Not all period/interval combinations are valid. Minute intervals (1m, 5m, etc.)
    only work with short periods (1d, 5d).
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        df = await asyncio.to_thread(
            ticker.history,
            period=period,
            interval=interval,
            prepost=prepost,
            rounding=True,
            raise_errors=True,
        )
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(
            f"fetching price history for '{symbol}'",
            exc,
            _price_history_details(symbol, period, interval, prepost),
        )
    except (YFTzMissingError, YFInvalidPeriodError) as exc:
        return _create_price_history_no_data_response(symbol, period, interval, prepost, exc)
    except YFPricesMissingError as exc:
        return _create_price_history_prices_missing_error_response(symbol, period, interval, prepost, exc)
    except Exception as exc:
        return _create_price_history_api_error_response(symbol, period, interval, prepost, exc)

    if df.empty:
        return _create_price_history_no_data_response(symbol, period, interval, prepost)

    if chart_type is None:
        return df.to_markdown()

    return generate_chart(symbol=symbol, df=df, chart_type=chart_type)


@mcp.tool(
    name="yfinance_get_financials",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_financials(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    frequency: Annotated[
        str,
        Field(
            description=(
                "Reporting frequency: 'annual' for yearly, 'quarterly' for quarterly, "
                "or 'ttm' for trailing twelve months"
            )
        ),
    ] = "annual",
) -> str:
    """Fetch financial statements (income statement, balance sheet, and cash flow) with historical data.

    Returns JSON with income statement, balance sheet, and cash flow data across reporting periods.

    Use the data to analyze trends, calculate ratios, or compare periods.
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching financials for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch financials for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    income_stmt = None
    balance_sheet = None
    cash_flow = None

    if frequency not in {"annual", "quarterly", "ttm"}:
        return create_error_response(
            f"Invalid frequency '{frequency}'. Valid options: 'annual', 'quarterly', 'ttm'.",
            error_code="INVALID_PARAMS",
            details={"frequency": frequency, "valid_options": ["annual", "quarterly", "ttm"]},
        )

    try:
        if frequency == "annual":
            income_stmt = await asyncio.to_thread(lambda: ticker.income_stmt)
            balance_sheet = await asyncio.to_thread(lambda: ticker.balance_sheet)
            cash_flow = await asyncio.to_thread(lambda: ticker.cashflow)
        elif frequency == "quarterly":
            income_stmt = await asyncio.to_thread(lambda: ticker.quarterly_income_stmt)
            balance_sheet = await asyncio.to_thread(lambda: ticker.quarterly_balance_sheet)
            cash_flow = await asyncio.to_thread(lambda: ticker.quarterly_cashflow)
        else:
            income_stmt = await asyncio.to_thread(lambda: ticker.ttm_income_stmt)
            balance_sheet = None  # TTM balance sheet not directly available
            cash_flow = None  # TTM cash flow not directly available

        result = _build_financials_response(income_stmt, balance_sheet, cash_flow)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(
            f"fetching financials for '{symbol}'",
            exc,
            {"symbol": symbol, "frequency": frequency},
        )
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch financials for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "frequency": frequency, "exception": str(exc)},
        )
    if not result:
        return create_error_response(
            f"No financial data available for '{symbol}' with frequency='{frequency}'.",
            error_code="NO_DATA",
            details={"symbol": symbol, "frequency": frequency},
        )

    return dump_json(result)


def _build_financials_response(income_stmt, balance_sheet, cash_flow=None) -> dict:
    """Build financials response from income statement, balance sheet, and cash flow DataFrames."""
    result = {}

    if income_stmt is not None and not income_stmt.empty:
        income_fields = [
            "EBIT",
            "Net Income",
            "Tax Provision",
            "Pretax Income",
            "Interest Expense",
            "Total Revenue",
            "Operating Income",
            "EBITDA",
            "Normalized Income",
        ]
        available_income_fields = [f for f in income_fields if f in income_stmt.index]
        result["income_statement"] = {}
        for field in available_income_fields:
            result["income_statement"][field] = {
                str(col.date()): income_stmt.loc[field, col] for col in income_stmt.columns
            }

    if balance_sheet is not None and not balance_sheet.empty:
        balance_fields = [
            "Stockholders Equity",
            "Total Debt",
            "Cash And Cash Equivalents",
            "Invested Capital",
            "Net Debt",
            "Total Assets",
            "Total Liabilities Net Minority Interest",
            "Net Tangible Assets",
            "Tangible Book Value",
        ]
        available_balance_fields = [f for f in balance_fields if f in balance_sheet.index]
        result["balance_sheet"] = {}
        for field in available_balance_fields:
            result["balance_sheet"][field] = {
                str(col.date()): balance_sheet.loc[field, col] for col in balance_sheet.columns
            }

    if cash_flow is not None and not cash_flow.empty:
        cash_flow_fields = [
            "Operating Cash Flow",
            "Free Cash Flow",
            "Capital Expenditure",
            "Net Income From Continuing Operations",
            "Depreciation And Amortization",
            "Change In Working Capital",
            "Cash Dividends Paid",
        ]
        available_cash_flow_fields = [f for f in cash_flow_fields if f in cash_flow.index]
        result["cash_flow"] = {}
        for field in available_cash_flow_fields:
            result["cash_flow"][field] = {str(col.date()): cash_flow.loc[field, col] for col in cash_flow.columns}

    return result


async def _fetch_option_chain_for_date(
    ticker: yf.Ticker,
    date: str,
    option_type: OptionChainType,
) -> dict[str, Any]:
    """Fetch option chain for a single expiration date."""
    opt = await asyncio.to_thread(lambda d=date: ticker.option_chain(d))

    calls_df = opt.calls
    puts_df = opt.puts
    date_data: dict[str, Any] = {}

    if calls_df is not None and not calls_df.empty and option_type in {"all", "calls"}:
        calls_df = calls_df.copy()
        calls_df["optionType"] = "CALL"
        date_data["calls"] = calls_df.to_dict(orient="records")

    if puts_df is not None and not puts_df.empty and option_type in {"all", "puts"}:
        puts_df = puts_df.copy()
        puts_df["optionType"] = "PUT"
        date_data["puts"] = puts_df.to_dict(orient="records")

    return {date: date_data} if date_data else {}


@mcp.tool(
    name="yfinance_get_option_chain",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_option_chain(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    expiration_date: Annotated[
        str | None,
        Field(
            description=(
                "Option expiration date in YYYY-MM-DD format. "
                "Use the 'yfinance_get_option_dates' tool to find available dates, "
                "or omit to fetch all available expiration dates."
            )
        ),
    ] = None,
    option_type: Annotated[
        OptionChainType,
        Field(description=("Which options to return: 'calls', 'puts', or 'all' (both calls and puts).")),
    ] = "all",
) -> str:
    """Fetch option chain data (calls and puts) for a stock with available strike prices.

    Returns JSON with calls and/or puts data for each expiration date.

    JSON fields include:
    - contractSymbol: Option contract identifier
    - strike: Strike price
    - lastPrice: Last traded price
    - bid/ask: Bid and ask prices
    - volume: Trading volume
    - openInterest: Open interest
    - impliedVolatility: Implied volatility (IV)
    - inTheMoney: Whether option is ITM
    - contractSize: Contract size (REGULAR)
    - currency: Currency (USD)

    Use this to analyze options pricing, IV surfaces, and strike levels.
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching options for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch options for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    try:
        available_dates = await asyncio.to_thread(lambda: ticker.options)
    except Exception as exc:
        return _create_option_dates_fetch_error(
            symbol,
            exc,
            f"Failed to fetch option dates for '{symbol}'. The symbol may not have options.",
        )

    if not available_dates:
        return create_error_response(
            f"No options available for symbol '{symbol}'. "
            "This symbol may not have listed options (e.g., ETFs, stocks without options).",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    if expiration_date is not None and expiration_date not in available_dates:
        return create_error_response(
            f"Invalid expiration date '{expiration_date}' for '{symbol}'. Valid dates: {', '.join(available_dates)}",
            error_code="INVALID_PARAMS",
            details={
                "symbol": symbol,
                "expiration_date": expiration_date,
                "valid_dates": available_dates,
            },
        )

    dates_to_fetch = [expiration_date] if expiration_date else list(available_dates)
    result: dict[str, Any] = {}
    fetch_errors: list[tuple[str, Exception]] = []

    for date in dates_to_fetch:
        try:
            date_result = await _fetch_option_chain_for_date(ticker, date, option_type)
        except Exception as exc:
            logger.warning("Failed to fetch option chain for {} {}: {}", symbol, date, exc)
            fetch_errors.append((date, exc))
            continue
        result.update(date_result)

    if result:
        return dump_json(result)

    if fetch_errors:
        return _create_option_chain_fetch_error(symbol, dates_to_fetch, fetch_errors)

    return create_error_response(
        f"No option data retrieved for '{symbol}'.",
        error_code="NO_DATA",
        details={"symbol": symbol, "dates_requested": list(dates_to_fetch)},
    )


@mcp.tool(
    name="yfinance_get_option_dates",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_option_dates(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
) -> str:
    """Fetch available option expiration dates for a stock.

    Returns JSON array of expiration dates in YYYY-MM-DD format.

    Use these dates with the 'yfinance_get_option_chain' tool to fetch
    the options chain for a specific date.
    """
    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
        dates = await asyncio.to_thread(lambda: ticker.options)
    except Exception as exc:
        return _create_option_dates_fetch_error(
            symbol,
            exc,
            f"Failed to fetch option dates for '{symbol}'. Verify the symbol is correct.",
        )

    if not dates:
        return create_error_response(
            f"No options available for symbol '{symbol}'. "
            "This symbol may not have listed options (e.g., ETFs, stocks without options).",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    return dump_json(dates)


async def _fetch_holder_section(
    symbol: str,
    ticker: yf.Ticker,
    attr_name: str,
    result_key: str,
    result: dict[str, Any],
    section_metadata: dict[str, dict[str, int | bool]],
    fetch_errors: list[Exception],
    max_rows: int,
) -> None:
    """Fetch a single holder data section, adding successful data to result and failures to fetch_errors."""
    try:
        df = await asyncio.to_thread(lambda t=ticker: getattr(t, attr_name))
    except Exception as exc:
        logger.warning("Failed to fetch {} for {}: {}", attr_name, symbol, exc)
        fetch_errors.append(exc)
        return
    if df is not None and not df.empty:
        if attr_name == "major_holders":
            df = df.reset_index()

        total_rows = len(df)
        limited_df = df if max_rows == 0 else df.head(max_rows)
        limited_records = limited_df.to_dict(orient="records")
        result[result_key] = limited_records
        section_metadata[result_key] = {
            "total_rows": total_rows,
            "returned_rows": len(limited_records),
            "truncated": len(limited_records) < total_rows,
        }


@mcp.tool(
    name="yfinance_get_holders",
    annotations=ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=True,
    ),
)
async def get_holders(
    symbol: Annotated[str, Field(description="Stock ticker symbol (e.g., 'AAPL', 'GOOGL', 'MSFT')")],
    max_rows: Annotated[
        int,
        Field(description="Maximum rows returned per holder section. Use 0 to return all rows."),
    ] = 10,
) -> str:
    """Fetch major holders, institutional holders, mutual fund holders, and insider data.

    Returns JSON with:
    - major_holders: Aggregated breakdown including insider % held, institutional % held,
      institutional % float held, and institution count.
    - institutional_holders: List of institutional investors with shares held, date reported,
      value, and % change.
    - mutualfund_holders: List of mutual fund holders with same fields.
    - insider_transactions: Recent insider transactions including shares, value, transaction
      type, and date.
    - insider_purchases: Summary of insider buy/sell activity over the last 6 months.
    - insider_roster: List of known insiders by name and position.

    Use this to analyze ownership concentration, insider activity, and institutional interest.
    """
    if max_rows < 0:
        return create_error_response(
            "max_rows must be greater than or equal to 0.",
            error_code="INVALID_PARAMS",
            details={"max_rows": max_rows},
        )

    try:
        ticker = await asyncio.to_thread(yf.Ticker, symbol)
    except _RETRYABLE_YFINANCE_EXCEPTIONS as exc:
        return _create_retryable_error_response(f"fetching holders for '{symbol}'", exc, {"symbol": symbol})
    except Exception as exc:
        return create_error_response(
            f"Failed to fetch holders for '{symbol}'. Verify the symbol is correct.",
            error_code="API_ERROR",
            details={"symbol": symbol, "exception": str(exc)},
        )

    result: dict[str, Any] = {}
    section_metadata: dict[str, dict[str, int | bool]] = {}
    fetch_errors: list[Exception] = []
    await _fetch_holder_section(
        symbol, ticker, "major_holders", "major_holders", result, section_metadata, fetch_errors, max_rows
    )
    await _fetch_holder_section(
        symbol,
        ticker,
        "institutional_holders",
        "institutional_holders",
        result,
        section_metadata,
        fetch_errors,
        max_rows,
    )
    await _fetch_holder_section(
        symbol,
        ticker,
        "mutualfund_holders",
        "mutualfund_holders",
        result,
        section_metadata,
        fetch_errors,
        max_rows,
    )
    await _fetch_holder_section(
        symbol,
        ticker,
        "insider_transactions",
        "insider_transactions",
        result,
        section_metadata,
        fetch_errors,
        max_rows,
    )
    await _fetch_holder_section(
        symbol, ticker, "insider_purchases", "insider_purchases", result, section_metadata, fetch_errors, max_rows
    )
    await _fetch_holder_section(
        symbol,
        ticker,
        "insider_roster_holders",
        "insider_roster",
        result,
        section_metadata,
        fetch_errors,
        max_rows,
    )

    if not result:
        retryable_exceptions = [exc for exc in fetch_errors if _is_retryable_yfinance_error(exc)]
        if retryable_exceptions:
            return _create_retryable_error_response(
                f"fetching holders for '{symbol}'",
                _select_retryable_exception(retryable_exceptions),
                {"symbol": symbol},
            )

        return create_error_response(
            f"No holder data available for '{symbol}'. Verify the symbol is correct.",
            error_code="NO_DATA",
            details={"symbol": symbol},
        )

    result["_metadata"] = {"max_rows": max_rows, "sections": section_metadata}
    return dump_json(result)


def main() -> None:
    mcp.run()
