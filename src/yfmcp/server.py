import base64
import io
import json
from datetime import datetime
from typing import Annotated

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from yfinance.const import SECTOR_INDUSTY_MAPPING

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
    period: Annotated[Period, Field(description="Time period to retrieve data for (e.g. '1d', '1mo', '1y')")] = "1mo",
    interval: Annotated[Interval, Field(description="Data interval frequency (e.g. '1d', '1h', '1m')")] = "1d",
) -> str:
    """Fetch historical price data for a given stock symbol over a specified period and interval."""
    ticker = yf.Ticker(symbol)
    df = ticker.history(
        period=period,
        interval=interval,
        rounding=True,
    )
    return df.to_markdown()


def _calculate_volume_profile(df: pd.DataFrame, bins: int = 50) -> pd.Series:
    """Calculate volume profile by distributing volume across price levels."""
    price_min = df["Low"].min()
    price_max = df["High"].max()

    # Create price bins
    price_bins = np.linspace(price_min, price_max, bins + 1)
    price_centers = (price_bins[:-1] + price_bins[1:]) / 2

    # Initialize volume profile
    volume_profile = pd.Series(0.0, index=price_centers)

    # Distribute volume for each bar based on price range
    # Use itertuples() instead of iterrows() for better performance (~14x faster)
    for row in df.itertuples():
        low = row.Low
        high = row.High
        volume = row.Volume

        # Find bins that this bar overlaps with
        overlapping_bins = (price_centers >= low) & (price_centers <= high)

        if overlapping_bins.any():
            # Distribute volume proportionally based on overlap
            # Simple approach: distribute evenly across overlapping bins
            num_bins = overlapping_bins.sum()
            if num_bins > 0:
                volume_per_bin = volume / num_bins
                volume_profile[overlapping_bins] += volume_per_bin

    return volume_profile


@mcp.tool()
def get_chart(
    symbol: Annotated[str, Field(description="The stock symbol")],
    chart_type: Annotated[
        ChartType,
        Field(
            description=(
                "Type of chart: 'price_volume' for candlestick with volume bars, "
                "'vwap' for Volume Weighted Average Price, or 'volume_profile' "
                "for volume distribution by price level"
            )
        ),
    ] = "price_volume",
    period: Annotated[
        Period,
        Field(
            description=(
                "Time period to retrieve data for (e.g. '1d', '5d', '1mo'). "
                "For intraday charts, use '1d' or '5d'"
            )
        ),
    ] = "1d",
    interval: Annotated[
        Interval,
        Field(
            description=(
                "Data interval frequency. For day charts, use '1m', '2m', '5m', "
                "'15m', '30m', '60m', or '1h'"
            )
        ),
    ] = "5m",
) -> str:
    """Generate a financial chart using mplfinance.

    Shows candlestick price data with volume, optionally with VWAP or volume profile.
    Returns base64-encoded WebP image for efficient token usage.
    """
    try:
        ticker = yf.Ticker(symbol)

        # Get historical data
        df = ticker.history(
            period=period,
            interval=interval,
            rounding=True,
        )

        if df.empty:
            error_msg = (
                f"No data available for symbol {symbol} "
                f"with period {period} and interval {interval}"
            )
            return json.dumps({"error": error_msg})

        # Prepare data for mplfinance (needs OHLCV columns)
        # Ensure column names match what mplfinance expects
        df = df[["Open", "High", "Low", "Close", "Volume"]]

        # Handle volume profile separately as it needs custom layout
        if chart_type == "volume_profile":
            # Calculate volume profile
            volume_profile = _calculate_volume_profile(df)

            # Create a custom figure with proper layout for side-by-side charts
            fig = plt.figure(figsize=(18, 10))

            # Create gridspec for layout: left side for candlestick+volume, right side for volume profile
            gs = fig.add_gridspec(2, 2, width_ratios=[3.5, 1], height_ratios=[3, 1],
                                 hspace=0.3, wspace=0.15, left=0.08, right=0.95, top=0.95, bottom=0.1)

            # Left side: candlestick chart (top) and volume bars (bottom)
            ax_price = fig.add_subplot(gs[0, 0])
            ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_price)

            # Right side: volume profile (aligned with price chart)
            ax_profile = fig.add_subplot(gs[0, 1], sharey=ax_price)

            # Plot candlestick and volume using mplfinance on our custom axes
            style = mpf.make_mpf_style(base_mpf_style="yahoo", rc={"figure.facecolor": "white"})
            mpf.plot(
                df,
                type="candle",
                volume=ax_volume,
                style=style,
                ax=ax_price,
                show_nontrading=False,
                returnfig=False,
            )

            # Plot volume profile as horizontal bars on the right
            colors = plt.cm.viridis(np.linspace(0, 1, len(volume_profile)))
            ax_profile.barh(volume_profile.index, volume_profile.values, color=colors, alpha=0.7)
            ax_profile.set_xlabel("Volume", fontsize=10)
            ax_profile.set_title("Volume Profile", fontsize=12, fontweight="bold", pad=10)
            ax_profile.grid(True, alpha=0.3, axis="x")
            ax_profile.set_ylabel("")  # Share y-axis label with main chart

            # Set overall title
            fig.suptitle(f"{symbol} - Volume Profile", fontsize=16, fontweight="bold", y=0.98)

            # Save directly to WebP format
            buf = io.BytesIO()
            fig.savefig(buf, format="webp", dpi=150, bbox_inches="tight")
            buf.seek(0)
            plt.close(fig)

        else:
            # Standard mplfinance chart (price_volume or vwap)
            addplots = []
            if chart_type == "vwap":
                # VWAP = Sum(Price * Volume) / Sum(Volume)
                typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
                vwap = (typical_price * df["Volume"]).cumsum() / df["Volume"].cumsum()
                addplots.append(
                    mpf.make_addplot(vwap, color="orange", width=2, linestyle="--", label="VWAP")
                )

            # Create style
            style = mpf.make_mpf_style(base_mpf_style="yahoo", rc={"figure.facecolor": "white"})

            # Save chart directly to WebP format
            buf = io.BytesIO()
            plot_kwargs = {
                "type": "candle",
                "volume": True,
                "style": style,
                "title": f"{symbol} - {chart_type.replace('_', ' ').title()}",
                "ylabel": "Price",
                "ylabel_lower": "Volume",
                "savefig": {"fname": buf, "format": "webp", "dpi": 150, "bbox_inches": "tight"},
                "show_nontrading": False,
                "returnfig": False,
            }
            if addplots:
                plot_kwargs["addplot"] = addplots

            mpf.plot(df, **plot_kwargs)
            buf.seek(0)

        # Encode WebP to base64 for efficient transmission
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")

        # Return as JSON with base64-encoded WebP
        return json.dumps({
            "image_base64": img_base64,
            "symbol": symbol,
            "chart_type": chart_type,
            "period": period,
            "interval": interval,
            "data_points": len(df),
            "format": "webp",
        }, ensure_ascii=False)

    except Exception as e:
        logger.error("Error generating chart for {}: {}", symbol, e)
        return json.dumps({"error": f"Failed to generate chart: {str(e)}"})


def main() -> None:
    mcp.run()
