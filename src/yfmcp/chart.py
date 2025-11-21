import base64
import io

import numpy as np
import pandas as pd
from mcp.types import ImageContent

from yfmcp.types import ChartType


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


def generate_chart(symbol: str, df: pd.DataFrame, chart_type: ChartType) -> ImageContent | str:
    """Generate a financial chart using mplfinance.

    Shows candlestick price data with volume, optionally with VWAP or volume profile.
    Returns base64-encoded WebP image for efficient token usage.
    """

    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend

    import matplotlib.cm as cm
    import matplotlib.pyplot as plt
    import mplfinance as mpf

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
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[3.5, 1],
            height_ratios=[3, 1],
            hspace=0.3,
            wspace=0.15,
            left=0.08,
            right=0.95,
            top=0.95,
            bottom=0.1,
        )

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
        viridis = cm.get_cmap("viridis")
        colors = viridis(np.linspace(0, 1, len(volume_profile)))
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
            addplots.append(mpf.make_addplot(vwap, color="orange", width=2, linestyle="--", label="VWAP"))

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

    return ImageContent(
        type="image",
        data=base64.b64encode(buf.read()).decode("utf-8"),
        mimeType="image/webp",
    )
