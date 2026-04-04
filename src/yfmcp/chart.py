import base64
import io

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend (must be set before pyplot is imported)

import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mcp.types import ImageContent

from yfmcp.types import ChartType

# Chart configuration constants
DEFAULT_VOLUME_PROFILE_BINS = 50  # Number of price bins for volume profile histogram
DEFAULT_CHART_DPI = 150  # Image resolution - balance between quality and file size
DEFAULT_CHART_FIGSIZE = (18, 10)  # Figure size (width, height) in inches, currently used for volume_profile charts
VOLUME_PROFILE_WIDTH_RATIOS = [3.5, 1]  # Chart width ratios: [price chart, volume profile]
VOLUME_PROFILE_HEIGHT_RATIOS = [3, 1]  # Chart height ratios: [price chart, volume bars]
VOLUME_PROFILE_HSPACE = 0.3  # Vertical spacing between subplots
VOLUME_PROFILE_WSPACE = 0.15  # Horizontal spacing between subplots
VOLUME_PROFILE_MARGINS = {  # Figure margins
    "left": 0.08,
    "right": 0.95,
    "top": 0.95,
    "bottom": 0.1,
}
VWAP_LINE_WIDTH = 2  # VWAP line thickness in points
VOLUME_PROFILE_ALPHA = 0.7  # Transparency for volume profile bars (0=transparent, 1=opaque)
UP_CANDLE_COLOR = "#2ca02c"
DOWN_CANDLE_COLOR = "#d62728"
CANDLE_WIDTH = 0.6
VOLUME_ALPHA = 0.8
MAX_X_TICKS = 8


def _calculate_volume_profile(df: pd.DataFrame, bins: int = DEFAULT_VOLUME_PROFILE_BINS) -> pd.Series:
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


def _compute_x_ticks(index: pd.Index, max_ticks: int = MAX_X_TICKS) -> tuple[np.ndarray, list[str]]:
    """Return evenly distributed tick positions and labels for chart x-axis."""
    total = len(index)
    if total == 0:
        return np.array([], dtype=int), []

    tick_count = min(max_ticks, total)
    positions = np.linspace(0, total - 1, tick_count, dtype=int)
    positions = np.unique(positions)

    if isinstance(index, pd.DatetimeIndex):
        labels = index[positions].strftime("%Y-%m-%d").tolist()
    else:
        labels = [str(index[pos]) for pos in positions]

    return positions, labels


def _plot_candlesticks(ax, df: pd.DataFrame, x: np.ndarray) -> np.ndarray:
    """Plot candlesticks on target axes and return per-bar colors."""
    opens = df["Open"].to_numpy(dtype=float)
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    closes = df["Close"].to_numpy(dtype=float)

    is_up = closes >= opens
    colors = np.where(is_up, UP_CANDLE_COLOR, DOWN_CANDLE_COLOR)

    ax.vlines(x, lows, highs, color=colors, linewidth=1.0, zorder=1)

    body_bottom = np.minimum(opens, closes)
    body_height = np.abs(closes - opens)
    # Keep near-flat candles visible.
    min_body_height = max((highs.max() - lows.min()) * 1e-4, 1e-6)
    body_height = np.maximum(body_height, min_body_height)

    ax.bar(
        x,
        body_height,
        width=CANDLE_WIDTH,
        bottom=body_bottom,
        color=colors,
        edgecolor=colors,
        linewidth=0.8,
        zorder=2,
    )

    return colors


def generate_chart(symbol: str, df: pd.DataFrame, chart_type: ChartType) -> ImageContent | str:
    """Generate a financial chart using matplotlib.

    Shows candlestick price data with volume, optionally with VWAP or volume profile.
    Returns base64-encoded WebP image for efficient token usage.
    """

    # Ensure required OHLCV columns exist and preserve order.
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    x = np.arange(len(df))
    tick_positions, tick_labels = _compute_x_ticks(df.index)

    # Handle volume profile separately as it needs custom layout
    if chart_type == "volume_profile":
        # Calculate volume profile
        volume_profile = _calculate_volume_profile(df)

        # Create a custom figure with proper layout for side-by-side charts
        fig = plt.figure(figsize=DEFAULT_CHART_FIGSIZE)

        # Create gridspec for layout: left side for candlestick+volume, right side for volume profile
        gs = gridspec.GridSpec(
            2,
            2,
            figure=fig,
            width_ratios=VOLUME_PROFILE_WIDTH_RATIOS,
            height_ratios=VOLUME_PROFILE_HEIGHT_RATIOS,
            hspace=VOLUME_PROFILE_HSPACE,
            wspace=VOLUME_PROFILE_WSPACE,
            **VOLUME_PROFILE_MARGINS,
        )

        # Left side: candlestick chart (top) and volume bars (bottom)
        ax_price = fig.add_subplot(gs[0, 0])
        ax_volume = fig.add_subplot(gs[1, 0], sharex=ax_price)

        # Right side: volume profile (aligned with price chart)
        ax_profile = fig.add_subplot(gs[0, 1], sharey=ax_price)

        # Plot candlestick and matching volume bars
        candle_colors = _plot_candlesticks(ax_price, df, x)
        ax_volume.bar(x, df["Volume"], width=CANDLE_WIDTH, color=candle_colors, alpha=VOLUME_ALPHA)
        ax_volume.set_ylabel("Volume")
        ax_volume.grid(True, alpha=0.3, axis="y")

        # Plot volume profile as horizontal bars on the right
        viridis = cm.get_cmap("viridis")
        colors = viridis(np.linspace(0, 1, len(volume_profile)))
        ax_profile.barh(volume_profile.index, volume_profile.values, color=colors, alpha=VOLUME_PROFILE_ALPHA)
        ax_profile.set_xlabel("Volume", fontsize=10)
        ax_profile.set_title("Volume Profile", fontsize=12, fontweight="bold", pad=10)
        ax_profile.grid(True, alpha=0.3, axis="x")
        ax_profile.set_ylabel("")  # Share y-axis label with main chart

        # Set overall title
        fig.suptitle(f"{symbol} - Volume Profile", fontsize=16, fontweight="bold", y=0.98)
        ax_price.set_ylabel("Price")
        ax_price.grid(True, alpha=0.3)
        ax_price.set_xlim(-0.5, len(df) - 0.5)
        if len(tick_positions) > 0:
            ax_volume.set_xticks(tick_positions)
            ax_volume.set_xticklabels(tick_labels, rotation=45, ha="right")
        plt.setp(ax_price.get_xticklabels(), visible=False)

        # Save directly to WebP format
        buf = io.BytesIO()
        fig.savefig(buf, format="webp", dpi=DEFAULT_CHART_DPI, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

    else:
        # Standard chart (price_volume or vwap)
        fig, (ax_price, ax_volume) = plt.subplots(
            2,
            1,
            figsize=DEFAULT_CHART_FIGSIZE,
            sharex=True,
            gridspec_kw={"height_ratios": VOLUME_PROFILE_HEIGHT_RATIOS, "hspace": 0.05},
        )
        fig.subplots_adjust(**VOLUME_PROFILE_MARGINS)

        candle_colors = _plot_candlesticks(ax_price, df, x)
        ax_volume.bar(x, df["Volume"], width=CANDLE_WIDTH, color=candle_colors, alpha=VOLUME_ALPHA)
        ax_volume.set_ylabel("Volume")
        ax_volume.grid(True, alpha=0.3, axis="y")
        ax_price.set_ylabel("Price")
        ax_price.grid(True, alpha=0.3)

        if chart_type == "vwap":
            # VWAP = Sum(Price * Volume) / Sum(Volume)
            typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
            cumulative_volume = df["Volume"].cumsum().replace(0, np.nan)
            vwap = (typical_price * df["Volume"]).cumsum() / cumulative_volume
            ax_price.plot(
                x, vwap.to_numpy(dtype=float), color="orange", linewidth=VWAP_LINE_WIDTH, linestyle="--", label="VWAP"
            )
            ax_price.legend(loc="upper left")

        ax_price.set_title(f"{symbol} - {chart_type.replace('_', ' ').title()}")
        ax_price.set_xlim(-0.5, len(df) - 0.5)
        if len(tick_positions) > 0:
            ax_volume.set_xticks(tick_positions)
            ax_volume.set_xticklabels(tick_labels, rotation=45, ha="right")
        plt.setp(ax_price.get_xticklabels(), visible=False)

        # Save chart directly to WebP format
        buf = io.BytesIO()
        fig.savefig(buf, format="webp", dpi=DEFAULT_CHART_DPI, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

    return ImageContent(
        type="image",
        data=base64.b64encode(buf.read()).decode("utf-8"),
        mimeType="image/webp",
    )
