from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from yfmcp.chart import generate_chart
from yfmcp.types import ChartType


def _build_sample_price_data(days: int = 120) -> pd.DataFrame:
    """Create deterministic OHLCV sample data for chart preview."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2025-01-01", periods=days, freq="D")

    close = 120 + np.cumsum(rng.normal(loc=0.15, scale=1.8, size=days))
    open_ = np.r_[close[0], close[:-1]] + rng.normal(loc=0.0, scale=0.8, size=days)
    high = np.maximum(open_, close) + rng.uniform(0.4, 2.6, size=days)
    low = np.minimum(open_, close) - rng.uniform(0.4, 2.6, size=days)
    volume = rng.integers(900_000, 6_500_000, size=days)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a sample WebP chart using yfmcp chart renderer.")
    parser.add_argument(
        "--chart-type",
        choices=["price_volume", "vwap", "volume_profile"],
        default="price_volume",
        help="Chart type to generate.",
    )
    parser.add_argument(
        "--output",
        default="sample-chart.webp",
        help="Output file path (WebP).",
    )
    parser.add_argument(
        "--symbol",
        default="AAPL",
        help="Symbol text shown in chart title.",
    )
    args = parser.parse_args()

    df = _build_sample_price_data()
    image = generate_chart(args.symbol, df, cast(ChartType, args.chart_type))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(base64.b64decode(image.data))

    print(f"Generated {args.chart_type} chart: {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
