"""Unit tests for chart generation functions."""

import base64

import pandas as pd
import pytest

from yfmcp.chart import DEFAULT_CHART_DPI
from yfmcp.chart import DEFAULT_VOLUME_PROFILE_BINS
from yfmcp.chart import _calculate_volume_profile
from yfmcp.chart import generate_chart


@pytest.fixture
def sample_price_data() -> pd.DataFrame:
    """Create sample price data for testing."""
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "High": [102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0],
            "Close": [101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0, 110.0],
            "Volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        },
        index=dates,
    )


def test_calculate_volume_profile(sample_price_data: pd.DataFrame) -> None:
    """Test volume profile calculation."""
    volume_profile = _calculate_volume_profile(sample_price_data, bins=10)

    # Check that volume profile is a Series
    assert isinstance(volume_profile, pd.Series)

    # Check that volume profile has the correct number of bins
    assert len(volume_profile) == 10

    # Check that total volume is distributed (should be positive)
    assert volume_profile.sum() > 0

    # Check that all values are non-negative
    assert (volume_profile >= 0).all()


def test_calculate_volume_profile_custom_bins(sample_price_data: pd.DataFrame) -> None:
    """Test volume profile with custom number of bins."""
    volume_profile = _calculate_volume_profile(sample_price_data, bins=20)
    assert len(volume_profile) == 20


def test_calculate_volume_profile_default_bins(sample_price_data: pd.DataFrame) -> None:
    """Test volume profile with default bins."""
    volume_profile = _calculate_volume_profile(sample_price_data)
    assert len(volume_profile) == DEFAULT_VOLUME_PROFILE_BINS


def test_generate_chart_volume_profile(sample_price_data: pd.DataFrame) -> None:
    """Test volume profile chart generation."""
    result = generate_chart("AAPL", sample_price_data, "volume_profile")

    # Check that result is ImageContent
    assert hasattr(result, "type")
    assert result.type == "image"
    assert hasattr(result, "mimeType")
    assert result.mimeType == "image/webp"

    # Validate that a real WebP image was produced
    raw = base64.b64decode(result.data)
    assert len(raw) > 0
    assert raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"


def test_generate_chart_price_volume(sample_price_data: pd.DataFrame) -> None:
    """Test price_volume chart generation."""
    result = generate_chart("AAPL", sample_price_data, "price_volume")

    # Check that result is ImageContent
    assert hasattr(result, "type")
    assert result.type == "image"
    assert hasattr(result, "mimeType")
    assert result.mimeType == "image/webp"
    assert isinstance(result.data, str)
    assert len(result.data) > 0
    assert len(base64.b64decode(result.data)) > 0


def test_generate_chart_vwap(sample_price_data: pd.DataFrame) -> None:
    """Test VWAP chart generation."""
    result = generate_chart("AAPL", sample_price_data, "vwap")

    # Check that result is ImageContent
    assert hasattr(result, "type")
    assert result.type == "image"
    assert isinstance(result.data, str)
    assert len(result.data) > 0
    assert len(base64.b64decode(result.data)) > 0


def test_chart_constants() -> None:
    """Test that chart constants are defined with reasonable values."""
    assert DEFAULT_VOLUME_PROFILE_BINS > 0
    assert DEFAULT_VOLUME_PROFILE_BINS == 50

    assert DEFAULT_CHART_DPI > 0
    assert DEFAULT_CHART_DPI == 150
