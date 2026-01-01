"""Unit tests for server.py functions with mocks."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pandas as pd
import pytest

from yfmcp.server import get_top_companies
from yfmcp.server import get_top_etfs
from yfmcp.server import get_top_growth_companies
from yfmcp.server import get_top_mutual_funds
from yfmcp.server import get_top_performing_companies


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Sector")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_top_etfs_success(mock_to_thread: AsyncMock, mock_sector: MagicMock) -> None:
    """Test successful ETF retrieval."""
    # Mock the yfinance Sector object
    mock_sector_obj = MagicMock()
    mock_sector_obj.top_etfs = {"SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ Trust"}

    # Setup asyncio.to_thread mock
    async def mock_thread_func(func, *args):
        if callable(func):
            return func(*args)
        return mock_sector_obj

    mock_to_thread.side_effect = mock_thread_func
    mock_sector.return_value = mock_sector_obj

    result = await get_top_etfs("Technology", 2)
    data = json.loads(result)

    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["symbol"] == "SPY"
    assert data[0]["name"] == "SPDR S&P 500 ETF"


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Sector")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_top_etfs_no_data(mock_to_thread: AsyncMock, mock_sector: MagicMock) -> None:
    """Test ETF retrieval with no data."""
    mock_sector_obj = MagicMock()
    mock_sector_obj.top_etfs = {}

    async def mock_thread_func(func, *args):
        if callable(func):
            return func(*args)
        return mock_sector_obj

    mock_to_thread.side_effect = mock_thread_func
    mock_sector.return_value = mock_sector_obj

    result = await get_top_etfs("Technology", 2)
    data = json.loads(result)

    assert "error" in data
    assert data["error_code"] == "NO_DATA"
    assert "details" in data


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Sector")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_top_etfs_api_error(mock_to_thread: AsyncMock, mock_sector: MagicMock) -> None:
    """Test ETF retrieval with API error."""
    mock_to_thread.side_effect = Exception("API Error")

    result = await get_top_etfs("Technology", 2)
    data = json.loads(result)

    assert "error" in data
    assert data["error_code"] == "API_ERROR"
    assert "details" in data
    assert "exception" in data["details"]


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Sector")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_top_mutual_funds_success(mock_to_thread: AsyncMock, mock_sector: MagicMock) -> None:
    """Test successful mutual fund retrieval."""
    mock_sector_obj = MagicMock()
    mock_sector_obj.top_mutual_funds = {"FXAIX": "Fidelity 500 Index", "VTSAX": "Vanguard Total Stock"}

    async def mock_thread_func(func, *args):
        if callable(func):
            return func(*args)
        return mock_sector_obj

    mock_to_thread.side_effect = mock_thread_func
    mock_sector.return_value = mock_sector_obj

    result = await get_top_mutual_funds("Technology", 2)
    data = json.loads(result)

    assert isinstance(data, list)
    assert len(data) == 2


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Sector")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_top_companies_success(mock_to_thread: AsyncMock, mock_sector: MagicMock) -> None:
    """Test successful company retrieval."""
    mock_sector_obj = MagicMock()
    mock_df = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT", "GOOGL"],
            "name": ["Apple", "Microsoft", "Google"],
            "marketCap": [2000000000000, 1800000000000, 1500000000000],
        }
    )
    mock_sector_obj.top_companies = mock_df

    async def mock_thread_func(func, *args):
        if callable(func):
            return func(*args)
        return mock_sector_obj

    mock_to_thread.side_effect = mock_thread_func
    mock_sector.return_value = mock_sector_obj

    result = await get_top_companies("Technology", 3)
    data = json.loads(result)

    assert isinstance(data, list)
    assert len(data) == 3
    assert data[0]["symbol"] == "AAPL"


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Sector")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_top_companies_empty_dataframe(mock_to_thread: AsyncMock, mock_sector: MagicMock) -> None:
    """Test company retrieval with empty dataframe."""
    mock_sector_obj = MagicMock()
    mock_sector_obj.top_companies = pd.DataFrame()

    async def mock_thread_func(func, *args):
        if callable(func):
            return func(*args)
        return mock_sector_obj

    mock_to_thread.side_effect = mock_thread_func
    mock_sector.return_value = mock_sector_obj

    result = await get_top_companies("Technology", 3)
    data = json.loads(result)

    assert "error" in data
    assert data["error_code"] == "NO_DATA"


@pytest.mark.asyncio
@patch("yfmcp.server.SECTOR_INDUSTY_MAPPING", {"Technology": ["Software", "Hardware"]})
@patch("yfmcp.server.yf.Industry")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_top_growth_companies_success(mock_to_thread: AsyncMock, mock_industry: MagicMock) -> None:
    """Test successful growth company retrieval."""
    mock_industry_obj = MagicMock()
    mock_df = pd.DataFrame(
        {
            "symbol": ["NVDA", "AMD"],
            "name": ["NVIDIA", "AMD"],
            "growth": [50.0, 45.0],
        }
    )
    mock_industry_obj.top_growth_companies = mock_df

    async def mock_thread_func(func, *args):
        if callable(func):
            return func(*args)
        return mock_industry_obj

    mock_to_thread.side_effect = mock_thread_func
    mock_industry.return_value = mock_industry_obj

    result = await get_top_growth_companies("Technology", 2)
    data = json.loads(result)

    assert isinstance(data, list)
    assert len(data) > 0
    assert "industry" in data[0]
    assert "top_growth_companies" in data[0]


@pytest.mark.asyncio
async def test_get_top_growth_companies_invalid_sector() -> None:
    """Test growth company retrieval with invalid sector."""
    result = await get_top_growth_companies("InvalidSector", 2)  # ty:ignore[invalid-argument-type]
    data = json.loads(result)

    assert "error" in data
    assert data["error_code"] == "INVALID_PARAMS"
    assert "valid_sectors" in data["details"]


@pytest.mark.asyncio
@patch("yfmcp.server.SECTOR_INDUSTY_MAPPING", {"Technology": ["Software"]})
@patch("yfmcp.server.yf.Industry")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_top_performing_companies_success(mock_to_thread: AsyncMock, mock_industry: MagicMock) -> None:
    """Test successful performing company retrieval."""
    mock_industry_obj = MagicMock()
    mock_df = pd.DataFrame(
        {
            "symbol": ["TSLA"],
            "name": ["Tesla"],
            "performance": [100.0],
        }
    )
    mock_industry_obj.top_performing_companies = mock_df

    async def mock_thread_func(func, *args):
        if callable(func):
            return func(*args)
        return mock_industry_obj

    mock_to_thread.side_effect = mock_thread_func
    mock_industry.return_value = mock_industry_obj

    result = await get_top_performing_companies("Technology", 1)
    data = json.loads(result)

    assert isinstance(data, list)
    assert len(data) > 0


@pytest.mark.asyncio
@patch("yfmcp.server.SECTOR_INDUSTY_MAPPING", {"Technology": []})
async def test_get_top_growth_companies_no_industries() -> None:
    """Test growth company retrieval with no industries."""
    result = await get_top_growth_companies("Technology", 2)
    data = json.loads(result)

    assert "error" in data
    assert data["error_code"] == "NO_DATA"
