"""Unit tests for server.py functions with mocks."""

import json
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

import pandas as pd
import pytest
from yfinance.exceptions import YFRateLimitError

from yfmcp.server import _build_financials_response
from yfmcp.server import get_financials
from yfmcp.server import get_holders
from yfmcp.server import get_option_chain
from yfmcp.server import get_option_dates
from yfmcp.server import get_price_history
from yfmcp.server import get_top_companies
from yfmcp.server import get_top_etfs
from yfmcp.server import get_top_growth_companies
from yfmcp.server import get_top_mutual_funds
from yfmcp.server import get_top_performing_companies


def _financials_df(rows: dict[str, list[int]]) -> pd.DataFrame:
    """Build a yfinance-shaped financial statement DataFrame."""
    return pd.DataFrame(
        rows,
        index=[pd.Timestamp("2024-12-31"), pd.Timestamp("2023-12-31")],
        dtype=object,
    ).T


async def _run_to_thread(func, *args, **kwargs):
    if callable(func):
        return func(*args, **kwargs)
    return func


def _expected_retryable_error(action: str, exception: Exception) -> str:
    if isinstance(exception, YFRateLimitError):
        return f"Rate limit reached while {action}. Try again later."
    return f"Temporary network issue while {action}. Try again later."


class _FinancialsReadErrorTicker:
    @property
    def income_stmt(self):
        raise RuntimeError("statement read failed")


def test_build_financials_response_with_all_sections() -> None:
    """Test building a financials response with all supported statement sections."""
    income_stmt = _financials_df(
        {
            "Total Revenue": [1000, 900],
            "Net Income": [120, 100],
            "Unsupported Income Row": [1, 2],
        }
    )
    balance_sheet = _financials_df(
        {
            "Total Assets": [5000, 4500],
            "Total Debt": [800, 750],
            "Unsupported Balance Row": [3, 4],
        }
    )
    cash_flow = _financials_df(
        {
            "Operating Cash Flow": [300, 280],
            "Free Cash Flow": [200, 180],
            "Unsupported Cash Flow Row": [5, 6],
        }
    )

    result = _build_financials_response(income_stmt, balance_sheet, cash_flow)

    assert set(result) == {"income_statement", "balance_sheet", "cash_flow"}
    assert result["income_statement"]["Total Revenue"]["2024-12-31"] == 1000
    assert result["income_statement"]["Net Income"]["2023-12-31"] == 100
    assert "Unsupported Income Row" not in result["income_statement"]
    assert result["balance_sheet"]["Total Assets"]["2024-12-31"] == 5000
    assert result["balance_sheet"]["Total Debt"]["2023-12-31"] == 750
    assert "Unsupported Balance Row" not in result["balance_sheet"]
    assert result["cash_flow"]["Operating Cash Flow"]["2024-12-31"] == 300
    assert result["cash_flow"]["Free Cash Flow"]["2023-12-31"] == 180
    assert "Unsupported Cash Flow Row" not in result["cash_flow"]


def test_build_financials_response_ignores_none_and_empty_dataframes() -> None:
    """Test that missing or empty statements do not produce response sections."""
    empty_df = pd.DataFrame()
    income_stmt = _financials_df({"EBIT": [100, 90]})

    partial_result = _build_financials_response(income_stmt, None, empty_df)
    empty_result = _build_financials_response(None, empty_df, None)

    assert set(partial_result) == {"income_statement"}
    assert partial_result["income_statement"]["EBIT"]["2024-12-31"] == 100
    assert empty_result == {}


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_financials_annual_success(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test annual financials retrieval."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.income_stmt = _financials_df({"Total Revenue": [1000, 900]})
    mock_ticker_obj.balance_sheet = _financials_df({"Total Assets": [5000, 4500]})
    mock_ticker_obj.cashflow = _financials_df({"Operating Cash Flow": [300, 280]})
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_financials("AAPL", "annual")
    data = json.loads(result)

    assert data["income_statement"]["Total Revenue"]["2024-12-31"] == 1000
    assert data["balance_sheet"]["Total Assets"]["2023-12-31"] == 4500
    assert data["cash_flow"]["Operating Cash Flow"]["2024-12-31"] == 300


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_financials_quarterly_success(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test quarterly financials retrieval."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.quarterly_income_stmt = _financials_df({"Operating Income": [400, 350]})
    mock_ticker_obj.quarterly_balance_sheet = _financials_df({"Stockholders Equity": [2200, 2100]})
    mock_ticker_obj.quarterly_cashflow = _financials_df({"Free Cash Flow": [150, 125]})
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_financials("MSFT", "quarterly")
    data = json.loads(result)

    assert data["income_statement"]["Operating Income"]["2024-12-31"] == 400
    assert data["balance_sheet"]["Stockholders Equity"]["2023-12-31"] == 2100
    assert data["cash_flow"]["Free Cash Flow"]["2024-12-31"] == 150


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_financials_ttm_success_only_income_statement(
    mock_to_thread: AsyncMock, mock_ticker: MagicMock
) -> None:
    """Test TTM financials retrieval only returns income statement data."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.ttm_income_stmt = _financials_df({"EBITDA": [700, 650]})
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_financials("NVDA", "ttm")
    data = json.loads(result)

    assert set(data) == {"income_statement"}
    assert data["income_statement"]["EBITDA"]["2024-12-31"] == 700


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_financials_invalid_frequency(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test invalid financials frequency returns structured parameter error."""
    mock_ticker.return_value = MagicMock()
    mock_to_thread.side_effect = _run_to_thread

    result = await get_financials("AAPL", "monthly")
    data = json.loads(result)

    assert data["error_code"] == "INVALID_PARAMS"
    assert data["details"]["frequency"] == "monthly"
    assert data["details"]["valid_options"] == ["annual", "quarterly", "ttm"]


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_financials_no_data(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test financials retrieval with no statement data."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.income_stmt = pd.DataFrame()
    mock_ticker_obj.balance_sheet = pd.DataFrame()
    mock_ticker_obj.cashflow = pd.DataFrame()
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_financials("EMPTY", "annual")
    data = json.loads(result)

    assert data["error_code"] == "NO_DATA"
    assert data["details"] == {"symbol": "EMPTY", "frequency": "annual"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception",
    [TimeoutError("timed out"), OSError("network unreachable"), YFRateLimitError()],
)
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_financials_ticker_creation_network_error(
    mock_to_thread: AsyncMock, mock_ticker: MagicMock, exception: Exception
) -> None:
    """Test network errors while creating a ticker return structured network errors."""
    mock_ticker.side_effect = exception
    mock_to_thread.side_effect = _run_to_thread

    result = await get_financials("AAPL", "annual")
    data = json.loads(result)

    assert data["error_code"] == "NETWORK_ERROR"
    assert data["error"] == _expected_retryable_error("fetching financials for 'AAPL'", exception)
    assert data["details"]["symbol"] == "AAPL"
    assert data["details"]["exception"] == str(exception)


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_financials_statement_read_api_error(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test statement read errors return structured API errors."""
    mock_ticker.return_value = _FinancialsReadErrorTicker()
    mock_to_thread.side_effect = _run_to_thread

    result = await get_financials("AAPL", "annual")
    data = json.loads(result)

    assert data["error_code"] == "API_ERROR"
    assert data["details"]["symbol"] == "AAPL"
    assert data["details"]["frequency"] == "annual"
    assert data["details"]["exception"] == "statement read failed"


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_price_history_returns_markdown_table_when_chart_type_is_none(
    mock_to_thread: AsyncMock, mock_ticker: MagicMock
) -> None:
    """Test price history without chart_type returns DataFrame markdown output."""
    df = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [110.0],
            "Low": [95.0],
            "Close": [105.0],
            "Volume": [1_000_000],
        },
        index=[pd.Timestamp("2024-01-02")],
    )
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.history.return_value = df
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_price_history("AAPL", "1mo", "1d", None)

    assert isinstance(result, str)
    assert result == df.to_markdown()
    assert "Open" in result
    assert "Close" in result
    assert "|" in result


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


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_dates_success(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test successful option dates retrieval."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.options = ["2025-05-02", "2025-05-09", "2025-05-16"]
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_dates("AAPL")
    data = json.loads(result)

    assert isinstance(data, list)
    assert len(data) == 3
    assert "2025-05-02" in data


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_dates_no_options(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test option dates with stock that has no options."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.options = []
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_dates("SPY")
    data = json.loads(result)

    assert "error" in data
    assert data["error_code"] == "NO_DATA"


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [TimeoutError("timed out"), YFRateLimitError()])
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_dates_options_network_error(
    mock_to_thread: AsyncMock, mock_ticker: MagicMock, exception: Exception
) -> None:
    """Test network errors while reading ticker.options in get_option_dates."""
    mock_ticker_obj = MagicMock()
    type(mock_ticker_obj).options = PropertyMock(side_effect=exception)
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_dates("AAPL")
    data = json.loads(result)

    assert data["error_code"] == "NETWORK_ERROR"
    assert data["error"] == _expected_retryable_error("fetching option dates for 'AAPL'", exception)


def _option_df() -> pd.DataFrame:
    """Build a yfinance-shaped option DataFrame."""
    return pd.DataFrame(
        {
            "contractSymbol": ["AAPL250515C00150000", "AAPL250515C00160000"],
            "strike": [150.0, 160.0],
            "lastPrice": [10.5, 5.2],
            "bid": [10.3, 5.0],
            "ask": [10.7, 5.4],
            "volume": [100, 50],
            "openInterest": [200, 100],
            "impliedVolatility": [0.30, 0.35],
            "inTheMoney": [True, False],
            "contractSize": ["REGULAR", "REGULAR"],
            "currency": ["USD", "USD"],
        }
    )


class _MockOptionChain:
    def __init__(self):
        self.calls = _option_df()
        self.puts = _option_df()


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_success_all(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test successful option chain retrieval for all dates and types."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.options = ["2025-05-02", "2025-05-09"]
    mock_opt = _MockOptionChain()
    mock_ticker_obj.option_chain.return_value = mock_opt
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_chain("AAPL")
    data = json.loads(result)

    assert "2025-05-02" in data
    assert "calls" in data["2025-05-02"]
    assert "puts" in data["2025-05-02"]
    assert data["2025-05-02"]["calls"][0]["optionType"] == "CALL"
    assert data["2025-05-02"]["puts"][0]["optionType"] == "PUT"


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_specific_date(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test option chain with specific expiration date."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.options = ["2025-05-02", "2025-05-09"]
    mock_opt = _MockOptionChain()
    mock_ticker_obj.option_chain.return_value = mock_opt
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_chain("AAPL", expiration_date="2025-05-02")
    data = json.loads(result)

    assert "2025-05-02" in data
    assert len(data) == 1


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_invalid_date(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test option chain with invalid expiration date."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.options = ["2025-05-02", "2025-05-09"]
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_chain("AAPL", expiration_date="2025-06-01")
    data = json.loads(result)

    assert "error" in data
    assert data["error_code"] == "INVALID_PARAMS"
    assert "valid_dates" in data["details"]


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_no_options(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test option chain with stock that has no options."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.options = []
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_chain("SPY")
    data = json.loads(result)

    assert "error" in data
    assert data["error_code"] == "NO_DATA"


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_calls_only(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test option chain returning calls only."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.options = ["2025-05-02"]
    mock_opt = _MockOptionChain()
    mock_ticker_obj.option_chain.return_value = mock_opt
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_chain("AAPL", option_type="calls")
    data = json.loads(result)

    assert "2025-05-02" in data
    assert "calls" in data["2025-05-02"]
    assert "puts" not in data["2025-05-02"]


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_puts_only_calls_empty(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test scenario where calls are empty but puts are present."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.options = ["2025-05-02"]

    mock_opt = MagicMock()
    mock_opt.calls = pd.DataFrame()
    mock_opt.puts = pd.DataFrame(
        {
            "contractSymbol": ["AAPL250515P00150000"],
            "strike": [150.0],
        }
    )

    mock_ticker_obj.option_chain.return_value = mock_opt
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_chain("AAPL", option_type="puts")
    data = json.loads(result)

    assert "2025-05-02" in data
    assert "puts" in data["2025-05-02"]
    assert "calls" not in data["2025-05-02"]


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_no_matching_type(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test scenario where requested type (e.g. calls) has no data for any date."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.options = ["2025-05-02"]

    mock_opt = MagicMock()
    mock_opt.calls = pd.DataFrame()  # No calls
    mock_opt.puts = pd.DataFrame({"strike": [150.0]})  # Has puts

    mock_ticker_obj.option_chain.return_value = mock_opt
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    # Request calls only, but only puts exist
    result = await get_option_chain("AAPL", option_type="calls")
    data = json.loads(result)

    assert "error" in data
    assert data["error_code"] == "NO_DATA"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception",
    [TimeoutError("timed out"), OSError("network unreachable"), YFRateLimitError()],
)
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_dates_network_error(
    mock_to_thread: AsyncMock, mock_ticker: MagicMock, exception: Exception
) -> None:
    """Test network errors while fetching option dates."""
    mock_ticker.side_effect = exception
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_dates("AAPL")
    data = json.loads(result)

    assert data["error_code"] == "NETWORK_ERROR"
    assert data["error"] == _expected_retryable_error("fetching option dates for 'AAPL'", exception)
    assert data["details"]["symbol"] == "AAPL"


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_dates_api_error(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test generic API errors while fetching option dates."""
    mock_ticker.side_effect = Exception("generic error")
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_dates("AAPL")
    data = json.loads(result)

    assert data["error_code"] == "API_ERROR"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception",
    [TimeoutError("timed out"), OSError("network unreachable"), YFRateLimitError()],
)
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_ticker_network_error(
    mock_to_thread: AsyncMock, mock_ticker: MagicMock, exception: Exception
) -> None:
    """Test network errors during ticker creation in get_option_chain."""
    mock_ticker.side_effect = exception
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_chain("AAPL")
    data = json.loads(result)

    assert data["error_code"] == "NETWORK_ERROR"
    assert data["error"] == _expected_retryable_error("fetching options for 'AAPL'", exception)
    assert data["details"]["symbol"] == "AAPL"


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_dates_fetch_api_error(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test API error during available_dates fetch in get_option_chain."""
    mock_ticker_obj = MagicMock()
    # Mocking a property that raises Exception
    type(mock_ticker_obj).options = PropertyMock(side_effect=Exception("dates failed"))

    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_chain("AAPL")
    data = json.loads(result)

    assert data["error_code"] == "API_ERROR"
    assert "Failed to fetch option dates" in data["error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [TimeoutError("timed out"), YFRateLimitError()])
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_dates_fetch_network_error(
    mock_to_thread: AsyncMock, mock_ticker: MagicMock, exception: Exception
) -> None:
    """Test network error during available_dates fetch in get_option_chain."""
    mock_ticker_obj = MagicMock()
    type(mock_ticker_obj).options = PropertyMock(side_effect=exception)

    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_chain("AAPL")
    data = json.loads(result)

    assert data["error_code"] == "NETWORK_ERROR"
    assert data["error"] == _expected_retryable_error("fetching option dates for 'AAPL'", exception)


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_partial_failure(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test get_option_chain when one date fails but another succeeds."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.options = ["2025-05-02", "2025-05-09"]

    mock_opt = _MockOptionChain()

    # We need a custom side effect to make one call fail
    async def side_effect(func, *args, **kwargs):
        if not hasattr(side_effect, "ticker_created"):
            side_effect.ticker_created = True
            return mock_ticker_obj

        if callable(func):
            res = func()
            if res == mock_ticker_obj.options:
                return ["2025-05-02", "2025-05-09"]

            # This is the ticker.option_chain(date) call
            # We'll make the first date fail
            if not hasattr(side_effect, "fetch_called"):
                side_effect.fetch_called = True
                raise RuntimeError("Fetch failed for first date")
            return mock_opt
        return mock_ticker_obj

    mock_to_thread.side_effect = side_effect
    mock_ticker.return_value = mock_ticker_obj

    result = await get_option_chain("AAPL")
    data = json.loads(result)

    # Should have data for the second date but not the first
    assert "2025-05-09" in data
    assert "2025-05-02" not in data


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [TimeoutError("timed out"), YFRateLimitError()])
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_specific_date_fetch_network_error(
    mock_to_thread: AsyncMock, mock_ticker: MagicMock, exception: Exception
) -> None:
    """Test a single-date option chain fetch failure remains retryable."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.options = ["2025-05-02"]
    mock_ticker_obj.option_chain.side_effect = exception
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_chain("AAPL", expiration_date="2025-05-02")
    data = json.loads(result)

    assert data["error_code"] == "NETWORK_ERROR"
    assert data["error"] == _expected_retryable_error(
        "fetching option chain for 'AAPL' on '2025-05-02'",
        exception,
    )
    assert data["details"]["expiration_date"] == "2025-05-02"


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [TimeoutError("timed out"), YFRateLimitError()])
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_option_chain_all_dates_fetch_network_error(
    mock_to_thread: AsyncMock, mock_ticker: MagicMock, exception: Exception
) -> None:
    """Test all-date option chain fetch failures remain retryable."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.options = ["2025-05-02", "2025-05-09"]
    mock_ticker_obj.option_chain.side_effect = exception
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_option_chain("AAPL")
    data = json.loads(result)

    assert data["error_code"] == "NETWORK_ERROR"
    assert data["error"] == _expected_retryable_error("fetching option chain for 'AAPL'", exception)
    assert data["details"]["failed_dates"] == ["2025-05-02", "2025-05-09"]


def _major_holders_df() -> pd.DataFrame:
    return pd.DataFrame(
        {"Value": [0.0015, 0.65, 0.66, 7540.0]},
        index=["insidersPercentHeld", "institutionsPercentHeld", "institutionsFloatPercentHeld", "institutionsCount"],
    )


def _institutional_holders_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date Reported": ["2025-12-31", "2025-12-31"],
            "Holder": ["Vanguard Group Inc", "Blackrock Inc."],
            "Shares": [100_000_000, 80_000_000],
            "Value": [20_000_000_000, 16_000_000_000],
            "pctChange": [0.019, 0.007],
        }
    )


def _mutualfund_holders_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Date Reported": ["2025-12-31"],
            "Holder": ["Fidelity 500 Index"],
            "Shares": [10_000_000],
            "Value": [2_000_000_000],
            "pctChange": [-0.01],
        }
    )


def _insider_transactions_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Shares": [10_000, 5_000],
            "Value": [2_000_000, 1_000_000],
            "Start Date": ["2025-12-01", "2025-11-15"],
            "Transaction": ["Sale", "Purchase"],
            "Insider": ["COOK TIMOTHY D", "ADAMS KATHERINE L"],
        }
    )


def _insider_purchases_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Insider Purchases Last 6m": ["Purchases", "Sales", "Net Shares Purchased (Sold)"],
            "Shares": [50_000.0, 15_000.0, 35_000.0],
        }
    )


def _insider_roster_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Name": ["COOK TIMOTHY D", "KHAN SABIH"],
            "Position": ["Chief Executive Officer", "SVP, General Counsel"],
        }
    )


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_holders_success_all_sections(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test successful holders retrieval with all data sections."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.major_holders = _major_holders_df()
    mock_ticker_obj.institutional_holders = _institutional_holders_df()
    mock_ticker_obj.mutualfund_holders = _mutualfund_holders_df()
    mock_ticker_obj.insider_transactions = _insider_transactions_df()
    mock_ticker_obj.insider_purchases = _insider_purchases_df()
    mock_ticker_obj.insider_roster_holders = _insider_roster_df()
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_holders("AAPL")
    data = json.loads(result)

    assert "major_holders" in data
    assert "institutional_holders" in data
    assert "mutualfund_holders" in data
    assert "insider_transactions" in data
    assert "insider_purchases" in data
    assert "insider_roster" in data

    assert data["major_holders"][0] == {"index": "insidersPercentHeld", "Value": 0.0015}
    assert data["major_holders"][1] == {"index": "institutionsPercentHeld", "Value": 0.65}

    # Spot check institutional holder
    assert data["institutional_holders"][0]["Holder"] == "Vanguard Group Inc"
    assert data["institutional_holders"][0]["Shares"] == 100_000_000

    # Spot check insider roster
    assert data["insider_roster"][0]["Name"] == "COOK TIMOTHY D"


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_holders_partial_data(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test holders retrieval where some sections are empty DataFrames."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.major_holders = _major_holders_df()
    mock_ticker_obj.institutional_holders = pd.DataFrame()
    mock_ticker_obj.mutualfund_holders = None
    mock_ticker_obj.insider_transactions = _insider_transactions_df()
    mock_ticker_obj.insider_purchases = pd.DataFrame()
    mock_ticker_obj.insider_roster_holders = None
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_holders("AAPL")
    data = json.loads(result)

    assert "major_holders" in data
    assert "insider_transactions" in data
    assert "institutional_holders" not in data
    assert "mutualfund_holders" not in data
    assert "insider_purchases" not in data
    assert "insider_roster" not in data


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_holders_no_data(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test holders retrieval with no data at all."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.major_holders = None
    mock_ticker_obj.institutional_holders = pd.DataFrame()
    mock_ticker_obj.mutualfund_holders = None
    mock_ticker_obj.insider_transactions = pd.DataFrame()
    mock_ticker_obj.insider_purchases = None
    mock_ticker_obj.insider_roster_holders = None
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_holders("EMPTY")
    data = json.loads(result)

    assert data["error_code"] == "NO_DATA"
    assert data["details"]["symbol"] == "EMPTY"


@pytest.mark.asyncio
@pytest.mark.parametrize("exception", [TimeoutError("timed out"), OSError("network unreachable"), YFRateLimitError()])
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_holders_ticker_network_error(
    mock_to_thread: AsyncMock, mock_ticker: MagicMock, exception: Exception
) -> None:
    """Test network errors during ticker creation return structured network errors."""
    mock_ticker.side_effect = exception
    mock_to_thread.side_effect = _run_to_thread

    result = await get_holders("AAPL")
    data = json.loads(result)

    assert data["error_code"] == "NETWORK_ERROR"
    assert data["error"] == _expected_retryable_error("fetching holders for 'AAPL'", exception)
    assert data["details"]["symbol"] == "AAPL"


@pytest.mark.asyncio
@patch("yfmcp.server.yf.Ticker")
@patch("yfmcp.server.asyncio.to_thread")
async def test_get_holders_partial_failure(mock_to_thread: AsyncMock, mock_ticker: MagicMock) -> None:
    """Test that partial failures in some sections still return available data."""
    mock_ticker_obj = MagicMock()
    mock_ticker_obj.major_holders = _major_holders_df()
    mock_ticker_obj.institutional_holders = _institutional_holders_df()
    # Raise error when accessing mutualfund_holders
    type(mock_ticker_obj).mutualfund_holders = PropertyMock(side_effect=RuntimeError("fetch failed"))
    # Raise error when accessing insider_transactions
    type(mock_ticker_obj).insider_transactions = PropertyMock(side_effect=RuntimeError("fetch failed"))
    mock_ticker_obj.insider_purchases = _insider_purchases_df()
    mock_ticker_obj.insider_roster_holders = _insider_roster_df()
    mock_ticker.return_value = mock_ticker_obj
    mock_to_thread.side_effect = _run_to_thread

    result = await get_holders("AAPL")
    data = json.loads(result)

    # Should still return the sections that succeeded
    assert "major_holders" in data
    assert "institutional_holders" in data
    assert "insider_purchases" in data
    assert "insider_roster" in data
    # Failed sections should be absent
    assert "mutualfund_holders" not in data
    assert "insider_transactions" not in data
