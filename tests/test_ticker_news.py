"""News symbol validation and response tests without network access."""

import json
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from yfmcp.server import get_ticker_news


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "symbol",
    [
        "",
        "<b>tofu-poc</b>",
        '{{"',
        "AAPL'",
        'AAPL"',
        "AAPL MSFT",
        " AAPL",
        "AAPL ",
        "AAPL\n",
        "AAPL\r",
        "AAPL\t",
        "AAPL\x00",
        "ＡＡＰＬ",
        "AAPL/XYZ",
    ],
)
async def test_invalid_news_symbol(symbol: str) -> None:
    with patch("yfmcp.server.yf.Ticker") as ticker:
        result = await get_ticker_news(symbol)

    assert json.loads(result) == {
        "error": "Invalid symbol. Use only ASCII letters, digits, and . - ^ = & _; the symbol must not be empty.",
        "error_code": "INVALID_PARAMS",
    }
    ticker.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "symbol",
    ["AAPL", "aapl", "BRK.B", "2330.TW", "^GSPC", "EURUSD=X", "ES=F", "BTC-USD", "M&M.NS", "TEST_SYMBOL", "A" * 21],
)
async def test_valid_news_symbol(symbol: str) -> None:
    news = [{"id": "article", "content": {"title": "Company news"}}]
    ticker = MagicMock()
    ticker.get_news.return_value = news
    with patch("yfmcp.server.yf.Ticker", return_value=ticker) as constructor:
        result = await get_ticker_news(symbol)

    assert json.loads(result) == news
    constructor.assert_called_once_with(symbol)
    ticker.get_news.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("outcome", "error_code"),
    [([], "NO_DATA"), (RuntimeError("fetch failed"), "API_ERROR"), (TimeoutError("timed out"), "NETWORK_ERROR")],
)
async def test_news_error_response(outcome: list | Exception, error_code: str) -> None:
    ticker = MagicMock()
    if isinstance(outcome, Exception):
        ticker.get_news.side_effect = outcome
    else:
        ticker.get_news.return_value = outcome
    with patch("yfmcp.server.yf.Ticker", return_value=ticker):
        result = await get_ticker_news("AAPL")

    data = json.loads(result)
    assert data["error_code"] == error_code
    assert data["details"]["symbol"] == "AAPL"
    if isinstance(outcome, Exception):
        assert data["details"]["exception"] == str(outcome)
