import pytest
from yfinance import EquityQuery
from yfinance import FundQuery

from yfmcp.screener import build_screener_query


def test_build_equity_query_success() -> None:
    """Test JSON-style equity query trees build yfinance EquityQuery objects."""
    query = {
        "operator": "and",
        "operands": [
            {"operator": "gt", "operands": ["percentchange", 3]},
            {"operator": "eq", "operands": ["region", "us"]},
        ],
    }

    result = build_screener_query("equity", query)

    assert isinstance(result, EquityQuery)
    assert result.operator == "AND"
    assert len(result.operands) == 2


def test_build_fund_query_success() -> None:
    """Test JSON-style fund query trees build yfinance FundQuery objects."""
    query = {
        "operator": "and",
        "operands": [
            {"operator": "eq", "operands": ["categoryname", "Large Growth"]},
            {"operator": "eq", "operands": ["exchange", "NAS"]},
        ],
    }

    result = build_screener_query("fund", query)

    assert isinstance(result, FundQuery)
    assert result.operator == "AND"


def test_build_query_invalid_operator() -> None:
    """Test unsupported operators are rejected."""
    query = {"operator": "contains", "operands": ["region", "us"]}

    with pytest.raises(ValueError, match="Unsupported operator"):
        build_screener_query("equity", query)


def test_build_query_invalid_type() -> None:
    """Test custom query builder only accepts equity and fund query types."""
    query = {"operator": "eq", "operands": ["region", "us"]}

    with pytest.raises(ValueError, match="query_type must be 'equity' or 'fund'"):
        build_screener_query("predefined", query)
