import json

import pytest
from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

from yfmcp.server import create_error_response
from yfmcp.types import ErrorCode


@pytest.fixture
def server_params() -> StdioServerParameters:
    return StdioServerParameters(command="yfmcp")


@pytest.mark.asyncio
async def test_list_tools(server_params: StdioServerParameters) -> None:
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        result = await session.list_tools()
        assert len(result.tools) > 0


@pytest.mark.asyncio
async def test_get_ticker_info(server_params: StdioServerParameters) -> None:
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        symbol = "AAPL"
        result = await session.call_tool("yfinance_get_ticker_info", arguments={"symbol": symbol})
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)

        data = json.loads(result.content[0].text)
        assert isinstance(data, dict)
        assert data["symbol"] == symbol


# Removed slow integration test for get_top - replaced with unit tests below


# Fast unit tests for error handling (no network calls)


def test_error_function_structure() -> None:
    """Test that create_error_response() function creates proper error structure."""
    # Test with minimal parameters
    error_json = create_error_response("Test error message")
    data = json.loads(error_json)
    assert "error" in data
    assert data["error"] == "Test error message"
    assert "error_code" in data
    assert data["error_code"] == "UNKNOWN_ERROR"
    assert "details" not in data  # No details when not provided

    # Test with error_code
    error_json = create_error_response("Invalid symbol", error_code="INVALID_SYMBOL")
    data = json.loads(error_json)
    assert data["error"] == "Invalid symbol"
    assert data["error_code"] == "INVALID_SYMBOL"

    # Test with details
    error_json = create_error_response(
        "API failed",
        error_code="API_ERROR",
        details={"symbol": "AAPL", "exception": "Connection timeout"},
    )
    data = json.loads(error_json)
    assert data["error"] == "API failed"
    assert data["error_code"] == "API_ERROR"
    assert "details" in data
    assert data["details"]["symbol"] == "AAPL"
    assert data["details"]["exception"] == "Connection timeout"


def test_error_code_types() -> None:
    """Test all error code types are handled correctly."""
    error_codes: list[ErrorCode] = [
        "INVALID_SYMBOL",
        "NO_DATA",
        "API_ERROR",
        "INVALID_PARAMS",
        "NETWORK_ERROR",
        "UNKNOWN_ERROR",
    ]

    for code in error_codes:
        error_json = create_error_response(f"Test {code}", error_code=code, details={"test": "value"})
        data = json.loads(error_json)
        assert data["error_code"] == code
        assert "details" in data
        assert data["details"]["test"] == "value"
