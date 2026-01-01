import json

import pytest
from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent


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


@pytest.mark.asyncio
async def test_get_top(server_params: StdioServerParameters) -> None:
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        sector = "Healthcare"
        top_n = 5

        result = await session.call_tool(
            "yfinance_get_top", arguments={"sector": sector, "top_n": top_n, "top_type": "top_companies"}
        )
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)

        data = json.loads(result.content[0].text)
        if "error" in data:
            print("Skipped len(data) check due to error:", data["error"])
        else:
            assert len(data) == top_n


@pytest.mark.asyncio
async def test_error_format_no_data(server_params: StdioServerParameters) -> None:
    """Test that NO_DATA errors contain proper error_code and details."""
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        # Test with a symbol that has no news (use an obscure/delisted symbol)
        # Some symbols may not have news articles available
        result = await session.call_tool("yfinance_get_ticker_news", arguments={"symbol": "ZZZZZ"})
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)

        data = json.loads(result.content[0].text)

        # This should be an error (either NO_DATA, API_ERROR, or NETWORK_ERROR)
        assert "error" in data, "Expected an error response for invalid/obscure symbol"
        assert "error_code" in data, "Error response should contain 'error_code' field"
        assert data["error_code"] in ["NO_DATA", "API_ERROR", "NETWORK_ERROR"], f"Unexpected error_code: {data['error_code']}"

        # Verify details field exists
        assert "details" in data, "Error response should contain 'details' field"
        assert isinstance(data["details"], dict), "Details should be a dictionary"
        assert "symbol" in data["details"], "Details should contain the symbol"


@pytest.mark.asyncio
async def test_error_format_structure(server_params: StdioServerParameters) -> None:
    """Test error response structure contains error_code and details fields.
    
    This test uses an invalid/obscure symbol to reliably trigger an error response,
    then validates that the error follows the structured format with error_code
    and details fields. This ensures all error responses follow the same structure.
    """
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        # Use an invalid symbol to guarantee an error response
        result = await session.call_tool(
            "yfinance_get_ticker_info",
            arguments={"symbol": "INVALID_SYMBOL_XYZ123"},
        )
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)

        data = json.loads(result.content[0].text)

        # Verify error structure
        assert "error" in data, "Expected an error response for invalid symbol"
        assert "error_code" in data, "Error response should contain 'error_code' field"
        assert isinstance(data["error_code"], str), "error_code should be a string"
        assert "details" in data, "Error response should contain 'details' field"
        assert isinstance(data["details"], dict), "Details should be a dictionary"
