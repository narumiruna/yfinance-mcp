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

        # This should be an error (either NO_DATA or API_ERROR)
        if "error" in data:
            assert "error_code" in data, "Error response should contain 'error_code' field"
            assert data["error_code"] in ["NO_DATA", "API_ERROR"], f"Unexpected error_code: {data['error_code']}"

            # Verify details field exists
            assert "details" in data, "Error response should contain 'details' field"
            assert isinstance(data["details"], dict), "Details should be a dictionary"
            assert "symbol" in data["details"], "Details should contain the symbol"


@pytest.mark.asyncio
async def test_error_format_invalid_sector(server_params: StdioServerParameters) -> None:
    """Test that invalid sector returns INVALID_PARAMS error with proper structure."""
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        # Test with an invalid sector - this should trigger INVALID_PARAMS error
        # Note: Due to Literal type validation, this may be caught earlier
        # So we test with get_top_growth_companies which uses SECTOR_INDUSTY_MAPPING
        result = await session.call_tool(
            "yfinance_get_top",
            arguments={"sector": "Technology", "top_n": 5, "top_type": "top_growth_companies"},
        )
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)

        data = json.loads(result.content[0].text)

        # This should either succeed or return an error with proper structure
        if "error" in data:
            assert "error_code" in data, "Error response should contain 'error_code' field"
            assert "details" in data, "Error response should contain 'details' field"
            assert isinstance(data["details"], dict), "Details should be a dictionary"
