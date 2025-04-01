import json

import pytest
from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent


@pytest.fixture
def server_params() -> StdioServerParameters:
    return StdioServerParameters(command="yfmcp")


@pytest.mark.anyio
async def test_list_tools(server_params: StdioServerParameters) -> None:
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        result = await session.list_tools()
        assert len(result.tools) > 0


@pytest.mark.anyio
async def test_get_ticker_info(server_params: StdioServerParameters) -> None:
    async with (
        stdio_client(server_params) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()

        symbol = "AAPL"
        result = await session.call_tool("get_ticker_info", arguments={"symbol": symbol})
        assert len(result.content) == 1
        assert isinstance(result.content[0], TextContent)

        data = json.loads(result.content[0].text)
        assert isinstance(data, dict)
        assert data["symbol"] == symbol
