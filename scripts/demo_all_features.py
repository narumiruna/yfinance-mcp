"""Run a live MCP demo covering every public yfinance-mcp feature."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult
from mcp.types import ImageContent
from mcp.types import TextContent

EXPECTED_TOOL_NAMES = frozenset(
    {
        "yfinance_get_ticker_info",
        "yfinance_get_analyst_price_targets",
        "yfinance_get_analyst_estimates",
        "yfinance_get_fund_data",
        "yfinance_get_upgrades_downgrades",
        "yfinance_get_ticker_news",
        "yfinance_search",
        "yfinance_screen",
        "yfinance_screen_gappers",
        "yfinance_get_top",
        "yfinance_get_price_history",
        "yfinance_get_financials",
        "yfinance_get_option_chain",
        "yfinance_get_option_dates",
        "yfinance_get_holders",
    }
)

ANALYST_SECTIONS = [
    "recommendations",
    "earnings_estimate",
    "revenue_estimate",
    "eps_trend",
    "eps_revisions",
    "earnings_history",
    "growth_estimates",
]

FUND_SECTIONS = [
    "description",
    "fund_overview",
    "fund_operations",
    "asset_classes",
    "top_holdings",
    "equity_holdings",
    "bond_holdings",
    "bond_ratings",
    "sector_weightings",
]


def _demo_calls(symbol: str, fund_symbol: str, sector: str) -> list[tuple[str, dict[str, Any]]]:
    """Return the static portion of the demo call plan."""
    return [
        ("yfinance_get_ticker_info", {"symbol": symbol}),
        ("yfinance_get_analyst_price_targets", {"symbol": symbol}),
        (
            "yfinance_get_analyst_estimates",
            {"symbol": symbol, "sections": ANALYST_SECTIONS, "max_rows": 3},
        ),
        (
            "yfinance_get_fund_data",
            {"symbol": fund_symbol, "sections": FUND_SECTIONS, "max_rows": 3},
        ),
        ("yfinance_get_upgrades_downgrades", {"symbol": symbol, "max_rows": 3}),
        ("yfinance_get_ticker_news", {"symbol": symbol}),
        ("yfinance_search", {"query": "Apple", "search_type": "all"}),
        ("yfinance_search", {"query": "Apple", "search_type": "quotes"}),
        ("yfinance_search", {"query": "Apple", "search_type": "news"}),
        (
            "yfinance_screen",
            {"query": "day_gainers", "query_type": "predefined", "count": 5},
        ),
        (
            "yfinance_screen",
            {
                "query_type": "equity",
                "query": {
                    "operator": "and",
                    "operands": [
                        {"operator": "gt", "operands": ["percentchange", 3]},
                        {"operator": "eq", "operands": ["region", "us"]},
                        {"operator": "gte", "operands": ["intradayprice", 5]},
                        {"operator": "gt", "operands": ["dayvolume", 500000]},
                    ],
                },
                "sort_field": "percentchange",
                "sort_asc": False,
                "size": 5,
            },
        ),
        (
            "yfinance_screen",
            {
                "query_type": "fund",
                "query": {
                    "operator": "and",
                    "operands": [
                        {"operator": "eq", "operands": ["categoryname", "Large Blend"]},
                        {"operator": "is-in", "operands": ["performanceratingoverall", 4, 5]},
                        {"operator": "eq", "operands": ["exchange", "NAS"]},
                    ],
                },
                "size": 5,
            },
        ),
        (
            "yfinance_screen",
            {
                "query_type": "etf",
                "query": {
                    "operator": "and",
                    "operands": [
                        {"operator": "gt", "operands": ["intradayprice", 10]},
                        {"operator": "eq", "operands": ["region", "us"]},
                    ],
                },
                "size": 5,
            },
        ),
        (
            "yfinance_screen_gappers",
            {
                "min_percent_change": 3.0,
                "min_price": 5.0,
                "min_volume": 500000,
                "min_market_cap": 2_000_000_000,
                "region": "us",
                "size": 5,
                "sort_asc": False,
            },
        ),
        ("yfinance_get_top", {"sector": sector, "top_type": "top_etfs", "top_n": 1}),
        ("yfinance_get_top", {"sector": sector, "top_type": "top_mutual_funds", "top_n": 1}),
        ("yfinance_get_top", {"sector": sector, "top_type": "top_companies", "top_n": 1}),
        ("yfinance_get_top", {"sector": sector, "top_type": "top_growth_companies", "top_n": 1}),
        (
            "yfinance_get_top",
            {"sector": sector, "top_type": "top_performing_companies", "top_n": 1},
        ),
        (
            "yfinance_get_price_history",
            {"symbol": symbol, "period": "5d", "interval": "1h", "prepost": True},
        ),
        (
            "yfinance_get_price_history",
            {"symbol": symbol, "period": "1mo", "interval": "1d", "chart_type": "price_volume"},
        ),
        (
            "yfinance_get_price_history",
            {"symbol": symbol, "period": "1mo", "interval": "1d", "chart_type": "vwap"},
        ),
        (
            "yfinance_get_price_history",
            {"symbol": symbol, "period": "1mo", "interval": "1d", "chart_type": "volume_profile"},
        ),
        ("yfinance_get_financials", {"symbol": symbol, "frequency": "annual"}),
        ("yfinance_get_financials", {"symbol": symbol, "frequency": "quarterly"}),
        ("yfinance_get_financials", {"symbol": symbol, "frequency": "ttm"}),
        ("yfinance_get_option_dates", {"symbol": symbol}),
        ("yfinance_get_holders", {"symbol": symbol, "max_rows": 3}),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol used for the stock-oriented calls.")
    parser.add_argument("--fund-symbol", default="SPY", help="ETF or mutual-fund symbol for fund data.")
    parser.add_argument("--sector", default="Technology", help="Sector used for ranking calls.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo-output"),
        help="Directory for complete text responses and returned chart images.",
    )
    parser.add_argument(
        "--server-command",
        default="uvx",
        help="MCP server command. Defaults to uvx for an independently installed yfmcp server.",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        dest="server_args",
        help="Argument passed to the MCP server command. Repeat for multiple arguments.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the complete feature plan without starting the MCP server or calling Yahoo Finance.",
    )
    return parser.parse_args()


def _server_parameters(args: argparse.Namespace) -> StdioServerParameters:
    if args.server_args is not None:
        server_args = args.server_args
    elif args.server_command == "uvx":
        server_args = ["yfmcp@latest"]
    else:
        server_args = []
    return StdioServerParameters(command=args.server_command, args=server_args)


def _json_payload(result: CallToolResult | None) -> Any:
    if result is None:
        return None
    for content in result.content:
        if isinstance(content, TextContent):
            try:
                return json.loads(content.text)
            except json.JSONDecodeError:
                continue
    return None


def _preview(text: str, limit: int = 900) -> str:
    try:
        display = json.dumps(json.loads(text), indent=2, ensure_ascii=False, default=str)
    except json.JSONDecodeError:
        lines = text.splitlines()
        display = "\n".join(lines[:12])
        if len(lines) > 12:
            display += f"\n... ({len(lines) - 12} more lines)"
    if len(display) > limit:
        return f"{display[:limit]}..."
    return display


def _save_result(result: CallToolResult, output_dir: Path, call_number: int, tool_name: str) -> list[Path]:
    saved_paths: list[Path] = []
    for content_number, content in enumerate(result.content, start=1):
        stem = f"{call_number:02d}_{tool_name}_{content_number}"
        if isinstance(content, TextContent):
            path = output_dir / f"{stem}.txt"
            path.write_text(content.text, encoding="utf-8")
        elif isinstance(content, ImageContent):
            extension = ".webp" if content.mimeType == "image/webp" else ".bin"
            path = output_dir / f"{stem}{extension}"
            path.write_bytes(base64.b64decode(content.data))
        else:
            continue
        saved_paths.append(path)
    return saved_paths


async def _call_tool(
    session: ClientSession,
    output_dir: Path,
    call_number: int,
    tool_name: str,
    arguments: dict[str, Any],
) -> CallToolResult | None:
    print(f"\n[{call_number}] {tool_name}")
    print(f"arguments: {json.dumps(arguments, sort_keys=True)}")
    try:
        result = await session.call_tool(tool_name, arguments=arguments)
    except Exception as exc:
        print(f"transport error: {exc}", file=sys.stderr)
        return None

    saved_paths = _save_result(result, output_dir, call_number, tool_name)
    if result.isError:
        print("server returned an MCP error result")
    for content in result.content:
        if isinstance(content, TextContent):
            print(_preview(content.text))
        elif isinstance(content, ImageContent):
            print(f"returned {content.mimeType} image")
    for path in saved_paths:
        print(f"saved: {path}")
    return result


def _first_option_date(result: CallToolResult | None) -> str | None:
    payload = _json_payload(result)
    if isinstance(payload, list) and payload and isinstance(payload[0], str):
        return payload[0]
    return None


def _print_dry_run(args: argparse.Namespace) -> None:
    calls = _demo_calls(args.symbol, args.fund_symbol, args.sector)
    print(f"The live demo will make {len(calls) + 3} MCP calls.")
    for number, (tool_name, arguments) in enumerate(calls, start=1):
        print(f"[{number:02d}] {tool_name}: {json.dumps(arguments, sort_keys=True)}")
    next_number = len(calls) + 1
    option_dates_call_number = next(
        number for number, (tool_name, _) in enumerate(calls, start=1) if tool_name == "yfinance_get_option_dates"
    )
    for option_type in ("all", "calls", "puts"):
        print(
            f"[{next_number:02d}] yfinance_get_option_chain: "
            f'{{"symbol": "{args.symbol}", "expiration_date": '
            f'"<first date from call {option_dates_call_number}>", "option_type": "{option_type}"}}'
        )
        next_number += 1


async def _run_demo(args: argparse.Namespace) -> int:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    server_parameters = _server_parameters(args)
    calls = _demo_calls(args.symbol, args.fund_symbol, args.sector)
    failures = 0

    async with (
        stdio_client(server_parameters) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        listed_tools = await session.list_tools()
        available_tools = {tool.name for tool in listed_tools.tools}
        missing_tools = sorted(EXPECTED_TOOL_NAMES - available_tools)
        if missing_tools:
            print(f"MCP server is missing expected tools: {', '.join(missing_tools)}", file=sys.stderr)
            return 1
        print(f"Connected to yfinance-mcp with {len(available_tools)} tools.")
        print(f"Complete responses will be saved in {output_dir}")

        option_dates_result: CallToolResult | None = None
        for call_number, (tool_name, arguments) in enumerate(calls, start=1):
            result = await _call_tool(session, output_dir, call_number, tool_name, arguments)
            if result is None:
                failures += 1
            if tool_name == "yfinance_get_option_dates":
                option_dates_result = result

        expiration_date = _first_option_date(option_dates_result)
        for offset, option_type in enumerate(("all", "calls", "puts"), start=1):
            chain_arguments: dict[str, Any] = {"symbol": args.symbol, "option_type": option_type}
            if expiration_date is not None:
                chain_arguments["expiration_date"] = expiration_date
            result = await _call_tool(
                session,
                output_dir,
                len(calls) + offset,
                "yfinance_get_option_chain",
                chain_arguments,
            )
            if result is None:
                failures += 1

    print(f"\nDemo finished with {failures} transport failure(s).")
    return 1 if failures else 0


def main() -> int:
    args = _parse_args()
    if args.dry_run:
        _print_dry_run(args)
        return 0
    try:
        return asyncio.run(_run_demo(args))
    except KeyboardInterrupt:
        print("Demo interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
