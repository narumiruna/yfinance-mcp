"""Run a live MCP demo covering every public yfinance-mcp feature."""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from dataclasses import dataclass
from datetime import datetime
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

TOOL_GUIDANCE: dict[str, tuple[str, str, str]] = {
    "yfinance_get_ticker_info": (
        "Company research",
        "Company snapshot",
        "Start here to understand what a company does, how it trades, and how it is valued.",
    ),
    "yfinance_get_analyst_price_targets": (
        "Company research",
        "Analyst price targets",
        "Compare the current price with the analyst consensus range and median target.",
    ),
    "yfinance_get_analyst_estimates": (
        "Company research",
        "Analyst estimates and trends",
        "See consensus earnings, revenue expectations, EPS revisions, recommendations, and growth estimates.",
    ),
    "yfinance_get_upgrades_downgrades": (
        "Company research",
        "Analyst actions",
        "Track recent upgrades, downgrades, initiations, reiterations, and target-price changes.",
    ),
    "yfinance_get_ticker_news": (
        "Company research",
        "Ticker news",
        "Collect recent company-specific news and press coverage for monitoring or research workflows.",
    ),
    "yfinance_get_fund_data": (
        "Fund research",
        "ETF or mutual-fund profile",
        "Inspect a fund's description, holdings, asset mix, expenses, ratings, and sector exposure.",
    ),
    "yfinance_search": (
        "Discovery",
        "Yahoo Finance search",
        "Find securities or news when you know a name, theme, or keyword but not the exact ticker.",
    ),
    "yfinance_screen": (
        "Discovery",
        "Custom or predefined screener",
        "Turn an investment idea into a repeatable filter for equities, mutual funds, or ETFs.",
    ),
    "yfinance_screen_gappers": (
        "Discovery",
        "Opening-session gappers",
        "Find liquid stocks making a large move while enforcing price, volume, market-cap, and region filters.",
    ),
    "yfinance_get_top": (
        "Market context",
        "Sector rankings",
        "Compare leading ETFs, funds, companies, growth names, or performers within a sector.",
    ),
    "yfinance_get_price_history": (
        "Market context",
        "Price history and charts",
        "Review OHLCV history or create technical-analysis visuals for a security.",
    ),
    "yfinance_get_financials": (
        "Fundamental analysis",
        "Financial statements",
        "Analyze revenue, profitability, balance-sheet strength, and cash flow across reporting frequencies.",
    ),
    "yfinance_get_holders": (
        "Fundamental analysis",
        "Ownership and insider activity",
        "Understand institutional concentration, mutual-fund ownership, and insider transactions.",
    ),
    "yfinance_get_option_dates": (
        "Options analysis",
        "Option expiration dates",
        "Discover the expirations available before requesting a specific option chain.",
    ),
    "yfinance_get_option_chain": (
        "Options analysis",
        "Option chain",
        "Inspect calls, puts, strikes, implied volatility, open interest, and liquidity for an expiration.",
    ),
}


@dataclass
class CallRecord:
    number: int
    tool_name: str
    arguments: dict[str, Any]
    result: CallToolResult | None
    saved_paths: list[Path]
    summary: str
    preview: str
    transport_error: str | None


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


def _format_number(value: Any) -> str:
    if isinstance(value, int | float):
        return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
    return str(value)


def _response_summary(tool_name: str, arguments: dict[str, Any], result: CallToolResult | None) -> str:  # noqa: C901
    if result is None:
        return "No MCP response was received."

    payload = _json_payload(result)
    if isinstance(payload, dict) and "error" in payload:
        error_code = payload.get("error_code", "unknown error")
        return f"Server returned {error_code}: {payload['error']}"

    if tool_name == "yfinance_get_ticker_info" and isinstance(payload, dict):
        name = payload.get("longName") or payload.get("shortName") or arguments["symbol"]
        sector = payload.get("sector")
        industry = payload.get("industry")
        price = payload.get("currentPrice") or payload.get("regularMarketPrice")
        parts = [f"{name} ({arguments['symbol']})"]
        if sector or industry:
            parts.append(" / ".join(str(item) for item in (sector, industry) if item))
        if price is not None:
            parts.append(f"current price ${_format_number(price)}")
        if payload.get("marketCap") is not None:
            parts.append(f"market cap ${_format_number(payload['marketCap'])}")
        return "; ".join(parts) + "."

    if tool_name == "yfinance_get_analyst_price_targets" and isinstance(payload, dict):
        current = payload.get("current")
        low = payload.get("low")
        high = payload.get("high")
        mean = payload.get("mean")
        median = payload.get("median")
        return (
            f"Current ${_format_number(current)}; analyst range ${_format_number(low)}–${_format_number(high)}; "
            f"mean ${_format_number(mean)}, median ${_format_number(median)}."
        )

    if tool_name in {"yfinance_get_analyst_estimates", "yfinance_get_fund_data"} and isinstance(payload, dict):
        sections = [key for key in payload if not key.startswith("_")]
        metadata = payload.get("_metadata", {})
        section_names = ", ".join(str(section) for section in sections)
        row_limit = metadata.get("max_rows", "configured")
        return f"Returned {len(sections)} sections ({section_names}), with a {row_limit} row limit per table."

    if tool_name == "yfinance_get_upgrades_downgrades" and isinstance(payload, dict):
        rows = payload.get("upgrades_downgrades", [])
        latest = rows[0] if rows else {}
        latest_action = ", ".join(
            str(value) for value in (latest.get("Firm"), latest.get("Action"), latest.get("ToGrade")) if value
        )
        suffix = f" Latest: {latest_action}." if latest_action else ""
        return f"Returned {len(rows)} recent analyst actions.{suffix}"

    if tool_name == "yfinance_get_ticker_news" and isinstance(payload, list):
        titles = []
        for item in payload[:3]:
            content = item.get("content", item) if isinstance(item, dict) else {}
            if content.get("title"):
                titles.append(str(content["title"]))
        latest = f" Latest: {'; '.join(titles)}." if titles else ""
        return f"Found {len(payload)} recent news items.{latest}"

    if tool_name == "yfinance_search":
        if isinstance(payload, dict):
            quotes = payload.get("quotes", [])
            news = payload.get("news", [])
            return f"Found {len(quotes)} quote matches and {len(news)} news matches."
        if isinstance(payload, list):
            label = "news articles" if arguments.get("search_type") == "news" else "securities"
            return f"Found {len(payload)} {label} for '{arguments['query']}'."

    if tool_name in {"yfinance_screen", "yfinance_screen_gappers"} and isinstance(payload, dict):
        quotes = payload.get("quotes", [])
        total = payload.get("total", len(quotes))
        symbols = [quote.get("symbol") for quote in quotes[:5] if isinstance(quote, dict) and quote.get("symbol")]
        match_text = ", ".join(symbols) if symbols else "no symbols in the response preview"
        return f"Found {total} matching securities; first returned symbols: {match_text}."

    if tool_name == "yfinance_get_top" and isinstance(payload, list):
        names = []
        for item in payload[:5]:
            if not isinstance(item, dict):
                continue
            if item.get("symbol") or item.get("name"):
                names.append(str(item.get("symbol") or item.get("name")))
            elif item.get("industry"):
                names.append(str(item["industry"]))
        return f"Returned {len(payload)} ranked results; examples: {', '.join(names)}."

    if tool_name == "yfinance_get_price_history":
        if any(isinstance(content, ImageContent) for content in result.content):
            chart_type = arguments.get("chart_type", "chart")
            return f"Returned the {chart_type} chart as a WebP image."
        text = next((content.text for content in result.content if isinstance(content, TextContent)), "")
        rows = max(len(text.splitlines()) - 3, 0)
        return f"Returned a Markdown OHLCV table with approximately {rows} data rows."

    if tool_name == "yfinance_get_financials" and isinstance(payload, dict):
        sections = list(payload)
        period_count = sum(
            len(periods)
            for section in payload.values()
            if isinstance(section, dict)
            for periods in section.values()
            if isinstance(periods, dict)
        )
        section_names = ", ".join(str(section) for section in sections)
        return f"Returned {section_names} for {arguments['frequency']} reporting ({period_count} field-period values)."

    if tool_name == "yfinance_get_option_dates" and isinstance(payload, list):
        first = payload[0] if payload else "none"
        last = payload[-1] if payload else "none"
        return f"Found {len(payload)} available expirations, from {first} through {last}."

    if tool_name == "yfinance_get_option_chain" and isinstance(payload, dict):
        date_summaries = []
        for date, data in payload.items():
            if not isinstance(data, dict):
                continue
            calls = len(data.get("calls", []))
            puts = len(data.get("puts", []))
            date_summaries.append(f"{date}: {calls} calls, {puts} puts")
        return f"Returned {len(payload)} expiration(s): {'; '.join(date_summaries)}."

    if tool_name == "yfinance_get_holders" and isinstance(payload, dict):
        sections = {key: len(value) for key, value in payload.items() if isinstance(value, list)}
        section_text = ", ".join(f"{key}={count}" for key, count in sections.items())
        return f"Returned ownership and insider sections ({section_text})."

    if isinstance(payload, dict):
        return f"Returned a JSON object with: {', '.join(payload.keys())}."
    if isinstance(payload, list):
        return f"Returned a JSON list with {len(payload)} items."
    return "Returned a text response."


def _call_title(tool_name: str, arguments: dict[str, Any]) -> str:
    title = TOOL_GUIDANCE[tool_name][1]
    if tool_name == "yfinance_search":
        return f"{title} ({arguments['search_type']})"
    if tool_name == "yfinance_screen":
        return f"{title} ({arguments['query_type']})"
    if tool_name == "yfinance_get_top":
        return f"{title} ({arguments['top_type']})"
    if tool_name == "yfinance_get_price_history":
        mode = arguments.get("chart_type", "table")
        return f"{title} ({mode})"
    if tool_name == "yfinance_get_financials":
        return f"{title} ({arguments['frequency']})"
    if tool_name == "yfinance_get_option_chain":
        return f"{title} ({arguments['option_type']})"
    return title


def _variant_label(tool_name: str, arguments: dict[str, Any]) -> str:
    if tool_name == "yfinance_search":
        return f"{arguments['search_type']}"
    if tool_name == "yfinance_screen":
        return f"{arguments['query_type']}"
    if tool_name == "yfinance_screen_gappers":
        return "gappers"
    if tool_name in {"yfinance_get_top", "yfinance_get_financials", "yfinance_get_option_chain"}:
        key = {
            "yfinance_get_top": "top_type",
            "yfinance_get_financials": "frequency",
            "yfinance_get_option_chain": "option_type",
        }[tool_name]
        return str(arguments[key])
    if tool_name == "yfinance_get_price_history":
        return str(arguments.get("chart_type", "table"))
    return "default"


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
    total_calls: int,
    tool_name: str,
    arguments: dict[str, Any],
) -> CallRecord:
    print(f"\n[{call_number}/{total_calls}] {_call_title(tool_name, arguments)}")
    print(f"MCP tool: {tool_name}")
    print(f"arguments: {json.dumps(arguments, sort_keys=True)}")
    try:
        result = await session.call_tool(tool_name, arguments=arguments)
    except Exception as exc:
        print(f"transport error: {exc}", file=sys.stderr)
        return CallRecord(
            number=call_number,
            tool_name=tool_name,
            arguments=arguments,
            result=None,
            saved_paths=[],
            summary=f"Transport error: {exc}",
            preview="",
            transport_error=str(exc),
        )

    saved_paths = _save_result(result, output_dir, call_number, tool_name)
    summary = _response_summary(tool_name, arguments, result)
    preview = "\n".join(_preview(content.text) for content in result.content if isinstance(content, TextContent))
    print(f"result: {summary}")
    for path in saved_paths:
        print(f"artifact: {path}")
    return CallRecord(
        number=call_number,
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        saved_paths=saved_paths,
        summary=summary,
        preview=preview,
        transport_error=None,
    )


def _first_option_date(record: CallRecord | None) -> str | None:
    payload = _json_payload(record.result if record is not None else None)
    if isinstance(payload, list) and payload and isinstance(payload[0], str):
        return payload[0]
    return None


def _markdown_artifact_link(path: Path) -> str:
    return f"[{path.name}]({path.name})"


def _render_report(
    records: list[CallRecord],
    available_tools: set[str],
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    report_path = output_dir / "demo-report.md"
    failures = [record for record in records if record.transport_error]
    tool_order = list(dict.fromkeys(record.tool_name for record in records))
    coverage_rows = []
    for tool_name in tool_order:
        tool_records = [record for record in records if record.tool_name == tool_name]
        variants = ", ".join(dict.fromkeys(_variant_label(tool_name, record.arguments) for record in tool_records))
        purpose = TOOL_GUIDANCE[tool_name][2]
        coverage_rows.append(f"| `{tool_name}` | {len(tool_records)} | {variants} | {purpose} |")

    groups: dict[str, list[CallRecord]] = {}
    for record in records:
        group = TOOL_GUIDANCE[record.tool_name][0]
        groups.setdefault(group, []).append(record)

    lines = [
        "# yfinance-mcp Feature Demo Report",
        "",
        f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        "## What this demonstrates",
        "",
        (
            "This is a real MCP client session. The demo never imports yfinance or yfinance-mcp internals; "
            "it asks the MCP server for data and presents the results in user-facing research workflows."
        ),
        "",
        (
            "Use this report to answer: **What can I ask yfinance-mcp to do, what arguments do I provide, "
            "and what comes back?**"
        ),
        "",
        "- **Research a company:** understand the business, valuation, analyst expectations, news, and ownership.",
        "- **Research funds:** inspect holdings, exposures, expenses, and fund characteristics.",
        "- **Discover opportunities:** search Yahoo Finance, run repeatable screeners, and find sector leaders.",
        "- **Analyze markets:** retrieve OHLCV history, technical charts, financial statements, and options data.",
        "",
        "## Run context",
        "",
        f"- Stock symbol: `{args.symbol}`",
        f"- Fund symbol: `{args.fund_symbol}`",
        f"- Sector: `{args.sector}`",
        f"- MCP server command: `{args.server_command}`",
        f"- Server tools advertised: **{len(available_tools)}**",
        f"- Calls completed: **{len(records)}**",
        f"- Transport failures: **{len(failures)}**",
        "",
        "## Coverage at a glance",
        "",
        (
            "Every public tool is represented below. Variants show the arguments that make the same tool useful "
            "for different jobs."
        ),
        "",
        "| MCP tool | Calls | Variants exercised | Why you would use it |",
        "|---|---:|---|---|",
        *coverage_rows,
        "",
    ]

    for group, group_records in groups.items():
        lines.extend([f"## {group}", ""])
        for record in group_records:
            _, _, why = TOOL_GUIDANCE[record.tool_name]
            lines.extend(
                [
                    f"### {record.number}. {_call_title(record.tool_name, record.arguments)}",
                    "",
                    f"**MCP tool:** `{record.tool_name}`  ",
                    f"**Why it matters:** {why}",
                    "",
                    "**Arguments used**",
                    "",
                    "```json",
                    json.dumps(record.arguments, indent=2, ensure_ascii=False),
                    "```",
                    "",
                    f"**What came back:** {record.summary}",
                    "",
                ]
            )
            if record.preview:
                excerpt = record.preview.replace("```", "`` `")
                lines.extend(["**Response excerpt**", "", "```text", excerpt, "```", ""])
            if record.saved_paths:
                artifact_links = ", ".join(_markdown_artifact_link(path) for path in record.saved_paths)
                lines.extend([f"**Complete artifact:** {artifact_links}", ""])
            if record.transport_error:
                lines.extend([f"**Transport error:** `{record.transport_error}`", ""])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


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
    total_calls = len(calls) + 3
    failures = 0
    records: list[CallRecord] = []
    available_tools: set[str] = set()

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
        print(f"A readable report and complete artifacts will be saved in {output_dir}")

        option_dates_record: CallRecord | None = None
        for call_number, (tool_name, arguments) in enumerate(calls, start=1):
            record = await _call_tool(session, output_dir, call_number, total_calls, tool_name, arguments)
            records.append(record)
            if record.transport_error:
                failures += 1
            if tool_name == "yfinance_get_option_dates":
                option_dates_record = record

        expiration_date = _first_option_date(option_dates_record)
        for offset, option_type in enumerate(("all", "calls", "puts"), start=1):
            chain_arguments: dict[str, Any] = {"symbol": args.symbol, "option_type": option_type}
            if expiration_date is not None:
                chain_arguments["expiration_date"] = expiration_date
            record = await _call_tool(
                session,
                output_dir,
                len(calls) + offset,
                total_calls,
                "yfinance_get_option_chain",
                chain_arguments,
            )
            records.append(record)
            if record.transport_error:
                failures += 1

    report_path = _render_report(records, available_tools, output_dir, args)
    print(f"\nReport: {report_path}")
    print(f"Demo finished with {failures} transport failure(s).")
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
