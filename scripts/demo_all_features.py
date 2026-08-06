"""Run a live MCP demo covering every public yfinance-mcp feature."""

# ruff: noqa: E501 -- Embedded standalone HTML/CSS/JS is intentionally kept readable.

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from html import escape
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

FUND_LOOKTHROUGH_LIMIT = 3

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
    context: str | None = None


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
        (
            "yfinance_get_fund_data",
            {"symbol": fund_symbol, "sections": ["top_holdings"], "max_rows": 0},
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
    parser.add_argument("--fund-symbol", default="SCHD", help="ETF or mutual-fund symbol for fund data.")
    parser.add_argument("--sector", default="Technology", help="Sector used for ranking calls.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("demo-output"),
        help="Directory for complete text responses and returned chart images.",
    )
    parser.add_argument(
        "--server-command",
        default="uv",
        help="MCP server command. Defaults to the repository's configured uv environment.",
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
    elif args.server_command == "uv":
        project_root = Path(__file__).resolve().parents[1]
        server_args = ["run", "--directory", str(project_root), "yfmcp"]
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
    if tool_name == "yfinance_get_fund_data" and arguments.get("sections") == ["top_holdings"]:
        return "Fund look-through (underlying holdings)"
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
    context: str | None = None,
) -> CallRecord:
    print(f"\n[{call_number}/{total_calls}] {_display_title(tool_name, arguments, context)}")
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
            context=context,
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
        context=context,
    )


def _first_option_date(record: CallRecord | None) -> str | None:
    payload = _json_payload(record.result if record is not None else None)
    if isinstance(payload, list) and payload and isinstance(payload[0], str):
        return payload[0]
    return None


def _fund_top_holdings(record: CallRecord | None, limit: int = FUND_LOOKTHROUGH_LIMIT) -> list[dict[str, Any]]:
    payload = _json_payload(record.result if record is not None else None)
    if not isinstance(payload, dict) or not isinstance(payload.get("top_holdings"), list):
        return []

    holdings: list[dict[str, Any]] = []
    for item in payload["top_holdings"]:
        if not isinstance(item, dict):
            continue
        symbol = item.get("Symbol") or item.get("symbol")
        if not symbol:
            continue
        holdings.append(
            {
                "symbol": str(symbol),
                "name": str(item.get("Name") or item.get("name") or symbol),
                "weight": item.get("Holding Percent") or item.get("holdingPercent"),
            }
        )
        if len(holdings) == limit:
            break
    return holdings


def _display_title(tool_name: str, arguments: dict[str, Any], context: str | None = None) -> str:
    title = _call_title(tool_name, arguments)
    return f"{context} — {title}" if context else title


def _slug(value: str) -> str:
    slug = "".join(character.lower() if character.isalnum() else "-" for character in value)
    return "-".join(part for part in slug.split("-") if part)


def _server_command_text(server_parameters: StdioServerParameters) -> str:
    parts = [server_parameters.command, *server_parameters.args]
    return " ".join(f'"{part}"' if " " in part else part for part in parts)


def _record_group(record: CallRecord) -> str:
    if record.context and record.context.startswith("Fund look-through"):
        return "Fund research"
    return TOOL_GUIDANCE[record.tool_name][0]


def _record_error(record: CallRecord) -> str | None:
    if record.transport_error:
        return record.transport_error
    payload = _json_payload(record.result)
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    if record.result is not None and record.result.isError:
        return "The MCP server marked this response as an error."
    return None


def _html_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    head = "".join(f"<th>{escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(_format_number(value))}</td>" for value in row) + "</tr>" for row in rows
    )
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _structured_result_html(record: CallRecord) -> str:  # noqa: C901
    payload = _json_payload(record.result)
    sections: list[str] = []

    image_paths = [path for path in record.saved_paths if path.suffix.lower() == ".webp"]
    for path in image_paths:
        sections.append(
            f'<figure><img src="{escape(path.name, quote=True)}" alt="{escape(_display_title(record.tool_name, record.arguments, record.context), quote=True)}">'
            f"<figcaption>{escape(path.name)}</figcaption></figure>"
        )

    if record.tool_name == "yfinance_get_ticker_info" and isinstance(payload, dict):
        fields = [
            ("Name", payload.get("longName") or payload.get("shortName")),
            ("Symbol", payload.get("symbol") or record.arguments.get("symbol")),
            ("Sector", payload.get("sector")),
            ("Industry", payload.get("industry")),
            ("Current price", payload.get("currentPrice") or payload.get("regularMarketPrice")),
            ("Market cap", payload.get("marketCap")),
            ("Trailing P/E", payload.get("trailingPE")),
            ("Dividend yield", payload.get("dividendYield")),
        ]
        sections.append(_html_table(["Field", "Value"], [[name, value] for name, value in fields if value is not None]))

    elif record.tool_name == "yfinance_get_analyst_price_targets" and isinstance(payload, dict):
        rows = [
            [label, payload.get(key)]
            for label, key in (
                ("Current", "current"),
                ("Low", "low"),
                ("Mean", "mean"),
                ("Median", "median"),
                ("High", "high"),
            )
        ]
        sections.append(_html_table(["Price measure", "Value"], [row for row in rows if row[1] is not None]))

    elif record.tool_name == "yfinance_get_fund_data" and isinstance(payload, dict):
        holdings = payload.get("top_holdings")
        if isinstance(holdings, list):
            rows = []
            for holding in holdings:
                if not isinstance(holding, dict):
                    continue
                weight = holding.get("Holding Percent") or holding.get("holdingPercent")
                rows.append(
                    [
                        holding.get("Symbol") or holding.get("symbol"),
                        holding.get("Name") or holding.get("name"),
                        f"{float(weight) * 100:.2f}%" if isinstance(weight, int | float) else weight,
                    ]
                )
            sections.append(_html_table(["Symbol", "Holding", "Portfolio weight"], rows))
        metadata = payload.get("_metadata", {})
        section_rows = []
        if isinstance(metadata, dict) and isinstance(metadata.get("sections"), dict):
            for name, details in metadata["sections"].items():
                if isinstance(details, dict):
                    section_rows.append([name, details.get("returned_rows"), details.get("total_rows")])
        sections.append(_html_table(["Fund section", "Returned rows", "Available rows"], section_rows))

    elif record.tool_name == "yfinance_get_upgrades_downgrades" and isinstance(payload, dict):
        rows = []
        for item in payload.get("upgrades_downgrades", [])[:10]:
            if isinstance(item, dict):
                rows.append(
                    [
                        item.get("GradeDate"),
                        item.get("Firm"),
                        item.get("Action"),
                        item.get("FromGrade"),
                        item.get("ToGrade"),
                    ]
                )
        sections.append(_html_table(["Date", "Firm", "Action", "From", "To"], rows))

    elif record.tool_name == "yfinance_get_ticker_news" and isinstance(payload, list):
        rows = []
        for item in payload[:10]:
            content = item.get("content", item) if isinstance(item, dict) else {}
            if isinstance(content, dict):
                provider = content.get("provider", {})
                rows.append(
                    [
                        content.get("pubDate"),
                        provider.get("displayName") if isinstance(provider, dict) else provider,
                        content.get("title"),
                    ]
                )
        sections.append(_html_table(["Published", "Source", "Headline"], rows))

    elif record.tool_name == "yfinance_search":
        if isinstance(payload, dict):
            quote_items = payload.get("quotes", [])
            news_items = payload.get("news", [])
        elif record.arguments.get("search_type") == "news":
            quote_items, news_items = [], payload
        else:
            quote_items, news_items = payload, []
        if isinstance(quote_items, list) and quote_items:
            rows = []
            for item in quote_items[:10]:
                if isinstance(item, dict):
                    rows.append(
                        [
                            item.get("symbol"),
                            item.get("shortname") or item.get("longname"),
                            item.get("quoteType"),
                            item.get("regularMarketPrice"),
                        ]
                    )
            sections.append("<h4>Security matches</h4>" + _html_table(["Symbol", "Name", "Type", "Price"], rows))
        if isinstance(news_items, list) and news_items:
            rows = []
            for item in news_items[:10]:
                if isinstance(item, dict):
                    rows.append([item.get("providerPublishTime"), item.get("publisher"), item.get("title")])
            sections.append("<h4>News matches</h4>" + _html_table(["Published", "Source", "Headline"], rows))

    elif record.tool_name in {"yfinance_screen", "yfinance_screen_gappers"} and isinstance(payload, dict):
        rows = []
        for item in payload.get("quotes", [])[:10]:
            if isinstance(item, dict):
                rows.append(
                    [
                        item.get("symbol"),
                        item.get("shortName") or item.get("longName"),
                        item.get("regularMarketPrice"),
                        item.get("regularMarketChangePercent"),
                        item.get("regularMarketVolume"),
                    ]
                )
        sections.append(_html_table(["Symbol", "Name", "Price", "% change", "Volume"], rows))

    elif record.tool_name == "yfinance_get_top" and isinstance(payload, list):
        dictionaries = [item for item in payload[:10] if isinstance(item, dict)]
        columns = list(dict.fromkeys(key for item in dictionaries for key in item))[:6]
        sections.append(_html_table(columns, [[item.get(column) for column in columns] for item in dictionaries]))

    elif record.tool_name == "yfinance_get_analyst_estimates" and isinstance(payload, dict):
        rows = []
        for name, value in payload.items():
            if name.startswith("_"):
                continue
            count = len(value) if isinstance(value, list | dict) else 1
            rows.append([name, count])
        sections.append(_html_table(["Analyst section", "Returned items"], rows))

    elif record.tool_name == "yfinance_get_financials" and isinstance(payload, dict):
        rows = []
        for name, value in payload.items():
            field_count = len(value) if isinstance(value, dict) else 0
            rows.append([name, field_count, record.arguments.get("frequency")])
        sections.append(_html_table(["Statement", "Line items", "Frequency"], rows))

    elif record.tool_name == "yfinance_get_option_dates" and isinstance(payload, list):
        sections.append(
            '<div class="token-list">' + "".join(f"<span>{escape(str(date))}</span>" for date in payload) + "</div>"
        )

    elif record.tool_name == "yfinance_get_option_chain" and isinstance(payload, dict):
        rows = []
        for date, chain in payload.items():
            if isinstance(chain, dict):
                rows.append([date, len(chain.get("calls", [])), len(chain.get("puts", []))])
        sections.append(_html_table(["Expiration", "Calls", "Puts"], rows))

    elif record.tool_name == "yfinance_get_holders" and isinstance(payload, dict):
        rows = [[name, len(value)] for name, value in payload.items() if isinstance(value, list)]
        sections.append(_html_table(["Ownership section", "Rows"], rows))

    raw_blocks = []
    if record.result is not None:
        for content in record.result.content:
            if not isinstance(content, TextContent):
                continue
            try:
                raw_text = json.dumps(json.loads(content.text), indent=2, ensure_ascii=False)
            except json.JSONDecodeError:
                raw_text = content.text
            raw_blocks.append(f"<pre>{escape(raw_text)}</pre>")
    if raw_blocks:
        sections.append("<details><summary>Complete MCP response</summary>" + "".join(raw_blocks) + "</details>")

    artifact_links = [
        f'<a class="artifact" href="{escape(path.name, quote=True)}">{escape(path.name)}</a>'
        for path in record.saved_paths
    ]
    if artifact_links:
        sections.append('<div class="artifacts"><span>Saved artifacts</span>' + "".join(artifact_links) + "</div>")
    return "".join(section for section in sections if section)


def _render_report(
    records: list[CallRecord],
    advertised_tools: list[Any],
    output_dir: Path,
    args: argparse.Namespace,
    server_parameters: StdioServerParameters,
) -> Path:
    report_path = output_dir / "demo-report.html"
    failures = [record for record in records if _record_error(record)]
    tool_order = [tool.name for tool in advertised_tools]
    groups: dict[str, list[CallRecord]] = {}
    for record in records:
        groups.setdefault(_record_group(record), []).append(record)

    group_nav = "".join(f'<a href="#{_slug(group)}">{escape(group)}</a>' for group in groups)
    coverage_rows = []
    catalog_sections = []
    for tool in advertised_tools:
        tool_records = [record for record in records if record.tool_name == tool.name]
        variants = ", ".join(dict.fromkeys(_variant_label(tool.name, record.arguments) for record in tool_records))
        purpose = TOOL_GUIDANCE.get(tool.name, ("", "", "No demo guidance is defined."))[2]
        status = "Covered" if tool_records else "Missing"
        coverage_rows.append(
            [
                f'<a href="#tool-{_slug(tool.name)}"><code>{escape(tool.name)}</code></a>',
                len(tool_records),
                variants,
                purpose,
                status,
            ]
        )
        schema = json.dumps(tool.inputSchema, indent=2, ensure_ascii=False)
        catalog_sections.append(
            f'<article class="catalog-item" id="tool-{_slug(tool.name)}"><div><p class="eyebrow">MCP tool</p>'
            f"<h3><code>{escape(tool.name)}</code></h3><p>{escape(tool.description or 'No description advertised.')}</p></div>"
            f"<details><summary>Supported arguments</summary><pre>{escape(schema)}</pre></details></article>"
        )

    coverage_head = "".join(
        f"<th>{heading}</th>" for heading in ("MCP tool", "Calls", "Variants exercised", "Why use it", "Status")
    )
    coverage_body = "".join(
        "<tr>"
        + "".join(
            f"<td>{value if column == 0 else escape(_format_number(value))}</td>" for column, value in enumerate(row)
        )
        + "</tr>"
        for row in coverage_rows
    )

    call_sections = []
    for group, group_records in groups.items():
        articles = []
        for record in group_records:
            _, _, why = TOOL_GUIDANCE[record.tool_name]
            error = _record_error(record)
            status_class = "error" if error else "success"
            status_text = "Error" if error else "Live response"
            articles.append(
                f'<article class="call" id="call-{record.number}"><header><div><p class="eyebrow">Call {record.number:02d}</p>'
                f"<h3>{escape(_display_title(record.tool_name, record.arguments, record.context))}</h3>"
                f'<code>{escape(record.tool_name)}</code></div><span class="status {status_class}">{status_text}</span></header>'
                f'<p class="why">{escape(why)}</p><div class="call-grid"><div><h4>Arguments used</h4>'
                f"<pre>{escape(json.dumps(record.arguments, indent=2, ensure_ascii=False))}</pre></div>"
                f'<div><h4>What came back</h4><p class="summary">{escape(record.summary)}</p>'
                f"{f'<p class="error-message">{escape(error)}</p>' if error else ''}</div></div>"
                f'{_structured_result_html(record)}<a class="back" href="#top">Back to index ↑</a></article>'
            )
        call_sections.append(
            f'<section class="feature-group" id="{_slug(group)}"><div class="section-heading"><p class="eyebrow">Feature group</p>'
            f"<h2>{escape(group)}</h2><p>{len(group_records)} live MCP calls</p></div>{''.join(articles)}</section>"
        )

    generated = datetime.now().astimezone().isoformat(timespec="seconds")
    command = _server_command_text(server_parameters)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>yfinance-mcp — Complete Feature Demo</title>
  <style>
    :root {{ --ink:#14211b; --paper:#f4f1e8; --panel:#fffdf7; --muted:#66716b; --line:#d7d4c9; --accent:#137b57; --error:#aa3b2f; }}
    * {{ box-sizing:border-box; }} html {{ scroll-behavior:smooth; scroll-padding-top:7rem; }}
    body {{ margin:0; color:var(--ink); background:var(--paper); font-family:"Segoe UI",system-ui,sans-serif; line-height:1.55; }}
    a {{ color:var(--accent); }} code, pre {{ font-family:"Cascadia Code","SFMono-Regular",Consolas,monospace; }}
    .hero {{ min-height:66svh; display:grid; align-content:end; padding:clamp(2rem,7vw,7rem); color:#f8f7ef;
      background:linear-gradient(115deg,rgba(6,27,20,.96),rgba(13,72,52,.78)),radial-gradient(circle at 80% 10%,#3ba77b,transparent 42%); }}
    .hero h1 {{ margin:.15rem 0 1rem; max-width:13ch; font-family:Georgia,serif; font-size:clamp(3.3rem,9vw,8rem); line-height:.88; letter-spacing:-.055em; }}
    .hero .lede {{ max-width:52rem; font-size:clamp(1rem,2vw,1.3rem); color:#dbe9df; }}
    .eyebrow {{ margin:0; color:var(--accent); font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }}
    .hero .eyebrow {{ color:#86dab9; }}
    .run-strip {{ display:flex; flex-wrap:wrap; gap:1.5rem 3rem; padding:1.15rem clamp(1.5rem,7vw,7rem); background:#0d2b20; color:#dbe9df; }}
    .run-strip span {{ display:block; color:#85a698; font-size:.7rem; text-transform:uppercase; letter-spacing:.1em; }}
    .index {{ position:sticky; top:0; z-index:10; padding:1rem clamp(1.5rem,7vw,7rem); background:rgba(244,241,232,.94); border-bottom:1px solid var(--line); backdrop-filter:blur(14px); }}
    .index strong {{ margin-right:1rem; }} .index a {{ display:inline-block; margin:.25rem 1rem .25rem 0; text-decoration:none; font-weight:650; }}
    .index a.active {{ color:var(--ink); text-decoration:underline; text-decoration-thickness:2px; text-underline-offset:5px; }}
    main {{ width:min(1180px,calc(100% - 3rem)); margin:0 auto; }}
    section {{ padding:clamp(4rem,8vw,8rem) 0; border-bottom:1px solid var(--line); }}
    .section-heading {{ display:grid; grid-template-columns:1fr auto; gap:1rem; align-items:end; margin-bottom:2.5rem; }}
    h2 {{ margin:.2rem 0 0; font-family:Georgia,serif; font-size:clamp(2.2rem,5vw,4.4rem); letter-spacing:-.04em; }}
    h3 {{ margin:.15rem 0 .45rem; font-size:clamp(1.35rem,3vw,2rem); line-height:1.15; }} h4 {{ margin:0 0 .7rem; }}
    .coverage table, table {{ width:100%; border-collapse:collapse; }} th {{ text-align:left; color:var(--muted); font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; }}
    th,td {{ padding:.85rem .7rem; border-bottom:1px solid var(--line); vertical-align:top; }} .table-wrap {{ overflow-x:auto; margin:1.25rem 0; }}
    .call {{ padding:2.8rem 0; border-top:1px solid var(--line); }} .call > header {{ display:flex; justify-content:space-between; gap:2rem; align-items:flex-start; }}
    .status {{ flex:0 0 auto; padding:.35rem .65rem; border:1px solid currentColor; border-radius:999px; font-size:.75rem; font-weight:750; }}
    .status.success {{ color:var(--accent); }} .status.error,.error-message {{ color:var(--error); }} .why,.summary {{ max-width:72ch; }}
    .call-grid {{ display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.35fr); gap:clamp(1.5rem,5vw,5rem); margin:1.8rem 0; }}
    pre {{ margin:0; padding:1rem; overflow:auto; background:#e9e6dc; border-left:3px solid var(--accent); font-size:.82rem; line-height:1.45; white-space:pre-wrap; }}
    details {{ margin:1.2rem 0; }} summary {{ cursor:pointer; font-weight:750; color:var(--accent); }} details pre {{ max-height:38rem; margin-top:.8rem; }}
    figure {{ margin:1.8rem 0; }} figure img {{ display:block; width:100%; height:auto; border:1px solid var(--line); }} figcaption {{ margin-top:.4rem; color:var(--muted); font-size:.8rem; }}
    .token-list {{ display:flex; flex-wrap:wrap; gap:.45rem; margin:1rem 0; }} .token-list span {{ padding:.3rem .55rem; background:#e2e9e2; border-radius:.25rem; font-family:monospace; }}
    .artifacts {{ display:flex; flex-wrap:wrap; gap:.6rem; align-items:center; margin:1rem 0; }} .artifacts span {{ color:var(--muted); }}
    .artifact {{ padding:.3rem .55rem; border:1px solid var(--line); text-decoration:none; }} .back {{ display:inline-block; margin-top:1.5rem; font-size:.85rem; }}
    .catalog-item {{ display:grid; grid-template-columns:1fr 1fr; gap:clamp(1.5rem,5vw,5rem); padding:2rem 0; border-top:1px solid var(--line); }}
    footer {{ padding:3rem clamp(1.5rem,7vw,7rem); background:var(--ink); color:#cbd7d0; }}
    @media (max-width:760px) {{ .hero {{ min-height:58svh; }} .call-grid,.catalog-item,.section-heading {{ grid-template-columns:1fr; }} .index {{ position:relative; }} .call > header {{ display:block; }} .status {{ display:inline-block; margin-top:1rem; }} }}
    @media (prefers-reduced-motion:no-preference) {{ .call {{ opacity:0; transform:translateY(16px); transition:opacity .45s ease,transform .45s ease; }} .call.visible {{ opacity:1; transform:none; }} }}
  </style>
</head>
<body id="top">
  <header class="hero"><p class="eyebrow">Live MCP capability report</p><h1>yfinance-mcp</h1>
    <p class="lede">Every advertised tool, its supported arguments, the exact calls made, and the real Yahoo Finance responses—organized for exploration.</p></header>
  <div class="run-strip"><div><span>Generated</span>{escape(generated)}</div><div><span>Stock</span>{escape(args.symbol)}</div>
    <div><span>Fund look-through</span>{escape(args.fund_symbol)}</div><div><span>Advertised tools</span>{len(tool_order)}</div>
    <div><span>Live calls</span>{len(records)}</div><div><span>Errors</span>{len(failures)}</div></div>
  <nav class="index" aria-label="Feature index"><strong>Index</strong><a href="#coverage">Coverage</a><a href="#catalog">Tool catalog</a>{group_nav}</nav>
  <main>
    <section id="coverage" class="coverage"><div class="section-heading"><div><p class="eyebrow">Server inventory</p><h2>Complete coverage</h2></div>
      <p>The client enumerated the server first, then exercised every advertised tool.</p></div>
      <p><strong>Server command:</strong> <code>{escape(command)}</code></p>
      <div class="table-wrap"><table><thead><tr>{coverage_head}</tr></thead><tbody>{coverage_body}</tbody></table></div></section>
    <section id="catalog"><div class="section-heading"><div><p class="eyebrow">Authoritative interface</p><h2>Tool catalog</h2></div>
      <p>Descriptions and argument schemas advertised by this live MCP server.</p></div>{"".join(catalog_sections)}</section>
    {"".join(call_sections)}
  </main>
  <footer>This report is generated by a standalone MCP client. It imports neither <code>yfinance</code> nor <code>yfmcp</code> server internals.</footer>
  <script>
    const links=[...document.querySelectorAll('.index a')];
    const observed=[...document.querySelectorAll('main > section')];
    const navObserver=new IntersectionObserver(entries=>{{for(const entry of entries){{if(entry.isIntersecting){{links.forEach(link=>link.classList.toggle('active',link.hash==='#'+entry.target.id));}}}}}},{{rootMargin:'-20% 0px -70%'}});
    observed.forEach(section=>navObserver.observe(section));
    const calls=[...document.querySelectorAll('.call')];
    const revealObserver=new IntersectionObserver(entries=>entries.forEach(entry=>{{if(entry.isIntersecting){{entry.target.classList.add('visible');revealObserver.unobserve(entry.target);}}}}),{{rootMargin:'0px 0px -8%'}});
    calls.forEach(call=>revealObserver.observe(call));
  </script>
</body>
</html>"""
    report_path.write_text(html, encoding="utf-8")
    return report_path


def _print_dry_run(args: argparse.Namespace) -> None:
    calls = _demo_calls(args.symbol, args.fund_symbol, args.sector)
    print(f"The live demo will make up to {len(calls) + FUND_LOOKTHROUGH_LIMIT + 3} MCP calls.")
    for number, (tool_name, arguments) in enumerate(calls, start=1):
        print(f"[{number:02d}] {tool_name}: {json.dumps(arguments, sort_keys=True)}")
    next_number = len(calls) + 1
    for holding_number in range(1, FUND_LOOKTHROUGH_LIMIT + 1):
        print(
            f"[{next_number:02d}] yfinance_get_ticker_info: "
            f'{{"symbol": "<SCHD look-through holding {holding_number}>"}}'
        )
        next_number += 1
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
    total_calls = len(calls) + FUND_LOOKTHROUGH_LIMIT + 3
    workflow_failures: list[str] = []
    records: list[CallRecord] = []
    advertised_tools: list[Any] = []

    async with (
        stdio_client(server_parameters) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        listed_tools = await session.list_tools()
        advertised_tools = list(listed_tools.tools)
        available_tools = {tool.name for tool in listed_tools.tools}
        missing_tools = sorted(EXPECTED_TOOL_NAMES - available_tools)
        planned_tools = {tool_name for tool_name, _ in calls} | {"yfinance_get_option_chain"}
        unplanned_tools = sorted(available_tools - planned_tools)
        if missing_tools:
            print(f"MCP server is missing expected tools: {', '.join(missing_tools)}", file=sys.stderr)
            return 1
        if unplanned_tools:
            print(
                f"MCP server advertises tools the demo does not exercise: {', '.join(unplanned_tools)}",
                file=sys.stderr,
            )
            return 1
        print(f"Connected to yfinance-mcp with {len(available_tools)} tools.")
        print(f"Server command: {_server_command_text(server_parameters)}")
        print(f"A readable report and complete artifacts will be saved in {output_dir}")

        option_dates_record: CallRecord | None = None
        for call_number, (tool_name, arguments) in enumerate(calls, start=1):
            record = await _call_tool(session, output_dir, call_number, total_calls, tool_name, arguments)
            records.append(record)
            if tool_name == "yfinance_get_option_dates":
                option_dates_record = record

        lookthrough_record = next(
            (
                record
                for record in records
                if record.tool_name == "yfinance_get_fund_data" and record.arguments.get("sections") == ["top_holdings"]
            ),
            None,
        )
        holdings = _fund_top_holdings(lookthrough_record)
        if not holdings:
            message = f"No underlying holdings were returned for {args.fund_symbol}; look-through cannot continue."
            print(message, file=sys.stderr)
            workflow_failures.append(message)
        for holding_number, holding in enumerate(holdings, start=1):
            weight = holding.get("weight")
            weight_text = f", {float(weight) * 100:.2f}% weight" if isinstance(weight, int | float) else ""
            context = f"Fund look-through {holding_number}: {holding['name']} ({holding['symbol']}{weight_text})"
            record = await _call_tool(
                session,
                output_dir,
                len(records) + 1,
                total_calls,
                "yfinance_get_ticker_info",
                {"symbol": holding["symbol"]},
                context=context,
            )
            records.append(record)

        expiration_date = _first_option_date(option_dates_record)
        for option_type in ("all", "calls", "puts"):
            chain_arguments: dict[str, Any] = {"symbol": args.symbol, "option_type": option_type}
            if expiration_date is not None:
                chain_arguments["expiration_date"] = expiration_date
            record = await _call_tool(
                session,
                output_dir,
                len(records) + 1,
                total_calls,
                "yfinance_get_option_chain",
                chain_arguments,
            )
            records.append(record)

    report_path = _render_report(records, advertised_tools, output_dir, args, server_parameters)
    failures = [record for record in records if _record_error(record)]
    print(f"\nReport: {report_path}")
    print(f"Demo finished with {len(failures)} MCP failure(s) and {len(workflow_failures)} workflow failure(s).")
    return 1 if failures or workflow_failures else 0


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
