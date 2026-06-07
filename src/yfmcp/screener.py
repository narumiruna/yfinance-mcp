from __future__ import annotations

from typing import Any
from typing import cast

from yfinance import EquityQuery
from yfinance import FundQuery
from yfinance.screener.query import Operator

ScreenerQuery = EquityQuery | FundQuery

LEAF_OPERATORS = {"eq", "is-in", "btwn", "gt", "lt", "gte", "lte"}
LOGICAL_OPERATORS = {"and", "or"}


def build_screener_query(query_type: str, query: dict[str, Any]) -> ScreenerQuery:
    """Build a yfinance screener query object from a JSON-serializable query tree."""
    match query_type.lower():
        case "equity":
            return _build_equity_node(query)
        case "fund":
            return _build_fund_node(query)
        case _:
            raise ValueError("query_type must be 'equity' or 'fund' for custom queries")


def _validate_node_shape(node: dict[str, Any]) -> tuple[Operator, list[Any]]:
    if not isinstance(node, dict):
        raise ValueError("Each query node must be an object with 'operator' and 'operands'")

    operator = node.get("operator")
    operands = node.get("operands")

    if not isinstance(operator, str) or not operator.strip():
        raise ValueError("Each query node must include a non-empty string 'operator'")
    if not isinstance(operands, list) or len(operands) == 0:
        raise ValueError("Each query node must include a non-empty list 'operands'")

    normalized_operator = operator.lower()
    if normalized_operator not in LEAF_OPERATORS.union(LOGICAL_OPERATORS):
        valid_operators = sorted(LEAF_OPERATORS.union(LOGICAL_OPERATORS))
        raise ValueError(f"Unsupported operator '{operator.upper()}'. Valid operators: {', '.join(valid_operators)}")

    return cast(Operator, normalized_operator), operands


def _build_equity_node(node: dict[str, Any]) -> EquityQuery:
    normalized_operator, operands = _validate_node_shape(node)

    if normalized_operator in LOGICAL_OPERATORS:
        nested_queries: list[EquityQuery] = []
        for operand in operands:
            if not isinstance(operand, dict):
                raise ValueError(f"Operator '{normalized_operator.upper()}' requires nested query objects")
            nested_queries.append(_build_equity_node(cast(dict[str, Any], operand)))
        return EquityQuery(normalized_operator, cast(Any, nested_queries))

    if normalized_operator in LEAF_OPERATORS:
        if any(isinstance(operand, dict) for operand in operands):
            raise ValueError(f"Operator '{normalized_operator.upper()}' does not accept nested query objects")
        return EquityQuery(normalized_operator, operands)
    raise ValueError(f"Unsupported operator '{normalized_operator.upper()}'")


def _build_fund_node(node: dict[str, Any]) -> FundQuery:
    normalized_operator, operands = _validate_node_shape(node)

    if normalized_operator in LOGICAL_OPERATORS:
        nested_queries: list[FundQuery] = []
        for operand in operands:
            if not isinstance(operand, dict):
                raise ValueError(f"Operator '{normalized_operator.upper()}' requires nested query objects")
            nested_queries.append(_build_fund_node(cast(dict[str, Any], operand)))
        return FundQuery(normalized_operator, cast(Any, nested_queries))

    if normalized_operator in LEAF_OPERATORS:
        if any(isinstance(operand, dict) for operand in operands):
            raise ValueError(f"Operator '{normalized_operator.upper()}' does not accept nested query objects")
        return FundQuery(normalized_operator, operands)
    raise ValueError(f"Unsupported operator '{normalized_operator.upper()}'")
