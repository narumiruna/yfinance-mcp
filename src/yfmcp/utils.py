import json

from yfmcp.types import ErrorCode


def dump_json(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def create_error_response(message: str, error_code: ErrorCode = "UNKNOWN_ERROR", details: dict | None = None) -> str:
    """Create a structured error response.

    Args:
        message: Human-readable error message
        error_code: Machine-readable error code for client handling
        details: Optional additional error details

    Returns:
        JSON string with error information
    """
    error_obj = {"error": message, "error_code": error_code}
    if details:
        error_obj["details"] = details
    return dump_json(error_obj)
