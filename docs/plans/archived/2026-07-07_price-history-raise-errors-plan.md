# Price History `raise_errors=True` Plan

## Goal

Make `get_price_history` distinguish "fetch failed" from "genuinely no data": pass `raise_errors=True` to `ticker.history()` so Yahoo-side failures raise exceptions instead of being swallowed into an empty DataFrame and misreported as `NO_DATA`. Add price-history-specific classification so rate limiting and retryable transport failures return `NETWORK_ERROR`, Yahoo upstream/status/auth/blocking failures do not become `NO_DATA`, and invalid symbols/no-price/no-data requests remain `NO_DATA`. Success criteria: rate-limit failures return `NETWORK_ERROR`, invalid symbols or no-data queries return `NO_DATA`, non-no-data Yahoo history failures return `API_ERROR`/`NETWORK_ERROR` as appropriate, and all tests pass.

## Context

- `src/yfmcp/server.py:860-868`: `ticker.history()` currently does not pass `raise_errors`; yfinance defaults to `raise_errors=False`, which swallows all fetch errors and returns an empty DataFrame, falling into the `df.empty` branch at `server.py:889` and misreporting `NO_DATA` (reproduced on 2026-07-07, see issue #134).
- 2026-07-07 reproduction record: using `CURL_CA_BUNDLE=/dev/null` to simulate the issue #134 reporter's broken-SSL environment, `ticker.history(period="1d", interval="1m")` silently returns an empty df (the log only prints the misleading "possibly delisted; no price data found" with no mention of SSL), while `yf.Search("AAPL")` directly raises `curl_cffi`'s `SSLError` (curl error 77) — exactly matching the two symptoms in the issue (history misreporting NO_DATA, search reporting an SSL error). With `raise_errors=True`, `history()` raises the same `SSLError` in that environment.
- The inheritance chain of `curl_cffi.requests.exceptions.SSLError` includes `OSError` (SSLError → ConnectionError → RequestException → CurlError → OSError), so it will be caught by `_RETRYABLE_YFINANCE_EXCEPTIONS` (which includes `OSError`) → reported as `NETWORK_ERROR` with the real exception message attached — exactly the behavior we want.
- **Note**: yfinance 1.5.1 has deprecated the `raise_errors` parameter of `history()` (calls emit a `DeprecationWarning` recommending the global `yf.config.debug.hide_exceptions = False` instead). This plan still uses the per-call `raise_errors=True`, because the global `hide_exceptions = False` would change the error behavior of every yfinance call in the process (search, lookup, base quote, market, etc.), violating the Non-Goals below; see Risks for the deprecation risk.
- Existing error handling: `_RETRYABLE_YFINANCE_EXCEPTIONS` (`server.py:32`, includes `YFRateLimitError`) → `NETWORK_ERROR`; any other `Exception` → `API_ERROR`.
- Relevant exceptions in yfinance 1.5.1 under `raise_errors=True` (all inherit from `YFException`):
  - `YFRateLimitError` — rate limiting; already in the retryable list.
  - `YFPricesMissingError` — broad history failure wrapper for both genuine no-price/no-data cases and Yahoo chart upstream failures (`debug_info` can include `Yahoo status_code = ...` or `Yahoo error = ...`), so it must be classified by `debug_info`/message rather than blindly mapped to `NO_DATA`.
  - `YFTzMissingError` — missing ticker timezone; semantically still `NO_DATA` for this tool.
  - `YFInvalidPeriodError` — invalid period; a user-input problem, semantically also a fit for the `NO_DATA` guidance message.

## Non-Goals

- Do not change error handling of other tools (`get_ticker_info`, `get_financials`, etc.).
- Do not add a retry mechanism; only do error classification.

## Plan

- [x] In `get_price_history` in `src/yfmcp/server.py`, add `raise_errors=True` to `ticker.history(...)`; verify: `grep -n "raise_errors=True" src/yfmcp/server.py` has a hit.
- [x] Add imports for `YFPricesMissingError`, `YFTzMissingError`, and `YFInvalidPeriodError` from `yfinance.exceptions`; do not catch broad `YFTickerMissingError`, because `YFPricesMissingError` can represent both no-data and upstream Yahoo chart failures; verify: `uv run ruff check src/yfmcp/server.py` passes.
- [x] Add a small price-history error helper/classifier in `src/yfmcp/server.py` so `NO_DATA` details consistently include `symbol`, `period`, `interval`, `prepost`, and optional `exception`; classify `YFPricesMissingError` conservatively: known no-data/delisted messages or missing chart data → `NO_DATA`, rate-limit-like messages (`Too Many Requests`, `rate limit`) → `NETWORK_ERROR`, and non-no-data Yahoo `status_code`/`chart.error` messages such as auth/crumb/blocking failures → `API_ERROR`; verify: targeted unit tests below pass.
- [x] In the `get_price_history` except chain, keep `_RETRYABLE_YFINANCE_EXCEPTIONS` first, then handle `(YFTzMissingError, YFInvalidPeriodError)` as `NO_DATA`, then handle `YFPricesMissingError` via the classifier, and keep the generic `Exception` branch as `API_ERROR`; verify: `uv run pytest tests/test_server_unit.py -k price_history -q` passes.
- [x] Keep the `df.empty` → `NO_DATA` branch as a safety net (edge cases may still return an empty df under `raise_errors=True`); update it only to reuse the shared no-data helper if one was added; verify: `test_get_price_history_no_data_includes_prepost_detail` still passes.
- [x] Update/add tests in `tests/test_server_unit.py`:
  - Add `raise_errors=True` to the existing `history.assert_called_once_with(...)` assertions (around lines 300 and 326).
  - New: `history` side_effect of `YFRateLimitError` → returns `error_code == "NETWORK_ERROR"` with rate-limit wording.
  - New: `history` side_effect of `YFTzMissingError("AAPL")` → returns `error_code == "NO_DATA"`.
  - New: `history` side_effect of `YFInvalidPeriodError("AAPL", "bad", "1d, 5d")` → returns `error_code == "NO_DATA"`.
  - New: `history` side_effect of `YFPricesMissingError("AAPL", "(period=1d)")` or a delisted/no-data debug string → returns `error_code == "NO_DATA"`.
  - New: `history` side_effect of `YFPricesMissingError("AAPL", "(period=1d) (Yahoo status_code = 401)")` or `Yahoo error = "Invalid Crumb"` → returns `error_code == "API_ERROR"` (not `NO_DATA`).
  - New: `history` side_effect of `YFPricesMissingError("AAPL", "(period=1d) (Yahoo error = \"Too Many Requests\")")` → returns `error_code == "NETWORK_ERROR"`.
  - In at least one new `NO_DATA` exception test, assert `details` contains `symbol`, `period`, `interval`, `prepost`, and `exception`; avoid asserting exact yfinance exception message text elsewhere.
  - Verify: `uv run pytest tests/ -x -q` all pass.
- [x] Manual end-to-end verification: call `get_price_history("AAPL", "5d", "1d")` over the real network and get a markdown table (not an error); call with an invalid symbol (e.g. `"NOTAREALTICKER123"`) and get `NO_DATA` rather than `API_ERROR`; verify: record the actual output of both calls.
- [x] Manually simulate the issue #134 SSL failure: run `get_price_history("AAPL", "1d", "1m")` with `CURL_CA_BUNDLE=/dev/null` and confirm it returns `NETWORK_ERROR` (details containing the SSLError message) rather than the misleading `NO_DATA`; verify: record the actual output.
- [x] Run repo validation commands: `uv run ruff check .`, `uv run ty check src tests`, and because `.pre-commit-config.yaml` exists, `prek run -a` (fallback: `pre-commit run -a`); verify: command output shows no errors.

## Risks

- `YFPricesMissingError` classification is coupled to yfinance's current `debug_info`/message strings. Keep the classifier conservative: if a `YFPricesMissingError` is not clearly no-data/delisted or rate-limit-like, return `API_ERROR` rather than `NO_DATA`. Tests should assert `error_code` and structured details, not exact yfinance wording except for synthetic classifier fixtures.
- `YFPricesMissingError` exception text (for example, "possibly delisted") will go into `details.exception`; it is useful signal for LLM users and needs no filtering, but tests should avoid brittle assertions on exact real yfinance messages.
- `raise_errors=True` behaves differently for multi-symbol requests in yfinance, but this tool only queries a single symbol at a time, so it is unaffected.
- **`raise_errors` is deprecated (yfinance 1.5.1)**: future versions may remove it; at that point switch to setting `yf.config.debug.hide_exceptions = False` at server startup and re-evaluate the impact on other tools (all tools' except chains already have a generic `Exception` → `API_ERROR` fallback, so nothing will crash, but error response contents will change). Watch the changelog when upgrading yfinance.
- Each `history(raise_errors=True)` call emits a `DeprecationWarning` to stderr; for an MCP stdio server, stderr is log-only and does not affect the protocol, and Python's default warning filter shows it only once per location — acceptable, no extra filtering needed.

## Completion Checklist

- [x] `ticker.history()` carries `raise_errors=True`, confirmed by `grep -n "raise_errors=True" src/yfmcp/server.py` and the updated call in `src/yfmcp/server.py`.
- [x] `YFRateLimitError` → `NETWORK_ERROR`, retryable transport errors → `NETWORK_ERROR`, `YFTzMissingError`/`YFInvalidPeriodError` → `NO_DATA`, no-data/delisted `YFPricesMissingError` → `NO_DATA`, rate-limit-like `YFPricesMissingError` → `NETWORK_ERROR`, non-no-data Yahoo status/chart `YFPricesMissingError` → `API_ERROR`, and empty df → `NO_DATA`, confirmed by `uv run pytest tests/ -x -q` (`110 passed`).
- [x] New exception-path responses preserve structured details (`symbol`, `period`, `interval`, `prepost`, and `exception` where applicable), confirmed by assertions in `tests/test_server_unit.py` and `uv run pytest tests/ -x -q` (`110 passed`).
- [x] Real call for AAPL 5d/1d successfully returns a table and an invalid symbol returns `NO_DATA`, confirmed by manual run output: AAPL returned a Markdown table beginning with a `Date` header (`lines=7`), and `NOTAREALTICKER123` returned `error_code: NO_DATA`.
- [x] Simulated SSL failure with `CURL_CA_BUNDLE=/dev/null` returns `NETWORK_ERROR` (not `NO_DATA`), confirmed by manual run output containing curl error 77 in `details.exception`.
- [x] Validation commands `uv run ruff check .`, `uv run ty check src tests`, and `prek run -a` pass, confirmed by command output (`All checks passed!` / hooks `Passed`).
