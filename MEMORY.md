## GOTCHA

- `ticker.options` and `ticker.option_chain()` can raise timeout, connection-related errors, or `YFRateLimitError`; in the options tools these must remain `NETWORK_ERROR` and must not be downgraded to `NO_DATA` or generic `API_ERROR`.
- If a retryable error bucket includes `YFRateLimitError`, the user-facing message must not tell users to check their internet connection; use temporary-network wording or an explicit rate-limit/backoff message instead.

## TASTE
