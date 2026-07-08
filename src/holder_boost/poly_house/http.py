"""poly-house HTTP helpers."""

from __future__ import annotations

from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from holder_boost.polymarket.http import PolymarketHTTPError

_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


class _TransientHTTP(Exception):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"transient {status}: {body[:200]}")
        self.status = status


_RETRYABLE_TYPES = (*_NETWORK_ERRORS, _TransientHTTP)


async def _get_json_with_body_once(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None,
    json_body: Any,
) -> Any:
    response = await client.request("GET", url, params=params, json=json_body)
    if 500 <= response.status_code < 600:
        raise _TransientHTTP(response.status_code, response.text)
    if response.status_code >= 400:
        raise PolymarketHTTPError(
            f"GET {url} failed: {response.status_code} {response.text[:200]}"
        )
    return response.json()


async def get_json_with_body(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: Any = None,
) -> Any:
    """GET ``url`` with optional JSON body, parse response JSON with retries."""
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.5),
            retry=retry_if_exception_type(_RETRYABLE_TYPES),
            reraise=False,
        ):
            with attempt:
                return await _get_json_with_body_once(
                    client, url, params=params, json_body=json_body
                )
    except RetryError as exc:
        last = exc.last_attempt.exception()
        raise PolymarketHTTPError(f"GET {url} failed after retries: {last}") from last
    raise PolymarketHTTPError(f"GET {url} unreachable")  # pragma: no cover
