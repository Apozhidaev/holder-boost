"""Shared async HTTP client with timeouts and a modest tenacity retry."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_fixed,
)

from holder_boost.config import HttpSettings, get_settings
from holder_boost.logging import get_logger

log = get_logger("polymarket.http")


class PolymarketHTTPError(RuntimeError):
    """Raised after retries on a Polymarket HTTP failure."""


_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)


class _TransientHTTP(Exception):
    """Internal marker for retryable 5xx responses."""

    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"transient {status}: {body[:200]}")
        self.status = status


_RETRYABLE_TYPES = (*_NETWORK_ERRORS, _TransientHTTP)


@asynccontextmanager
async def make_client(settings: HttpSettings | None = None) -> AsyncIterator[httpx.AsyncClient]:
    cfg = settings or get_settings().http
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(cfg.timeout_seconds),
        headers={"User-Agent": "holder-boost/0.1"},
    ) as client:
        yield client


async def _get_json_once(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None,
) -> Any:
    response = await client.get(url, params=params)
    if 500 <= response.status_code < 600:
        log.warning("http 5xx, retrying", url=url, status=response.status_code)
        raise _TransientHTTP(response.status_code, response.text)
    if response.status_code >= 400:
        raise PolymarketHTTPError(f"GET {url} failed: {response.status_code} {response.text[:200]}")
    return response.json()


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    """GET ``url`` and parse JSON, retrying transient failures up to 3x.

    Retries cover network errors and 5xx; 4xx responses fail immediately.
    After exhausting attempts we raise :class:`PolymarketHTTPError`.
    """
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.5),
            retry=retry_if_exception_type(_RETRYABLE_TYPES),
            reraise=False,
        ):
            with attempt:
                return await _get_json_once(client, url, params)
    except RetryError as exc:
        last = exc.last_attempt.exception()
        raise PolymarketHTTPError(f"GET {url} failed after retries: {last}") from last
    raise PolymarketHTTPError(f"GET {url} unreachable")  # pragma: no cover
