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

log = get_logger("http")


class HTTPError(RuntimeError):
    """Raised after retries on an HTTP failure."""


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


async def _request_json_once(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None,
    json_body: Any | None = None,
) -> Any:
    if json_body is None:
        response = await client.get(url, params=params)
    else:
        response = await client.request("GET", url, params=params, json=json_body)
    if 500 <= response.status_code < 600:
        log.warning("http 5xx, retrying", url=url, status=response.status_code, params=params)
        raise _TransientHTTP(response.status_code, response.text)
    if response.status_code >= 400:
        raise HTTPError(f"GET {url} failed: {response.status_code} {response.text[:200]}")
    return response.json()


async def _request_json_with_retries(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: Any | None = None,
) -> Any:
    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(3),
            wait=wait_fixed(0.5),
            retry=retry_if_exception_type(_RETRYABLE_TYPES),
            reraise=False,
        ):
            with attempt:
                return await _request_json_once(
                    client,
                    url,
                    params=params,
                    json_body=json_body,
                )
    except RetryError as exc:
        last = exc.last_attempt.exception()
        raise HTTPError(f"GET {url} failed after retries: {last}") from last
    raise HTTPError(f"GET {url} unreachable")  # pragma: no cover


async def get_json(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
) -> Any:
    """GET ``url`` and parse JSON, retrying transient failures up to 3x."""
    return await _request_json_with_retries(client, url, params=params)


async def get_json_with_body(
    client: httpx.AsyncClient,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: Any = None,
) -> Any:
    """GET ``url`` with optional JSON body, parse response JSON with retries."""
    return await _request_json_with_retries(
        client,
        url,
        params=params,
        json_body=json_body,
    )
