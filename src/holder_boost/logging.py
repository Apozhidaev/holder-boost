"""structlog setup: stderr only."""

from __future__ import annotations

import logging
import sys
from datetime import UTC, datetime
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from holder_boost.config import LoggingSettings


def _add_utc_timestamp(_: Any, __: str, event_dict: EventDict) -> EventDict:
    event_dict["ts"] = datetime.now(UTC)
    return event_dict


def _stringify_ts_for_render(_: Any, __: str, event_dict: EventDict) -> EventDict:
    """Convert ``ts`` to an ISO string just before rendering to stderr."""
    ts = event_dict.get("ts")
    if isinstance(ts, datetime):
        event_dict["ts"] = ts.isoformat(timespec="milliseconds")
    return event_dict


def configure_logging(settings: LoggingSettings) -> None:
    """Configure ``structlog`` once for the process.

    Idempotent: subsequent calls just re-bind the configuration.
    """
    level = getattr(logging, settings.level)
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
    )
    # httpx is chatty at INFO ("HTTP Request: ..."); demote unless caller wants debug.
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(max(level, logging.WARNING))

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        _add_utc_timestamp,
        structlog.stdlib.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _stringify_ts_for_render,
    ]

    if settings.format == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name) if name else structlog.get_logger()
