"""Repository for the ``logs`` Mongo collection (PLAN.md §5).

Strategy: each ``>= INFO`` structlog event lands in an in-memory buffer
through :func:`buffer_event`; the CLI calls :func:`flush` once at the end
of the command. This is the simplest variant per PLAN.md §5; if it ever
becomes a bottleneck we'll revisit.

If the Mongo flush itself fails we just print the traceback to stderr and
move on — stderr already saw the events live, and we explicitly do not
want to loop on logging errors (PLAN.md §7).
"""

from __future__ import annotations

import sys
import traceback
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from holder_boost.storage.client import LOGS_COLLECTION

_buffer: list[dict[str, Any]] = []


def buffer_event(event: dict[str, Any]) -> None:
    """Append a structlog event dict to the pending Mongo write buffer."""
    _buffer.append(event)


def buffer_size() -> int:
    return len(_buffer)


async def flush(db: AsyncIOMotorDatabase) -> int:
    """Write buffered events to Mongo. Returns the number of inserted docs.

    On failure: prints the traceback to stderr and returns 0 — never raises.
    """
    if not _buffer:
        return 0
    pending = list(_buffer)
    _buffer.clear()
    try:
        result = await db[LOGS_COLLECTION].insert_many(pending)
        return len(result.inserted_ids)
    except Exception:
        print("warning: failed to flush logs to Mongo", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 0
