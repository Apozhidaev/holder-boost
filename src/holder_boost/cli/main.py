"""``holder-boost`` CLI entry point."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import typer
from motor.motor_asyncio import AsyncIOMotorDatabase
from structlog.contextvars import bind_contextvars, clear_contextvars

from holder_boost.cli.serve_cmd import run_serve
from holder_boost.cli.sync_cmd import run_sync
from holder_boost.cli.train_cmd import run_train
from holder_boost.config import get_settings
from holder_boost.http import HTTPError, make_client
from holder_boost.logging import configure_logging, get_logger
from holder_boost.storage.client import MongoConnectionError, ensure_indexes, get_database

app = typer.Typer(
    name="holder-boost",
    help="Polymarket No-outcome screening and analysis.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Initialize logging once for any subcommand."""
    settings = get_settings()
    configure_logging(settings.logging)


@asynccontextmanager
async def _command_session(*, command_name: str) -> AsyncIterator[AsyncIOMotorDatabase]:
    """Common per-command setup: Mongo connect + ensure_indexes."""
    log = get_logger(f"cli.{command_name}")
    try:
        db = await get_database()
    except MongoConnectionError as exc:
        log.error("mongo connection failed", error=str(exc))
        raise typer.Exit(code=1) from exc

    await ensure_indexes(db)

    try:
        yield db
    except HTTPError as exc:
        log.error("http failure", error=str(exc))
        raise typer.Exit(code=1) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        log.error("unexpected error", error=str(exc), exc_info=True)
        raise typer.Exit(code=1) from exc


@app.command()
def sync(
    max_pages: int | None = typer.Option(
        None,
        "--max-pages",
        min=1,
        help="Stop poly-house /markets pagination after N pages (for testing).",
    ),
) -> None:
    """poly-house ``sync``: resolved markets load and insert into ``markets``."""
    clear_contextvars()
    bind_contextvars(source="cli.sync")
    asyncio.run(_run_sync(max_pages=max_pages))


async def _run_sync(*, max_pages: int | None = None) -> None:
    async with (
        _command_session(command_name="sync") as db,
        make_client() as client,
    ):
        await run_sync(db, client, max_pages=max_pages)


@app.command()
def train(
    skip_tuning: bool = typer.Option(
        False,
        "--skip-tuning",
        help="Skip Optuna hyperparameter search (faster smoke runs).",
    ),
) -> None:
    """Train CatBoost on holder distribution rows; save model to ``model_dir``."""
    clear_contextvars()
    bind_contextvars(source="cli.train")
    asyncio.run(_run_train(skip_tuning=skip_tuning))


async def _run_train(*, skip_tuning: bool = False) -> None:
    async with _command_session(command_name="train") as db:
        await run_train(db, skip_tuning=skip_tuning)


@app.command()
def serve() -> None:
    """Run FastAPI server with ``POST /predict``."""
    clear_contextvars()
    bind_contextvars(source="cli.serve")
    run_serve()


if __name__ == "__main__":
    app()
