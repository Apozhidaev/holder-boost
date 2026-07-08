"""``holder-boost`` CLI entry point."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import typer
from motor.motor_asyncio import AsyncIOMotorDatabase
from structlog.contextvars import bind_contextvars, clear_contextvars

from holder_boost.cli.predict_cmd import run_predict
from holder_boost.cli.search_cmd import run_search
from holder_boost.cli.search_holder_cmd import run_search_holder
from holder_boost.cli.sync_cmd import run_sync
from holder_boost.cli.train_cmd import run_train
from holder_boost.config import get_settings
from holder_boost.logging import configure_logging, get_logger
from holder_boost.polymarket import ContractError, PolymarketHTTPError, make_client
from holder_boost.storage.client import MongoConnectionError, ensure_indexes, get_database
from holder_boost.storage.repos import logs as logs_repo

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
    """Common per-command setup: Mongo connect + ensure_indexes + log flush."""
    log = get_logger(f"cli.{command_name}")
    db: AsyncIOMotorDatabase | None = None
    try:
        try:
            db = await get_database()
        except MongoConnectionError as exc:
            log.error("mongo connection failed", error=str(exc))
            raise typer.Exit(code=1) from exc

        await ensure_indexes(db)

        try:
            yield db
        except ContractError as exc:
            log.error(f"{command_name} failed", error=str(exc))
            raise typer.Exit(code=1) from exc
        except PolymarketHTTPError as exc:
            log.error("polymarket http failure", error=str(exc))
            raise typer.Exit(code=1) from exc
        except typer.Exit:
            raise
        except Exception as exc:
            log.error("unexpected error", error=str(exc), exc_info=True)
            raise typer.Exit(code=1) from exc
    finally:
        if db is not None:
            await logs_repo.flush(db)


@app.command()
def search() -> None:
    """Screen open Gamma markets by CLOB best ask; write ``csv/search.csv``."""
    clear_contextvars()
    bind_contextvars(source="cli.search")
    asyncio.run(_run_search())


async def _run_search() -> None:
    async with (
        _command_session(command_name="search") as db,
        make_client() as client,
    ):
        await run_search(client, db)


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
def predict() -> None:
    """Compute Yes win rate per tag from ``markets``; upsert ``predicts``."""
    clear_contextvars()
    bind_contextvars(source="cli.predict")
    asyncio.run(_run_predict())


async def _run_predict() -> None:
    async with _command_session(command_name="predict") as db:
        await run_predict(db)


@app.command()
def train(
    skip_tuning: bool = typer.Option(
        False,
        "--skip-tuning",
        help="Skip Optuna hyperparameter search (faster smoke runs).",
    ),
) -> None:
    """Train CatBoost on resolved market take-prices; save model to ``model_dir``."""
    clear_contextvars()
    bind_contextvars(source="cli.train")
    asyncio.run(_run_train(skip_tuning=skip_tuning))


async def _run_train(*, skip_tuning: bool = False) -> None:
    async with _command_session(command_name="train") as db:
        await run_train(db, skip_tuning=skip_tuning)


@app.command("search-holder")
def search_holder() -> None:
    """Find holders with private usernames on open markets; write ``csv/search_holder.csv``."""
    clear_contextvars()
    bind_contextvars(source="cli.search_holder")
    asyncio.run(_run_search_holder())


async def _run_search_holder() -> None:
    log = get_logger("cli.search_holder")
    try:
        async with make_client() as client:
            await run_search_holder(client)
    except ContractError as exc:
        log.error("search-holder failed", error=str(exc))
        raise typer.Exit(code=1) from exc
    except PolymarketHTTPError as exc:
        log.error("polymarket http failure", error=str(exc))
        raise typer.Exit(code=1) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        log.error("unexpected error", error=str(exc), exc_info=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
