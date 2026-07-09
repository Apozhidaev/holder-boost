"""``holder-boost`` CLI entry point."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import typer
from structlog.contextvars import bind_contextvars, clear_contextvars

from holder_boost.cli.dataset_cmd import run_dataset
from holder_boost.cli.serve_cmd import run_serve
from holder_boost.cli.train_cmd import run_train
from holder_boost.config import get_settings
from holder_boost.http import HTTPError, make_client
from holder_boost.logging import configure_logging, get_logger

app = typer.Typer(
    name="holder-boost",
    help="Polymarket outcome prediction with CatBoost from holder distribution.",
    no_args_is_help=True,
)


@app.callback()
def _root() -> None:
    """Initialize logging once for any subcommand."""
    settings = get_settings()
    configure_logging(settings.logging)


@asynccontextmanager
async def _http_session(*, command_name: str) -> AsyncIterator[None]:
    """Common per-command HTTP error handling."""
    log = get_logger(f"cli.{command_name}")
    try:
        yield
    except HTTPError as exc:
        log.error("http failure", error=str(exc))
        raise typer.Exit(code=1) from exc
    except typer.Exit:
        raise
    except Exception as exc:
        log.error("unexpected error", error=str(exc), exc_info=True)
        raise typer.Exit(code=1) from exc


@app.command()
def dataset(
    max_pages: int | None = typer.Option(
        None,
        "--max-pages",
        min=1,
        help="Stop poly-house /markets pagination after N pages (for testing).",
    ),
) -> None:
    """Fetch poly-house data and save training parquet to ``dataset_dir``."""
    clear_contextvars()
    bind_contextvars(source="cli.dataset")
    asyncio.run(_run_dataset(max_pages=max_pages))


async def _run_dataset(*, max_pages: int | None = None) -> None:
    async with _http_session(command_name="dataset"), make_client() as client:
        await run_dataset(client, max_pages=max_pages)


@app.command()
def train(
    skip_tuning: bool = typer.Option(
        False,
        "--skip-tuning",
        help="Skip Optuna hyperparameter search (faster smoke runs).",
    ),
) -> None:
    """Train CatBoost from saved dataset; save model to ``model_dir``."""
    clear_contextvars()
    bind_contextvars(source="cli.train")
    try:
        run_train(skip_tuning=skip_tuning)
    except (FileNotFoundError, ValueError) as exc:
        log = get_logger("cli.train")
        log.error("train failed", error=str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        log = get_logger("cli.train")
        log.error("unexpected error", error=str(exc), exc_info=True)
        raise typer.Exit(code=1) from exc


@app.command()
def serve() -> None:
    """Run FastAPI server with ``POST /predict``."""
    clear_contextvars()
    bind_contextvars(source="cli.serve")
    run_serve()


if __name__ == "__main__":
    app()
