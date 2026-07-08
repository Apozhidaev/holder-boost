"""``serve`` command: run FastAPI with POST /predict."""

from __future__ import annotations

import uvicorn

from holder_boost.config import get_settings
from holder_boost.logging import configure_logging, get_logger


def run_serve() -> None:
    settings = get_settings()
    configure_logging(settings.logging)
    log = get_logger("cli.serve")
    host = settings.api.host
    port = settings.api.port
    log.info("starting api server", host=host, port=port, model_dir=settings.train.model_dir)
    uvicorn.run(
        "holder_boost.api.app:app",
        host=host,
        port=port,
        log_level=settings.logging.level.lower(),
    )
