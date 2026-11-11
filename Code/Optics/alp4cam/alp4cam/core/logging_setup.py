from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    level: str = "INFO",
    *,
    console: bool = True,
    file: bool = True,
    file_path: str | None = None,
) -> logging.Logger:
    logger = logging.getLogger("alp4cam")
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if file:
        if file_path is None:
            log_dir = Path(__file__).resolve().parents[2] / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            file_path = str(log_dir / "alp4cam.log")

        file_handler = logging.FileHandler(file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
