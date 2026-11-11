from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(tool_name: str, level: str = "INFO") -> logging.Logger:
    project_root = Path(__file__).resolve().parents[2]
    log_dir = project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(tool_name)
    logger.setLevel(getattr(logging, str(level).upper(), logging.INFO))
    logger.propagate = False

    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        stream = logging.StreamHandler()
        stream.setLevel(getattr(logging, str(level).upper(), logging.INFO))
        stream.setFormatter(formatter)
        logger.addHandler(stream)

        file_handler = RotatingFileHandler(log_dir / f"{tool_name}.log", maxBytes=1_000_000, backupCount=3)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
