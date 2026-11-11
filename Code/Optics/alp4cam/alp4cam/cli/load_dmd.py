from __future__ import annotations

import argparse
from pathlib import Path

from alp4cam.core.config import AppConfig, load_config
from alp4cam.core.exceptions import DMDCamError, guarded
from alp4cam.core.logging_setup import configure_logging
from alp4cam.dmd import DMDController
from alp4cam.patterns import discover_pattern_files, load_pattern_stack


@guarded
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="alp4cam-load", description="Load a pattern folder onto the ALP-4.2 DMD and project it.")
    parser.add_argument("--patterns", required=True, help="Directory containing PNG patterns to project.")
    parser.add_argument("--picture-time-us", type=int, default=10000, help="DMD picture time in microseconds.")
    parser.add_argument("--duration", type=int, default=0, help="Duration in seconds to keep the sequence running.")
    parser.add_argument("--once", action="store_true", help="Project the sequence once instead of looping.")
    parser.add_argument("--config", help="Path to a YAML config file.")
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    args = parser.parse_args(argv)

    config: AppConfig = load_config(args.config) if args.config else load_config()
    if args.log_level:
        config.logging.log_level = str(args.log_level).upper()

    logger = configure_logging(
        level=config.logging.log_level,
        console=config.logging.console,
        file=config.logging.file,
    )
    logger.info("Loaded DMD config with picture_time_us=%s, duration=%s, once=%s", args.picture_time_us, args.duration, args.once)

    config.DMD.picture_time_us = args.picture_time_us
    config.DMD.duration_s = args.duration
    config.DMD.once = args.once

    pattern_dir = Path(args.patterns)
    if not pattern_dir.exists():
        raise DMDCamError(f"Pattern directory does not exist: {pattern_dir}")

    patterns = load_pattern_stack(pattern_dir)
    logger.info("Loading %d patterns from %s", len(patterns), pattern_dir)

    controller = DMDController(library_dir=config.DMD.library_dir, version=config.DMD.version)
    controller.load_patterns(patterns, picture_time_us=args.picture_time_us, once=args.once, duration_s=args.duration)
    logger.info("Projected %d patterns from %s", len(patterns), pattern_dir)
    print(f"Projected {len(patterns)} patterns from {pattern_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
