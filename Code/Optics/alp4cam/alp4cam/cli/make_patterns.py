from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from alp4cam.core.config import AppConfig, load_config
from alp4cam.core.exceptions import ConfigurationError, DMDCamError, guarded
from alp4cam.core.logging_setup import configure_logging
from alp4cam.patterns import generate_hadamard_set, generate_test_patterns, save_pattern_set


def default_output_dir(pattern_type: str, config: AppConfig) -> Path:
    """Build a timestamped directory for the generated pattern batch."""
    root = Path(config.pattern_dir)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"{pattern_type}_{timestamp}"


@guarded
def main(argv=None) -> int:
    """Generate either a small test dataset or a Hadamard pattern set from the CLI arguments."""
    parser = argparse.ArgumentParser(prog="alp4cam-patterns", description="Generate DMD test or Hadamard patterns.")
    parser.add_argument("pattern_type", choices=["test", "hadamard"], help="Type of pattern set to generate.")
    parser.add_argument("--out", help="Directory to write the generated pattern PNG files. Defaults to a new folder inside the configured pattern directory.")
    parser.add_argument("--mode-side", type=int, default=32, help="Size of the mode grid for Hadamard generation.")
    parser.add_argument("--macro", type=int, default=2, help="Macro factor used in the Hadamard set size.")
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
    logger.info("Loaded config for pattern generation (type=%s, log_level=%s)", args.pattern_type, config.logging.log_level)

    output_dir = Path(args.out) if args.out else default_output_dir(args.pattern_type, config)

    if args.pattern_type == "test":
        logger.info("Generating test patterns with side=%s", max(2, args.mode_side))
        patterns = generate_test_patterns(side=max(2, args.mode_side))
    else:
        if args.mode_side <= 0:
            raise ConfigurationError("--mode-side must be > 0")
        logger.info("Generating Hadamard pattern set with mode_side=%s and macro=%s", args.mode_side, args.macro)
        patterns = generate_hadamard_set(mode_side=args.mode_side, macro=args.macro)

    saved = save_pattern_set(patterns, output_dir, prefix=args.pattern_type)
    logger.info("Generated %d patterns in %s", len(saved), output_dir.resolve())
    print(f"Generated {len(saved)} patterns in {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
