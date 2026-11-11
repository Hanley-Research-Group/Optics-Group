from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .exceptions import ConfigurationError


ALLOWED_TOP_LEVEL_KEYS = {"logging", "DMD", "app"}
ALLOWED_LOGGING_KEYS = {"log_level", "console", "file"}
ALLOWED_DMD_KEYS = {"version", "library_dir", "picture_time_us", "duration_s", "once"}
ALLOWED_APP_KEYS = {"output_dir", "pattern_dir"}


@dataclass
class LoggingConfig:
    log_level: str = "INFO"
    console: bool = True
    file: bool = True


@dataclass
class DMDConfig:
    version: str = "4.2"
    library_dir: Optional[str] = None
    picture_time_us: int = 10000
    duration_s: int = 0
    once: bool = False


@dataclass
class AppConfig:
    output_dir: str = "data"
    pattern_dir: str = "data/patterns"
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    DMD: DMDConfig = field(default_factory=DMDConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "logging": asdict(self.logging),
            "DMD": asdict(self.DMD),
            "app": {"output_dir": self.output_dir, "pattern_dir": self.pattern_dir},
        }


def _validate_mapping(name: str, mapping: Dict[str, Any], allowed: set[str]) -> None:
    """Reject unknown configuration keys early to keep the config schema explicit."""
    unknown = set(mapping) - allowed
    if unknown:
        raise ConfigurationError(f"Unknown keys in '{name}': {sorted(unknown)}")


def _coerce_config(raw: Dict[str, Any]) -> AppConfig:
    """Convert a raw YAML mapping into the application's strongly typed config model."""
    if not isinstance(raw, dict):
        raise ConfigurationError("Configuration file must contain a mapping at the top level.")

    _validate_mapping("root", raw, ALLOWED_TOP_LEVEL_KEYS)

    logging_cfg = raw.get("logging", {})
    dmd_cfg = raw.get("DMD", {})
    app_cfg = raw.get("app", {})

    _validate_mapping("logging", logging_cfg, ALLOWED_LOGGING_KEYS)
    _validate_mapping("DMD", dmd_cfg, ALLOWED_DMD_KEYS)
    _validate_mapping("app", app_cfg, ALLOWED_APP_KEYS)

    config = AppConfig(
        output_dir=str(app_cfg.get("output_dir", "data")),
        pattern_dir=str(app_cfg.get("pattern_dir", "data/patterns")),
        logging=LoggingConfig(
            log_level=str(logging_cfg.get("log_level", "INFO")).upper(),
            console=bool(logging_cfg.get("console", True)),
            file=bool(logging_cfg.get("file", True)),
        ),
        DMD=DMDConfig(
            version=str(dmd_cfg.get("version", "4.2")),
            library_dir=dmd_cfg.get("library_dir"),
            picture_time_us=int(dmd_cfg.get("picture_time_us", 10000)),
            duration_s=int(dmd_cfg.get("duration_s", 0)),
            once=bool(dmd_cfg.get("once", False)),
        ),
    )
    return config


def load_config(path: Optional[str | Path] = None) -> AppConfig:
    """Load the default config file or return the in-memory defaults when none exists."""
    config_file = Path(path) if path else Path(__file__).resolve().parents[3] / "config" / "default.yaml"
    if not config_file.exists():
        return AppConfig()

    with config_file.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return _coerce_config(raw)


def _apply_cli_override(data: Dict[str, Any], key: str, value: Any) -> None:
    """Apply one override to the config dictionary using a clear, readable dispatch table."""
    handlers = {
        "log_level": lambda payload, item: payload["logging"].__setitem__("log_level", str(item).upper()),
        "picture_time_us": lambda payload, item: payload["DMD"].__setitem__("picture_time_us", int(item)),
        "duration": lambda payload, item: payload["DMD"].__setitem__("duration_s", int(item)),
        "once": lambda payload, item: payload["DMD"].__setitem__("once", bool(item)),
        "out": lambda payload, item: payload["app"].__setitem__("output_dir", str(item)),
        "patterns": lambda payload, item: payload["app"].__setitem__("pattern_dir", str(item)),
    }

    handler = handlers.get(key)
    if handler is None:
        raise ConfigurationError(f"Unsupported CLI override '{key}'")
    handler(data, value)


def apply_cli_overrides(config: AppConfig, overrides: Dict[str, Any]) -> AppConfig:
    """Merge CLI arguments over the loaded config without mutating the original object."""
    if not overrides:
        return config

    data = config.to_dict()
    for key, value in overrides.items():
        _apply_cli_override(data, key, value)

    return _coerce_config(data)
