from __future__ import annotations

import functools
import sys
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class ALP4Error(Exception):
    exit_code = 1

    def __str__(self) -> str:
        return self.args[0] if self.args else self.__class__.__name__


class DMDCamError(ALP4Error):
    pass


class ConfigurationError(ALP4Error):
    exit_code = 2


class DependencyError(ALP4Error):
    exit_code = 3


class PatternError(ALP4Error):
    exit_code = 4


class DMDHardwareError(ALP4Error):
    exit_code = 5


class WriteError(ALP4Error):
    exit_code = 7


def guarded(func: F) -> F:
    """Wrap CLI entry points so ALP4-specific failures exit cleanly with a stable code."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ALP4Error as exc:
            print(f"{exc}", file=sys.stderr)
            raise SystemExit(exc.exit_code) from exc
        except KeyboardInterrupt:
            print("Interrupted", file=sys.stderr)
            raise SystemExit(130)
        except Exception as exc:
            print(f"Unexpected error: {exc}", file=sys.stderr)
            raise SystemExit(1) from exc

    return wrapper
