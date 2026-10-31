"""Console logging with level filtering and colour.

Kept deliberately tiny — this app has no logging infrastructure requirements
beyond readable startup and request output.
"""

from __future__ import annotations

import sys
from datetime import datetime

from ..core.env import LOG_LEVEL

_LEVELS = {"debug": 10, "info": 20, "warning": 30, "error": 40}

_COLORS = {
    "debug": "\033[90m",
    "info": "\033[36m",
    "warning": "\033[33m",
    "error": "\033[31m",
}
_RESET = "\033[0m"


class ConsoleIO:
    """A minimal levelled logger."""

    def __init__(self, name: str = "talentbook", mode: str | None = None) -> None:
        self.name = name
        self.threshold = _LEVELS.get((mode or LOG_LEVEL), 20)

    def _emit(self, level: str, message: str) -> None:
        if _LEVELS[level] < self.threshold:
            return
        stamp = datetime.now().strftime("%H:%M:%S")
        colour = _COLORS[level]
        stream = sys.stderr if level == "error" else sys.stdout
        print(
            f"{colour}[{stamp}] {level.upper():<7}{_RESET} {self.name}: {message}",
            file=stream,
        )

    def debug(self, message: str) -> None:
        self._emit("debug", message)

    def info(self, message: str) -> None:
        self._emit("info", message)

    def warning(self, message: str) -> None:
        self._emit("warning", message)

    def error(self, message: str, exc_info: bool = False) -> None:
        self._emit("error", message)
        if exc_info:
            import traceback

            traceback.print_exc()


def get_logger(name: str = "talentbook") -> ConsoleIO:
    """Module-level logger factory."""
    return ConsoleIO(name)
