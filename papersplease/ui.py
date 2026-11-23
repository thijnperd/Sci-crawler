"""Terminal styling: colour, width and wrapping with no dependencies.

Colour is opt-in by context, like every well-behaved Unix tool: enabled only
when stdout is a TTY (or ``PAPERSPLEASE_COLOR=always``), disabled by
``NO_COLOR``, ``TERM=dumb`` or ``PAPERSPLEASE_COLOR=never``. Piped output and
JSON stay free of escape codes.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import threading
import time
from dataclasses import dataclass
from typing import TextIO

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "reset": "\033[0m",
}


def _stream(stream: TextIO | None = None) -> TextIO:
    return stream if stream is not None else sys.stdout


def color_enabled(stream: TextIO | None = None) -> bool:
    """Resolve the colour mode: explicit env var, NO_COLOR, then TTY."""
    mode = os.environ.get("PAPERSPLEASE_COLOR", "auto").strip().lower()
    if mode in {"never", "0", "false", "no", "off"}:
        return False
    if mode in {"always", "1", "true", "yes", "on"}:
        return True
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    return bool(getattr(_stream(stream), "isatty", lambda: False)())


def unicode_enabled(stream: TextIO | None = None) -> bool:
    encoding = (getattr(_stream(stream), "encoding", "") or "").lower()
    return "utf" in encoding


def terminal_width(minimum: int = 68, maximum: int = 100) -> int:
    columns = shutil.get_terminal_size((96, 24)).columns
    return max(minimum, min(maximum, columns))


def visible_len(text: str) -> int:
    return len(_ANSI.sub("", text))


class Progress:
    """A one-line spinner for network waits; stderr only, TTY only.

    It never touches stdout, so `--json` and piped output stay clean. When
    stderr is not a terminal this is a no-op, which is exactly what logs want.
    """

    def __init__(self, label: str, *, stream: TextIO | None = None):
        self.label = label
        self.stream = stream or sys.stderr
        self.enabled = bool(getattr(self.stream, "isatty", lambda: False)())
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "Progress":
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self) -> None:
        frames = (
            ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            if unicode_enabled(self.stream)
            else ["-", "\\", "|", "/"]
        )
        index = 0
        started = time.monotonic()
        while not self._stop.wait(0.12):
            elapsed = time.monotonic() - started
            frame = frames[index % len(frames)]
            index += 1
            self.stream.write(f"\r{frame} {self.label}  {elapsed:4.1f}s")
            self.stream.flush()

    def __exit__(self, *_: object) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        self.stream.write("\r" + " " * (len(self.label) + 12) + "\r")
        self.stream.flush()


def progress(label: str, *, stream: TextIO | None = None) -> Progress:
    return Progress(label, stream=stream)


@dataclass(frozen=True)
class Theme:
    enabled: bool = False
    unicode: bool = True

    @classmethod
    def auto(cls, stream: TextIO | None = None) -> "Theme":
        return cls(enabled=color_enabled(stream), unicode=unicode_enabled(stream))

    def paint(self, text: str, *names: str) -> str:
        if not self.enabled or not names:
            return text
        return "".join(_CODES[name] for name in names) + text + _CODES["reset"]

    def bold(self, text: str) -> str:
        return self.paint(text, "bold")

    def dim(self, text: str) -> str:
        return self.paint(text, "dim")

    def cyan(self, text: str) -> str:
        return self.paint(text, "cyan")

    def green(self, text: str) -> str:
        return self.paint(text, "green")

    def yellow(self, text: str) -> str:
        return self.paint(text, "yellow")

    def red(self, text: str) -> str:
        return self.paint(text, "red")

    def magenta(self, text: str) -> str:
        return self.paint(text, "magenta")

    @property
    def dot(self) -> str:
        return "●" if self.unicode else "*"

    @property
    def ring(self) -> str:
        return "○" if self.unicode else "o"

    @property
    def half(self) -> str:
        return "◐" if self.unicode else "~"

    @property
    def arrow(self) -> str:
        return "↗" if self.unicode else "->"

    @property
    def check(self) -> str:
        return "✓" if self.unicode else "ok"

    @property
    def cross(self) -> str:
        return "✗" if self.unicode else "x"


def truncate(text: str, width: int) -> str:
    """Truncate plain text to ``width`` visible cells."""
    text = " ".join(text.split())
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    return text[: max(0, width - 1)].rstrip() + ("…" if width > 1 else "")


def wrap(text: str, width: int, *, initial: str = "", subsequent: str | None = None) -> list[str]:
    """Wrap text to ``width``, with optional first/subsequent indents.

    Indents may contain ANSI escapes; their *visible* length is what counts.
    """
    subsequent = initial if subsequent is None else subsequent
    body = " ".join((text or "").split())
    if not body:
        return [initial.rstrip()]
    first_limit = max(10, width - visible_len(initial))
    rest_limit = max(10, width - visible_len(subsequent))
    lines: list[str] = []
    current: list[str] = []
    current_len = 0
    limit = first_limit
    for word in body.split(" "):
        if not current:
            current = [word]
            current_len = len(word)
            continue
        if current_len + 1 + len(word) <= limit:
            current.append(word)
            current_len += 1 + len(word)
            continue
        lines.append(" ".join(current))
        current = [word]
        current_len = len(word)
        limit = rest_limit
    if current:
        lines.append(" ".join(current))
    return [
        (initial if index == 0 else subsequent) + line
        for index, line in enumerate(lines)
    ]


def rule(width: int, theme: Theme, char: str | None = None) -> str:
    glyph = char or ("─" if theme.unicode else "-")
    return theme.dim(glyph * max(1, width))


def kv(
    label: str,
    value: str,
    theme: Theme,
    *,
    label_width: int = 8,
    width: int | None = None,
) -> list[str]:
    """A label column plus a wrapped value, aligned on the value."""
    width = width or terminal_width()
    prefix = theme.dim(label.ljust(label_width))
    pad = " " * label_width
    return [
        (prefix if index == 0 else pad) + line
        for index, line in enumerate(wrap(value, max(20, width - label_width)))
    ]


def badge(bucket: str, theme: Theme) -> str:
    """One coloured status badge for a screening bucket."""
    if bucket == "in_scope":
        return theme.paint(f"{theme.dot} IN SCOPE", "green", "bold")
    if bucket == "adjacent":
        return theme.paint(f"{theme.half} ADJACENT", "yellow", "bold")
    return theme.paint(f"{theme.ring} REJECTED", "red", "bold")


def state(text: str, kind: str, theme: Theme) -> str:
    """Colour a short state word (ok/blocked/unreachable/on/off)."""
    painter = {
        "ok": theme.green,
        "on": theme.green,
        "saved": theme.green,
        "blocked": theme.red,
        "failed": theme.red,
        "error": theme.red,
        "unreachable": theme.yellow,
        "off": theme.dim,
        "unknown": theme.dim,
    }.get(kind, theme.yellow)
    return painter(text)
