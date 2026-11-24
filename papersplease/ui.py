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

_ANSI = re.compile(r"\x1b(?:\[[0-9;?]*[a-zA-Z]|\][^\x1b\x07]*(?:\x1b\\|\x07))")
_ANSI_TOKEN_RE = re.compile(r"(\x1b(?:\[[0-9;?]*[a-zA-Z]|\][^\x1b\x07]*(?:\x1b\\|\x07)))")
_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "reverse": "\033[7m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "white": "\033[37m",
    "blue": "\033[34m",
    "reset": "\033[0m",
}

_WIN_VT_CACHE: dict[int, bool] = {}


def _stream(stream: TextIO | None = None) -> TextIO:
    return stream if stream is not None else sys.stdout


def _enable_windows_vt(stream: TextIO | None = None) -> bool:
    """Enable ANSI Virtual Terminal Processing on Windows conhost/cmd.exe.

    On Windows 10/11 legacy cmd.exe, ANSI escapes render as raw ``←[1m`` unless
    ``ENABLE_VIRTUAL_TERMINAL_PROCESSING`` (0x0004) is enabled on the console
    handle. Returns True when VT sequences are supported on ``stream``.
    """
    target = _stream(stream)
    if not getattr(target, "isatty", lambda: False)():
        return False
    if sys.platform != "win32":
        return True

    handle_id = -12 if target is sys.stderr else -11
    if handle_id in _WIN_VT_CACHE:
        return _WIN_VT_CACHE[handle_id]

    ok = False
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        handle = kernel32.GetStdHandle(handle_id)
        if handle and handle != -1:
            mode = ctypes.c_uint32()
            if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
                vt_flag = 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
                if (mode.value & vt_flag) or kernel32.SetConsoleMode(handle, mode.value | vt_flag):
                    ok = True
    except Exception:  # noqa: BLE001 - ctypes failure simply disables VT escapes
        ok = False

    if not ok and (
        os.environ.get("WT_SESSION")
        or os.environ.get("ANSICON")
        or os.environ.get("TERM_PROGRAM")
        or os.environ.get("TERM", "").lower().startswith(("xterm", "vt", "ansi", "screen", "tmux"))
    ):
        ok = True

    _WIN_VT_CACHE[handle_id] = ok
    return ok


def vt_enabled(stream: TextIO | None = None) -> bool:
    """True when ``stream`` is a TTY that understands ANSI VT escape codes."""
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    return _enable_windows_vt(stream)


def color_enabled(stream: TextIO | None = None) -> bool:
    """Resolve the colour mode: explicit env var, NO_COLOR, then TTY + VT."""
    mode = os.environ.get("PAPERSPLEASE_COLOR", "auto").strip().lower()
    if mode in {"never", "0", "false", "no", "off"}:
        return False
    if mode in {"always", "1", "true", "yes", "on"}:
        if sys.platform == "win32":
            _enable_windows_vt(stream)
        return True
    if "NO_COLOR" in os.environ:
        return False
    if os.environ.get("TERM", "").lower() == "dumb":
        return False
    target = _stream(stream)
    if not getattr(target, "isatty", lambda: False)():
        return False
    if sys.platform == "win32":
        return _enable_windows_vt(target)
    return True


def _is_legacy_win_console(stream: TextIO | None = None) -> bool:
    """True when running in legacy Windows conhost rather than WT or a pipe."""
    if sys.platform != "win32":
        return False
    target = _stream(stream)
    if not getattr(target, "isatty", lambda: False)():
        return False
    return not any(
        os.environ.get(var)
        for var in ("WT_SESSION", "TERM_PROGRAM", "ConEmuANSI", "ANSICON")
    )


def unicode_enabled(stream: TextIO | None = None) -> bool:
    mode = os.environ.get("PAPERSPLEASE_ASCII", "").strip().lower()
    if mode in {"1", "true", "yes", "on"}:
        return False
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

    def __enter__(self) -> Progress:  # noqa: PYI034 - Self requires Python 3.11+
        if not self.enabled:
            return self
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self) -> None:
        frames = Theme.auto(self.stream).spinner_frames
        index = 0
        started = time.monotonic()
        while not self._stop.wait(0.16):
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


class LivePanel:
    """An in-place, bordered live stage dashboard on stderr (TTY + VT only).

    Redraws in the exact same terminal position using ANSI cursor-up + clear,
    showing live per-source hit counts and stage checkmarks, then cleanly
    erases itself before the final report prints. Completely silent off-TTY.
    """

    def __init__(
        self,
        title: str,
        subtitle: str = "",
        stages: list[tuple[str, str]] | None = None,
        *,
        stream: TextIO | None = None,
        clear_on_exit: bool = True,
        show_brand: bool = True,
    ):
        self.title = title
        self.subtitle = subtitle
        self.stream = stream or sys.stderr
        self.clear_on_exit = clear_on_exit
        self.show_brand = show_brand
        self.enabled = vt_enabled(self.stream)
        self.theme = Theme.auto(self.stream)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = time.monotonic()
        self._frame = 0
        self._last_lines = 0
        self._stages: list[dict[str, str]] = [
            {"key": key, "label": label, "state": "pending", "detail": ""}
            for key, label in (
                stages
                or [
                    ("parse", "Parse intent"),
                    ("search", "Search sources"),
                    ("read", "Read abstracts"),
                    ("screen", "Screen scope"),
                    ("access", "Resolve access"),
                ]
            )
        ]
        self._sources: dict[str, str] = {}

    def _stage(self, key: str) -> dict[str, str] | None:
        for item in self._stages:
            if item["key"] == key:
                return item
        return None

    def set_stage(self, key: str, state: str, detail: str = "") -> None:
        with self._lock:
            item = self._stage(key)
            if item is None:
                item = {"key": key, "label": key.title(), "state": state, "detail": detail}
                self._stages.append(item)
            else:
                item["state"] = state
                if detail or state != "running":
                    item["detail"] = detail

    def event(self, stage: str, data: dict) -> None:
        """Pipeline progress callback: updates stage states and source chips."""
        with self._lock:
            if stage == "parse":
                facets = " + ".join(data.get("facets") or [])
                years = ""
                if data.get("year_from") or data.get("year_to"):
                    years = f" ({data.get('year_from') or 'any'}-{data.get('year_to') or 'any'})"
                item = self._stage("parse")
                if item:
                    item["state"] = "done"
                    item["detail"] = (facets or "parsed") + years
            elif stage == "search_start":
                for name in data.get("sources") or []:
                    self._sources.setdefault(name, "running")
                item = self._stage("search")
                if item:
                    item["state"] = "running"
                    item["detail"] = self._format_sources()
            elif stage == "source_done":
                name = str(data.get("source") or "")
                count = int(data.get("count") or 0)
                ok = bool(data.get("ok", True))
                self._sources[name] = f"ok:{count}" if ok else f"warn:{count}"
                item = self._stage("search")
                if item:
                    item["state"] = "running"
                    item["detail"] = self._format_sources()
            elif stage == "search_done":
                item = self._stage("search")
                if item:
                    item["state"] = "done"
                    cands = int(data.get("candidates") or 0)
                    chips = self._format_sources()
                    item["detail"] = f"{cands} candidates  {chips}".strip()
            elif stage == "read":
                done = int(data.get("done") or 0)
                total = int(data.get("total") or 0)
                item = self._stage("read")
                if item:
                    item["state"] = "done" if (total == 0 or done >= total) else "running"
                    item["detail"] = f"{done}/{total} abstracts" if total else "all abstracts present"
            elif stage == "screen_done":
                item = self._stage("screen")
                if item:
                    item["state"] = "done"
                    sep = self.theme.sep
                    item["detail"] = (
                        f"{data.get('in_scope', 0)} in scope {sep} "
                        f"{data.get('adjacent', 0)} adjacent {sep} "
                        f"{data.get('out_of_scope', 0)} rejected"
                    )
            elif stage == "access":
                done = int(data.get("done") or 0)
                total = int(data.get("total") or 0)
                item = self._stage("access")
                if item:
                    item["state"] = "done" if (total == 0 or done >= total) else "running"
                    item["detail"] = f"{done}/{total} links" if total else "no links needed"
            elif stage == "done":
                for item in self._stages:
                    if item["state"] == "running":
                        item["state"] = "done"

    def _format_sources(self) -> str:
        parts: list[str] = []
        spinner = self.theme.spinner_frames
        spin = spinner[self._frame % len(spinner)]
        for name, status in self._sources.items():
            short = name.replace("semanticscholar", "s2").replace("europepmc", "epmc")
            if status == "running":
                parts.append(f"{short} {self.theme.cyan(spin)}")
            elif status.startswith("ok:"):
                count = status.split(":", 1)[1]
                parts.append(f"{short} {self.theme.green(self.theme.check + count)}")
            else:
                count = status.split(":", 1)[1]
                parts.append(f"{short} {self.theme.yellow('!' + count)}")
        return "  ".join(parts)

    def render_lines(self, width: int | None = None) -> list[str]:
        width = width or max(40, min(96, shutil.get_terminal_size((96, 24)).columns))
        inner = max(20, width - 4)
        t = self.theme
        elapsed = f"{time.monotonic() - self._started:4.1f}s"
        tl, tr, bl, br, h, v = t.box_chars
        frames = t.spinner_frames
        spin = frames[self._frame % len(frames)]

        head_right = f" {t.dim(elapsed)} "
        max_left = max(14, width - 4 - visible_len(head_right))
        if self.show_brand:
            title_shown = truncate(self.title, max(4, max_left - 18), unicode=t.unicode)
            head_left = f" {t.paint('PapersPlease', 'bold', 'cyan')} {t.dim(t.sep)} {t.dim(title_shown)} "
        else:
            title_shown = truncate(self.title, max(4, max_left - 4), unicode=t.unicode)
            head_left = f" {t.bold(title_shown)} "
        if visible_len(head_left) > max_left:
            head_left = truncate_ansi(head_left, max_left, unicode=t.unicode)
        fill_len = max(1, width - 3 - visible_len(head_left) - visible_len(head_right))
        top = t.dim(tl + h) + head_left + t.dim(h * fill_len) + head_right + t.dim(tr)

        def box_line(content: str) -> str:
            vis = visible_len(content)
            if vis > inner:
                content = truncate_ansi(content, inner, unicode=t.unicode)
                vis = visible_len(content)
            pad = " " * max(0, inner - vis)
            return f"{t.dim(v)} {content}{pad} {t.dim(v)}"

        lines = [top]
        if self.subtitle:
            lines.append(box_line(t.bold(truncate(self.subtitle, inner, unicode=t.unicode))))
            lines.append(box_line(""))

        label_w = max((len(s["label"]) for s in self._stages), default=12) + 2
        for stage in self._stages:
            st = stage["state"]
            if st == "done":
                glyph = t.green(t.check)
                lbl = t.bold(stage["label"].ljust(label_w))
            elif st == "running":
                glyph = t.cyan(spin)
                lbl = t.cyan(stage["label"].ljust(label_w))
            elif st == "error":
                glyph = t.red(t.cross)
                lbl = t.red(stage["label"].ljust(label_w))
            else:
                glyph = t.dim(t.ring)
                lbl = t.dim(stage["label"].ljust(label_w))
            detail = stage["detail"]
            if stage["key"] == "search" and st == "running" and self._sources:
                detail = self._format_sources()
            if not detail and st == "pending":
                detail = t.dim("waiting")
            row = f" {glyph} {lbl}{detail}"
            lines.append(box_line(row))

        lines.append(t.dim(bl + h * (width - 2) + br))
        return lines

    def render_snapshot(self, width: int | None = None) -> list[str]:
        with self._lock:
            self._frame += 1
            return self.render_lines(width=width)

    def _draw(self) -> None:
        with self._lock:
            self._frame += 1
            lines = self.render_lines()
            buf = ""
            if self._last_lines > 0:
                buf += f"\x1b[{self._last_lines}A\r\x1b[J"
            buf += "\n".join(lines) + "\n"
            self._last_lines = len(lines)
        self.stream.write(buf)
        self.stream.flush()

    def _loop(self) -> None:
        self._draw()
        while not self._stop.wait(0.20):
            self._draw()

    def __enter__(self) -> LivePanel:  # noqa: PYI034 - Self requires Python 3.11+
        if not self.enabled:
            return self
        self._started = time.monotonic()
        self.stream.write("\x1b[?25l")
        self.stream.flush()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        if not self.enabled:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=0.5)
        with self._lock:
            if self.clear_on_exit and self._last_lines > 0:
                self.stream.write(f"\x1b[{self._last_lines}A\r\x1b[J")
                self._last_lines = 0
            self.stream.write("\x1b[?25h")
            self.stream.flush()


def live_panel(
    title: str,
    subtitle: str = "",
    stages: list[tuple[str, str]] | None = None,
    *,
    stream: TextIO | None = None,
) -> LivePanel:
    return LivePanel(title, subtitle, stages, stream=stream)


@dataclass(frozen=True)
class Theme:
    enabled: bool = False
    unicode: bool = True
    rich_unicode: bool = True

    @classmethod
    def auto(cls, stream: TextIO | None = None) -> Theme:
        uni = unicode_enabled(stream)
        return cls(
            enabled=color_enabled(stream),
            unicode=uni,
            rich_unicode=uni and not _is_legacy_win_console(stream),
        )

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
        return "◐" if (self.unicode and self.rich_unicode) else "~"

    @property
    def arrow(self) -> str:
        return "→" if self.unicode else "->"

    @property
    def check(self) -> str:
        return "✓" if (self.unicode and self.rich_unicode) else "ok"

    @property
    def cross(self) -> str:
        return "×" if self.unicode else "x"

    @property
    def star(self) -> str:
        return "★" if (self.unicode and self.rich_unicode and sys.platform != "win32") else "*"

    @property
    def pointer(self) -> str:
        return "▸" if self.unicode else ">"

    @property
    def scroll_thumb(self) -> str:
        return "┃" if self.unicode else "#"

    @property
    def box_chars(self) -> tuple[str, str, str, str, str, str]:
        if not self.unicode:
            return ("+", "+", "+", "+", "-", "|")
        if self.rich_unicode:
            return ("╭", "╮", "╰", "╯", "─", "│")
        return ("┌", "┐", "└", "┘", "─", "│")

    @property
    def spinner_frames(self) -> list[str]:
        if self.unicode and self.rich_unicode:
            return ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        return ["-", "\\", "|", "/"]

    @property
    def sep(self) -> str:
        return "·" if self.unicode else "|"

    @property
    def ge(self) -> str:
        return "≥" if self.unicode else ">="

    @property
    def dash(self) -> str:
        return "—" if self.unicode else "-"

    @property
    def ellipsis(self) -> str:
        return "…" if self.unicode else "..."

    @property
    def times(self) -> str:
        return "×" if self.unicode else "x"

    def bar(self, fraction: float, width: int = 8) -> str:
        clamped = max(0.0, min(1.0, fraction))
        filled = round(clamped * width)
        empty = max(0, width - filled)
        if self.unicode:
            return self.green("█" * filled) + self.dim("░" * empty)
        return self.green("#" * filled) + self.dim("-" * empty)


def truncate(text: str, width: int, *, unicode: bool = True) -> str:
    """Truncate plain text to ``width`` visible cells."""
    text = " ".join(text.split())
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    ell = "…" if unicode else "..."
    if width <= len(ell):
        return text[:width]
    return text[: max(0, width - len(ell))].rstrip() + ell


def truncate_ansi(text: str, width: int, *, unicode: bool = True) -> str:
    """Truncate ANSI-styled text to ``width`` visible cells without stripping colour."""
    if width <= 0:
        return ""
    if visible_len(text) <= width:
        return text
    ell = "…" if unicode else "..."
    if width <= len(ell):
        return _ANSI.sub("", text)[:width]
    limit = width - len(ell)
    out: list[str] = []
    vis = 0
    sgr_open = False
    osc_open = False
    for part in _ANSI_TOKEN_RE.split(text):
        if not part:
            continue
        if _ANSI.fullmatch(part):
            if part.startswith("\x1b]8;;"):
                osc_open = part not in ("\x1b]8;;\x1b\\", "\x1b]8;;\x07")
            elif part.startswith("\x1b[") and part.endswith("m"):
                sgr_open = part != "\x1b[0m"
            out.append(part)
            continue
        take = limit - vis
        if take <= 0:
            break
        if len(part) <= take:
            out.append(part)
            vis += len(part)
        else:
            out.append(part[:take].rstrip())
            vis += take
            break
    if osc_open:
        out.append("\x1b]8;;\x1b\\")
    if sgr_open:
        out.append("\x1b[0m")
    out.append(ell)
    return "".join(out)


def osc8_link(url: str, label: str, enabled: bool) -> str:
    """Wrap ``label`` in a clickable OSC 8 terminal hyperlink when enabled."""
    if not enabled or not url:
        return label
    return f"\x1b]8;;{url}\x1b\\{label}\x1b]8;;\x1b\\"


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
        while len(word) > limit:
            if current:
                lines.append(" ".join(current))
                current = []
                current_len = 0
                limit = rest_limit
            lines.append(word[:limit])
            word = word[limit:]
            limit = rest_limit
        if not word:
            continue
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
