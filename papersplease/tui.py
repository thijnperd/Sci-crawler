"""Interactive in-place terminal UI (zero dependencies, Windows + POSIX).

Uses the terminal alternate screen buffer (``\\x1b[?1049h`` / ``\\x1b[?1049l``)
so the interface stays pinned in the exact same place with no scrollback spam.
Every screen is rendered by ``TuiApp.render_lines(cols, rows)`` as pure text,
keeping the rendering logic completely testable without a TTY.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, TextIO

from . import __version__, export, ui
from .config import Settings
from .pipeline import Finding, Report, explain_intent, find, read_one
from .render import render_explain, render_markdown, render_mirrors, render_providers
from .resolver import access_for, download_best, filename_for, mirror_health
from .sources import SOURCE_INFO

_BUCKETS = ("in_scope", "adjacent", "out_of_scope")
_BUCKET_LABELS = {
    "in_scope": "In Scope",
    "adjacent": "Adjacent",
    "out_of_scope": "Rejected",
}

_EXAMPLES = [
    "optimal training for hypertrophy in younger adults since 2015",
    "muscle strength in older adults since 2015",
    "sets and frequency for hypertrophy",
]


def _read_key_windows(timeout: float = 0.12) -> str | None:
    import msvcrt

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if msvcrt.kbhit():
            ch = msvcrt.getwch()
            if ch in ("\x00", "\xe0"):
                code = msvcrt.getwch()
                return {
                    "H": "UP",
                    "P": "DOWN",
                    "K": "LEFT",
                    "M": "RIGHT",
                    "G": "HOME",
                    "O": "END",
                    "I": "PGUP",
                    "Q": "PGDN",
                }.get(code, "")
            if ch == "\r":
                return "ENTER"
            if ch == "\x1b":
                return "ESC"
            if ch == "\t":
                return "TAB"
            if ch in ("\x08", "\x7f"):
                return "BACKSPACE"
            if ch == "\x03":
                return "CTRL_C"
            return ch
        time.sleep(0.015)
    return None


def _read_key_posix(timeout: float = 0.12) -> str | None:
    import select

    fd = sys.stdin.fileno()
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None
    ch = os.read(fd, 1).decode("utf-8", errors="ignore")
    if not ch:
        return None
    if ch == "\x1b":
        ready2, _, _ = select.select([fd], [], [], 0.02)
        if not ready2:
            return "ESC"
        seq = os.read(fd, 2).decode("utf-8", errors="ignore")
        return {
            "[A": "UP",
            "[B": "DOWN",
            "[C": "RIGHT",
            "[D": "LEFT",
            "[H": "HOME",
            "[F": "END",
        }.get(seq, "ESC")
    if ch in ("\r", "\n"):
        return "ENTER"
    if ch == "\t":
        return "TAB"
    if ch in ("\x08", "\x7f"):
        return "BACKSPACE"
    if ch == "\x03":
        return "CTRL_C"
    return ch


def _pad_visible(text: str, width: int) -> str:
    vis = ui.visible_len(text)
    if vis > width:
        plain = ui._ANSI.sub("", text)
        text = ui.truncate(plain, width)
        vis = ui.visible_len(text)
    return text + " " * max(0, width - vis)


class TuiApp:
    """Stateful in-place terminal application for PapersPlease."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        report: Report | None = None,
        initial_intent: str = "",
        find_kwargs: dict[str, Any] | None = None,
        stream: TextIO | None = None,
    ):
        self.settings = settings or Settings.from_env()
        self.stream = stream or sys.stdout
        self.theme = ui.Theme(
            enabled=ui.color_enabled(self.stream),
            unicode=ui.unicode_enabled(self.stream),
        )
        self.report: Report | None = report
        self.intent: str = (report.spec.intent if report else initial_intent).strip()
        self.find_kwargs: dict[str, Any] = dict(find_kwargs or {})

        self.bucket_index: int = 0
        self.selected: dict[str, int] = {b: 0 for b in _BUCKETS}
        self.scroll: dict[str, int] = {b: 0 for b in _BUCKETS}
        self.detail_expanded: bool = False
        self.overlay: str | None = None  # None | "explain" | "mirrors" | "providers" | "help"
        self.overlay_lines: list[str] = []
        self.overlay_scroll: int = 0

        # Inline prompt state: None | "search" | "refine" | "export"
        self.prompt_mode: str | None = None
        self.prompt_buffer: str = ""
        self.status_message: str = ""
        self.status_kind: str = "info"  # info | ok | warn | error

        # Background task state (search / download / mirrors / deep-read)
        self._busy_lock = threading.Lock()
        self._busy_panel: ui.LivePanel | None = None
        self._busy_label: str = ""
        self._running: bool = True

        if self.report is not None:
            self._pick_initial_bucket()

    def _pick_initial_bucket(self) -> None:
        if self.report is None:
            return
        for idx, bucket in enumerate(_BUCKETS):
            if self.report.bucket(bucket):
                self.bucket_index = idx
                return
        self.bucket_index = 0

    @property
    def current_bucket(self) -> str:
        return _BUCKETS[self.bucket_index]

    def current_findings(self) -> list[Finding]:
        if self.report is None:
            return []
        return self.report.bucket(self.current_bucket)

    def current_finding(self) -> Finding | None:
        items = self.current_findings()
        if not items:
            return None
        idx = max(0, min(self.selected[self.current_bucket], len(items) - 1))
        self.selected[self.current_bucket] = idx
        return items[idx]

    def set_status(self, message: str, kind: str = "info") -> None:
        self.status_message = message
        self.status_kind = kind

    @property
    def is_busy(self) -> bool:
        with self._busy_lock:
            return self._busy_panel is not None

    # ------------------------------------------------------------------ actions

    def start_search(self, intent: str) -> None:
        intent = " ".join(intent.split())
        if not intent:
            return
        self.intent = intent
        self.overlay = None
        self.prompt_mode = None
        self.detail_expanded = False
        panel = ui.LivePanel(
            "live pipeline",
            f'"{intent}"',
            stream=self.stream,
            clear_on_exit=False,
        )
        panel.theme = self.theme
        with self._busy_lock:
            self._busy_panel = panel
            self._busy_label = "searching"

        def worker() -> None:
            try:
                kwargs = dict(self.find_kwargs)
                kwargs.pop("on_progress", None)
                rep = find(
                    intent,
                    settings=self.settings,
                    on_progress=panel.event,
                    **kwargs,
                )
                self.report = rep
                self.selected = {b: 0 for b in _BUCKETS}
                self.scroll = {b: 0 for b in _BUCKETS}
                self._pick_initial_bucket()
                self.set_status(
                    f"Loaded {len(rep.in_scope)} in scope, {len(rep.adjacent)} adjacent, "
                    f"{len(rep.out_of_scope)} rejected.",
                    "ok",
                )
            except Exception as exc:
                self.set_status(f"Search failed: {exc}", "error")
            finally:
                with self._busy_lock:
                    self._busy_panel = None
                    self._busy_label = ""

        threading.Thread(target=worker, daemon=True).start()

    def _open_selected_link(self) -> None:
        finding = self.current_finding()
        if not finding:
            self.set_status("No paper selected.", "warn")
            return
        url = None
        if finding.access:
            if finding.access.open_access:
                url = finding.access.open_access[0].get("url")
            elif finding.access.scihub:
                ok_links = [l for l in finding.access.scihub if l.get("state") == "ok"]
                url = (ok_links or finding.access.scihub)[0].get("url")
            elif finding.access.doi_url:
                url = finding.access.doi_url
        if not url and finding.paper.doi:
            url = f"https://doi.org/{finding.paper.doi}"
        if not url:
            self.set_status("No URL available for this paper.", "warn")
            return
        try:
            webbrowser.open(url)
            self.set_status(f"Opened {ui.truncate(url, 60)}", "ok")
        except Exception as exc:
            self.set_status(f"Could not open browser: {exc}", "error")

    def _download_selected(self) -> None:
        finding = self.current_finding()
        if not finding:
            self.set_status("No paper selected.", "warn")
            return
        panel = ui.LivePanel(
            "download",
            ui.truncate(finding.paper.title, 72),
            stages=[("access", "Resolve PDF"), ("save", "Save file")],
            stream=self.stream,
            clear_on_exit=False,
        )
        panel.theme = self.theme
        panel.set_stage("access", "running", "checking OA and Sci-Hub")
        with self._busy_lock:
            self._busy_panel = panel
            self._busy_label = "downloading"

        def worker() -> None:
            try:
                access = finding.access or access_for(finding.paper, self.settings)
                finding.access = access
                panel.set_stage("access", "done", "links ready")
                panel.set_stage("save", "running", "downloading PDF")
                out_dir = Path("papers")
                out_dir.mkdir(parents=True, exist_ok=True)
                dest = out_dir / filename_for(finding.paper)
                outcome = download_best(access, str(dest), self.settings)
                if outcome.get("saved"):
                    panel.set_stage("save", "done", str(outcome.get("path")))
                    self.set_status(f"Saved PDF to {outcome.get('path')} ({outcome.get('source')})", "ok")
                else:
                    panel.set_stage("save", "error", str(outcome.get("note")))
                    self.set_status(f"Download failed: {outcome.get('note')}", "warn")
            except Exception as exc:
                self.set_status(f"Download error: {exc}", "error")
            finally:
                with self._busy_lock:
                    self._busy_panel = None
                    self._busy_label = ""

        threading.Thread(target=worker, daemon=True).start()

    def _deep_read_selected(self) -> None:
        finding = self.current_finding()
        if not finding:
            self.set_status("No paper selected.", "warn")
            return
        target = finding.paper.doi or finding.paper.arxiv_id or finding.paper.title
        panel = ui.LivePanel(
            "reading paper",
            ui.truncate(finding.paper.title, 72),
            stages=[("read", "Fetch full text")],
            stream=self.stream,
            clear_on_exit=False,
        )
        panel.theme = self.theme
        panel.set_stage("read", "running", target)
        with self._busy_lock:
            self._busy_panel = panel
            self._busy_label = "reading"

        def worker() -> None:
            try:
                refreshed = read_one(target, settings=self.settings, deep=True)
                if refreshed and refreshed.abstract:
                    finding.paper.abstract = refreshed.abstract
                    finding.paper.read_note = refreshed.read_note
                    self.detail_expanded = True
                    self.set_status(f"Read updated ({refreshed.read_note or 'abstract loaded'})", "ok")
                else:
                    self.set_status("No additional text found for this paper.", "warn")
            except Exception as exc:
                self.set_status(f"Read failed: {exc}", "error")
            finally:
                with self._busy_lock:
                    self._busy_panel = None
                    self._busy_label = ""

        threading.Thread(target=worker, daemon=True).start()

    def _open_explain_overlay(self) -> None:
        if not self.intent:
            self.set_status("Search an intent first (press /).", "warn")
            return
        payload = explain_intent(
            self.intent,
            settings=self.settings,
            min_scope=self.find_kwargs.get("min_scope"),
            require=self.find_kwargs.get("require") or (),
            exclude=self.find_kwargs.get("exclude") or (),
            require_in=self.find_kwargs.get("require_in") or "",
            since=self.find_kwargs.get("since"),
            until=self.find_kwargs.get("until"),
        )
        width = max(60, min(96, shutil.get_terminal_size((96, 28)).columns - 4))
        text = render_explain(payload, width=width, color=self.theme.enabled)
        self.overlay = "explain"
        self.overlay_lines = text.splitlines()
        self.overlay_scroll = 0

    def _open_providers_overlay(self) -> None:
        rows = []
        for name, info in SOURCE_INFO.items():
            env = info["key_env"]
            rows.append(
                {
                    "source": name,
                    "label": info["label"],
                    "default": name in self.settings.sources,
                    "key_env": env or None,
                    "configured": not env or bool(os.environ.get(env)),
                    "note": info["note"],
                }
            )
        width = max(60, min(96, shutil.get_terminal_size((96, 28)).columns - 4))
        text = render_providers(rows, width=width, color=self.theme.enabled)
        self.overlay = "providers"
        self.overlay_lines = text.splitlines()
        self.overlay_scroll = 0

    def _open_mirrors_overlay(self) -> None:
        panel = ui.LivePanel(
            "Sci-Hub mirrors",
            "Checking mirror reachability",
            stages=[("probe", "Probe mirrors")],
            stream=self.stream,
            clear_on_exit=False,
        )
        panel.theme = self.theme
        panel.set_stage("probe", "running", f"{len(self.settings.mirrors)} configured mirrors")
        with self._busy_lock:
            self._busy_panel = panel
            self._busy_label = "probing mirrors"

        def worker() -> None:
            try:
                statuses = mirror_health(self.settings, force=False)
                width = max(60, min(96, shutil.get_terminal_size((96, 28)).columns - 4))
                text = render_mirrors(statuses, width=width, color=self.theme.enabled)
                self.overlay = "mirrors"
                self.overlay_lines = text.splitlines()
                self.overlay_scroll = 0
            except Exception as exc:
                self.set_status(f"Mirror check failed: {exc}", "error")
            finally:
                with self._busy_lock:
                    self._busy_panel = None
                    self._busy_label = ""

        threading.Thread(target=worker, daemon=True).start()

    def _open_help_overlay(self) -> None:
        t = self.theme
        self.overlay = "help"
        self.overlay_scroll = 0
        self.overlay_lines = [
            t.paint("PapersPlease Interactive Controls", "bold", "cyan"),
            ui.rule(68, t),
            "",
            f"  {t.cyan('↑ / ↓  or  j / k')}     Move selection in the paper list",
            f"  {t.cyan('1 / 2 / 3  or  Tab')}   Switch between In Scope, Adjacent, and Rejected",
            f"  {t.cyan('Enter')}                Toggle expanded abstract & detail view",
            f"  {t.cyan('/')}                    Search a new research intent",
            f"  {t.cyan('f')}                    Refine scope (e.g. min=0.55 since=2018 +term -excl)",
            f"  {t.cyan('e')}                    Inspect scope contract (offline explain)",
            f"  {t.cyan('o')}                    Open best link (OA / Sci-Hub / DOI) in browser",
            f"  {t.cyan('d')}                    Download PDF of selected paper to ./papers",
            f"  {t.cyan('r')}                    Deep-read full text / refresh abstract",
            f"  {t.cyan('x')}                    Export in-scope papers (bibtex / ris / md / json)",
            f"  {t.cyan('p / m')}                View search providers / probe Sci-Hub mirrors",
            f"  {t.cyan('Esc')}                  Close overlay or prompt",
            f"  {t.cyan('q')}                    Quit and restore terminal",
        ]

    def _apply_refine(self, raw: str) -> None:
        tokens = raw.strip().split()
        if not tokens:
            return
        require = list(self.find_kwargs.get("require") or [])
        exclude = list(self.find_kwargs.get("exclude") or [])
        for tok in tokens:
            if tok.startswith("min="):
                try:
                    self.find_kwargs["min_scope"] = float(tok.split("=", 1)[1])
                except ValueError:
                    self.set_status(f"Invalid min_scope: {tok}", "warn")
                    return
            elif tok.startswith("since="):
                try:
                    self.find_kwargs["since"] = int(tok.split("=", 1)[1])
                except ValueError:
                    self.set_status(f"Invalid year: {tok}", "warn")
                    return
            elif tok.startswith("until="):
                try:
                    self.find_kwargs["until"] = int(tok.split("=", 1)[1])
                except ValueError:
                    self.set_status(f"Invalid year: {tok}", "warn")
                    return
            elif tok.startswith("expand="):
                val = tok.split("=", 1)[1]
                if val in ("none", "citations", "references", "both"):
                    self.find_kwargs["expand"] = val
            elif tok.startswith("+") and len(tok) > 1:
                require.append(tok[1:])
            elif tok.startswith("-") and len(tok) > 1:
                exclude.append(tok[1:])
        self.find_kwargs["require"] = require
        self.find_kwargs["exclude"] = exclude
        if self.intent:
            self.start_search(self.intent)
        else:
            self.set_status("Refinement saved. Press / to search.", "ok")

    def _apply_export(self, raw: str) -> None:
        if self.report is None:
            self.set_status("Nothing to export yet.", "warn")
            return
        fmt = (raw.strip().lower() or "bibtex")
        ext_map = {"bibtex": "bib", "ris": "ris", "csljson": "csl.json", "md": "md", "json": "json"}
        if fmt not in ext_map:
            self.set_status("Supported formats: bibtex, ris, csljson, md, json", "warn")
            return
        out_path = Path(f"papersplease_export.{ext_map[fmt]}")
        if fmt == "json":
            content = json.dumps(self.report.to_dict(abstract_chars=1000), ensure_ascii=False, indent=2)
        elif fmt == "md":
            content = render_markdown(self.report)
        else:
            content = export.render(self.report.in_scope, fmt)
        out_path.write_text(content, encoding="utf-8")
        self.set_status(f"Exported {len(self.report.in_scope)} in-scope papers to {out_path}", "ok")

    # ------------------------------------------------------------ key handling

    def handle_key(self, key: str) -> bool:
        """Process one keypress. Returns False when the app should exit."""
        if key == "CTRL_C":
            self._running = False
            return False

        # While typing in the bottom prompt bar
        if self.prompt_mode is not None:
            if key == "ESC":
                self.prompt_mode = None
                self.prompt_buffer = ""
                return True
            if key == "ENTER":
                mode = self.prompt_mode
                val = self.prompt_buffer
                self.prompt_mode = None
                self.prompt_buffer = ""
                if mode == "search":
                    self.start_search(val)
                elif mode == "refine":
                    self._apply_refine(val)
                elif mode == "export":
                    self._apply_export(val)
                return True
            if key == "BACKSPACE":
                self.prompt_buffer = self.prompt_buffer[:-1]
                return True
            if len(key) == 1 and key.isprintable():
                self.prompt_buffer += key
            return True

        # While viewing an overlay
        if self.overlay is not None:
            if key in ("ESC", "q", "ENTER"):
                self.overlay = None
                self.overlay_lines = []
                self.overlay_scroll = 0
                return True
            if key in ("DOWN", "j"):
                self.overlay_scroll = min(max(0, len(self.overlay_lines) - 5), self.overlay_scroll + 1)
            elif key in ("UP", "k"):
                self.overlay_scroll = max(0, self.overlay_scroll - 1)
            return True

        if self.is_busy:
            if key == "q":
                self._running = False
                return False
            return True

        if key == "q":
            self._running = False
            return False
        if key == "?":
            self._open_help_overlay()
            return True
        if key == "/":
            self.prompt_mode = "search"
            self.prompt_buffer = self.intent
            return True
        if key == "p":
            self._open_providers_overlay()
            return True
        if key == "m":
            self._open_mirrors_overlay()
            return True
        if key == "e":
            self._open_explain_overlay()
            return True
        if key == "f":
            self.prompt_mode = "refine"
            self.prompt_buffer = ""
            return True
        if key == "x":
            self.prompt_mode = "export"
            self.prompt_buffer = "bibtex"
            return True

        # Home screen quick-launch examples
        if self.report is None:
            if key == "ENTER":
                self.prompt_mode = "search"
                self.prompt_buffer = ""
                return True
            if key in ("1", "2", "3"):
                idx = int(key) - 1
                if idx < len(_EXAMPLES):
                    self.start_search(_EXAMPLES[idx])
            return True

        # Report navigation
        if key in ("1", "2", "3"):
            self.bucket_index = int(key) - 1
            self.detail_expanded = False
            return True
        if key == "TAB":
            self.bucket_index = (self.bucket_index + 1) % len(_BUCKETS)
            self.detail_expanded = False
            return True
        if key in ("DOWN", "j"):
            items = self.current_findings()
            if items:
                b = self.current_bucket
                self.selected[b] = min(len(items) - 1, self.selected[b] + 1)
            return True
        if key in ("UP", "k"):
            b = self.current_bucket
            self.selected[b] = max(0, self.selected[b] - 1)
            return True
        if key == "ENTER":
            self.detail_expanded = not self.detail_expanded
            return True
        if key == "ESC":
            self.detail_expanded = False
            return True
        if key == "o":
            self._open_selected_link()
            return True
        if key == "d":
            self._download_selected()
            return True
        if key == "r":
            self._deep_read_selected()
            return True
        return True

    # ------------------------------------------------------------ frame render

    def render_lines(self, cols: int, rows: int) -> list[str]:
        """Render the complete screen as exactly ``rows`` lines of width ``cols``."""
        cols = max(56, cols)
        rows = max(14, rows)
        t = self.theme
        lines: list[str] = []

        # 1. Top Header Bar (2 lines)
        brand = t.paint(f" PapersPlease {__version__} ", "bold", "cyan")
        if self.intent:
            right = t.dim(ui.truncate(f'"{self.intent}"', max(16, cols - 26)))
        else:
            right = t.dim("in-place literature screener")
        gap = max(1, cols - ui.visible_len(brand) - ui.visible_len(right) - 1)
        lines.append(_pad_visible(brand + " " * gap + right, cols))
        lines.append(ui.rule(cols, t))

        # Reserve 2 bottom lines (status/prompt + keybind bar)
        body_height = max(6, rows - 4)

        with self._busy_lock:
            busy_panel = self._busy_panel

        if busy_panel is not None:
            busy_panel._frame += 1
            panel_w = min(cols - 2, 92)
            panel_lines = busy_panel.render_lines(width=panel_w)
            pad_top = max(0, (body_height - len(panel_lines)) // 2)
            pad_left = " " * max(0, (cols - panel_w) // 2)
            body: list[str] = [""] * pad_top
            for pl in panel_lines:
                body.append(pad_left + pl)
        elif self.overlay is not None:
            body = self._render_overlay(cols, body_height)
        elif self.report is None:
            body = self._render_welcome(cols, body_height)
        else:
            body = self._render_report(cols, body_height)

        body = body[:body_height]
        while len(body) < body_height:
            body.append("")
        lines.extend(_pad_visible(line, cols) for line in body)

        # 3. Bottom Rule / Prompt / Status Bar (2 lines)
        if self.prompt_mode is not None:
            prompt_labels = {
                "search": "Search intent",
                "refine": "Refine (min=0.6 since=2018 +req -excl)",
                "export": "Export format (bibtex|ris|csljson|md|json)",
            }
            lbl = t.paint(f" {prompt_labels[self.prompt_mode]}: ", "bold", "cyan")
            cursor = t.paint(" ", "reverse") if t.enabled else "_"
            lines.append(_pad_visible(lbl + self.prompt_buffer + cursor, cols))
            lines.append(_pad_visible(t.dim("  Enter submit  ·  Esc cancel"), cols))
        else:
            if self.status_message:
                painter = {
                    "ok": t.green,
                    "warn": t.yellow,
                    "error": t.red,
                }.get(self.status_kind, t.dim)
                lines.append(_pad_visible(" " + painter(ui.truncate(self.status_message, cols - 2)), cols))
            else:
                lines.append(ui.rule(cols, t))
            lines.append(_pad_visible(self._keybar(cols), cols))

        return lines[:rows]

    def _keybar(self, cols: int) -> str:
        t = self.theme
        if self.overlay is not None:
            items = [("↑↓", "scroll"), ("Esc/q", "close")]
        elif self.report is None:
            items = [("/", "search"), ("1-3", "examples"), ("p", "providers"), ("m", "mirrors"), ("?", "help"), ("q", "quit")]
        else:
            items = [
                ("↑↓", "move"),
                ("Tab", "bucket"),
                ("Enter", "detail"),
                ("/", "search"),
                ("f", "refine"),
                ("e", "explain"),
                ("o", "open"),
                ("d", "pdf"),
                ("x", "export"),
                ("?", "help"),
                ("q", "quit"),
            ]
        chunks = [f"{t.cyan(k)} {t.dim(v)}" for k, v in items]
        out = " "
        for chunk in chunks:
            cand = (out + "  " + chunk) if out.strip() else (" " + chunk)
            if ui.visible_len(cand) > cols:
                break
            out = cand
        return out

    def _render_welcome(self, cols: int, height: int) -> list[str]:
        t = self.theme
        box_w = min(cols - 4, 76)
        inner = box_w - 4
        tl, tr, bl, br, h, v = ("╭", "╮", "╰", "╯", "─", "│") if t.unicode else ("+", "+", "+", "+", "-", "|")

        def box_row(content: str) -> str:
            vis = ui.visible_len(content)
            pad = " " * max(0, inner - vis)
            return f"{t.dim(v)} {content}{pad} {t.dim(v)}"

        card = [
            t.dim(tl + h * (box_w - 2) + tr),
            box_row(t.paint("Search, read, and screen the literature in one place.", "bold", "cyan")),
            box_row(t.dim("Press / or Enter to type a research intent, or pick an example:")),
            box_row(""),
        ]
        for idx, example in enumerate(_EXAMPLES, start=1):
            card.append(box_row(f"  {t.cyan(f'[{idx}]')} {ui.truncate(example, inner - 6)}"))
        card.append(box_row(""))
        card.append(box_row(t.dim("Zero dependencies  " + t.sep + "  Europe PMC, Crossref, arXiv, OpenAlex, S2")))
        card.append(t.dim(bl + h * (box_w - 2) + br))

        pad_top = max(0, (height - len(card)) // 2)
        pad_left = " " * max(0, (cols - box_w) // 2)
        return [""] * pad_top + [pad_left + line for line in card]

    def _render_overlay(self, cols: int, height: int) -> list[str]:
        t = self.theme
        title = f" {self.overlay.upper()} " if self.overlay else ""
        header = t.paint(title, "bold", "cyan") + t.dim("  (Esc to return)")
        visible = self.overlay_lines[self.overlay_scroll : self.overlay_scroll + max(1, height - 2)]
        return [header, ""] + ["  " + line for line in visible]

    def _render_report(self, cols: int, height: int) -> list[str]:
        assert self.report is not None
        t = self.theme
        rep = self.report
        out: list[str] = []

        # Tab bar + scope summary
        tabs: list[str] = []
        for idx, bucket in enumerate(_BUCKETS, start=1):
            count = len(rep.bucket(bucket))
            glyph = {"in_scope": t.dot, "adjacent": t.half, "out_of_scope": t.ring}[bucket]
            raw = f" [{idx}] {glyph} {_BUCKET_LABELS[bucket]} ({count}) "
            if idx - 1 == self.bucket_index:
                color = {"in_scope": "green", "adjacent": "yellow", "out_of_scope": "red"}[bucket]
                tabs.append(t.paint(raw, "bold", color, "reverse"))
            else:
                tabs.append(t.dim(raw))
        summary_bits = [f"{rep.candidates} candidates"]
        if rep.spec.year_from or rep.spec.year_to:
            summary_bits.append(f"{rep.spec.year_from or 'any'}-{rep.spec.year_to or 'any'}")
        summary_bits.append(f"min {rep.spec.min_scope:.2f}")
        tab_line = " " + " ".join(tabs) + "  " + t.dim(f" {t.sep} ".join(summary_bits))
        out.append(tab_line)
        out.append("")

        avail = max(4, height - 2)
        findings = self.current_findings()
        if not findings:
            out.append("  " + t.yellow(f"No papers in {_BUCKET_LABELS[self.current_bucket]}."))
            out.append("  " + t.dim("Press Tab to view other buckets, f to refine scope, or / for a new search."))
            return out

        # Wide terminal & not expanded -> side-by-side split pane!
        if cols >= 96 and not self.detail_expanded:
            left_w = max(42, int(cols * 0.48))
            right_w = cols - left_w - 3
            left = self._render_list_pane(findings, left_w, avail)
            right = self._render_detail_pane(self.current_finding(), right_w, avail)
            sep = t.dim("│" if t.unicode else "|")
            for idx in range(avail):
                l_line = _pad_visible(left[idx] if idx < len(left) else "", left_w)
                r_line = right[idx] if idx < len(right) else ""
                out.append(f"{l_line} {sep} {r_line}")
            return out

        # Narrow terminal or Enter-expanded detail view
        if self.detail_expanded:
            detail = self._render_detail_pane(self.current_finding(), cols - 2, avail)
            out.extend(" " + line for line in detail)
            return out

        list_h = max(4, avail // 2)
        detail_h = max(3, avail - list_h - 1)
        out.extend(self._render_list_pane(findings, cols, list_h))
        out.append(ui.rule(cols, t))
        detail = self._render_detail_pane(self.current_finding(), cols - 2, detail_h)
        out.extend(" " + line for line in detail)
        return out

    def _render_list_pane(self, findings: list[Finding], width: int, height: int) -> list[str]:
        t = self.theme
        b = self.current_bucket
        sel = max(0, min(self.selected[b], len(findings) - 1))
        self.selected[b] = sel

        # Each item takes 2 rows (title + score/meta)
        per_page = max(1, height // 2)
        top = self.scroll[b]
        if sel < top:
            top = sel
        elif sel >= top + per_page:
            top = sel - per_page + 1
        self.scroll[b] = top

        lines: list[str] = []
        for idx in range(top, min(len(findings), top + per_page)):
            f = findings[idx]
            is_sel = idx == sel
            ptr = t.paint("▸" if t.unicode else ">", "bold", "cyan") if is_sel else " "
            title = ui.truncate(f.paper.title, max(16, width - 7))
            title_s = t.paint(title, "bold", "cyan") if is_sel else t.bold(title)
            lines.append(f" {ptr} {t.dim(f'{idx + 1:>2}.')} {title_s}")

            bar = t.bar(f.verdict.score, width=6)
            year = str(f.paper.year or "n.d.")
            cites = f"c:{f.paper.cited_by}" if f.paper.cited_by else ""
            pills: list[str] = []
            if f.access and f.access.open_access:
                pills.append(t.green("OA"))
            if f.access and f.access.scihub:
                pills.append(t.cyan("SH"))
            meta = f"       {bar} {f.verdict.score:.2f}  {t.dim(year)}"
            if cites:
                meta += f" {t.dim(cites)}"
            if pills:
                meta += "  " + " ".join(pills)
            lines.append(meta)
        return lines[:height]

    def _render_detail_pane(self, finding: Finding | None, width: int, height: int) -> list[str]:
        if finding is None:
            return []
        t = self.theme
        p = finding.paper
        v = finding.verdict
        out: list[str] = []

        for line in ui.wrap(p.title, width):
            out.append(t.bold(line))
        bits = [p.author_line(limit=3), str(p.year or "n.d.")]
        if p.cited_by:
            bits.append(f"cited {p.cited_by}")
        if p.journal:
            bits.append(p.journal)
        for line in ui.wrap(f" {t.sep} ".join(bits), width):
            out.append(t.dim(line))

        if p.doi:
            out.append(f"{t.dim('DOI')}  {p.doi}")
        elif p.arxiv_id:
            out.append(f"{t.dim('arXiv')}  {p.arxiv_id}")

        out.append("")
        out.append(
            f"{ui.badge(v.bucket, t)}  {t.bar(v.score, 8)} {v.score:.2f}  "
            f"{t.dim(f'coverage {v.coverage:.2f}')}"
        )
        if v.matched:
            out.extend(ui.wrap(", ".join(v.matched), width, initial=f"{t.green('matched')} ", subsequent="        "))
        if v.missing:
            out.extend(ui.wrap(", ".join(v.missing), width, initial=f"{t.yellow('missing')} ", subsequent="        "))
        reason = v.excluded_reason or v.reason
        if reason:
            out.extend(ui.wrap(reason, width, initial=f"{t.dim('why')} ", subsequent="    "))

        if finding.access:
            out.append("")
            for item in finding.access.open_access[:2]:
                out.append(f"{t.green('OA')}      {ui.truncate(item['url'], max(16, width - 8))}")
            for link in finding.access.scihub[:2]:
                st = link.get("state", "unchecked")
                out.append(
                    f"{t.cyan('Sci-Hub')} {ui.truncate(link['url'], max(16, width - 18))} {t.dim(f'[{st}]')}"
                )

        if p.abstract:
            out.append("")
            out.append(t.paint("ABSTRACT", "bold", "cyan"))
            remaining = max(2, height - len(out))
            wrapped = ui.wrap(p.abstract, width)
            if len(wrapped) > remaining:
                wrapped = wrapped[: remaining - 1] + [t.dim(f"{t.ellipsis} (press Enter/r for more)")]
            out.extend(wrapped)
        return out[:height]

    # --------------------------------------------------------------- main loop

    def _draw(self) -> None:
        size = shutil.get_terminal_size((100, 30))
        lines = self.render_lines(size.columns, size.lines)
        # Move home and paint every line with clear-to-EOL (zero flicker)
        frame = "\x1b[H" + "\n".join(line + "\x1b[K" for line in lines)
        self.stream.write(frame)
        self.stream.flush()

    def run(self) -> int:
        if not ui.vt_enabled(self.stream) or not getattr(sys.stdin, "isatty", lambda: False)():
            return 1

        # Save terminal state on POSIX
        old_termios = None
        if sys.platform != "win32":
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_termios = termios.tcgetattr(fd)
            tty.setraw(fd)

        try:
            # Enter alternate screen buffer, hide cursor, clear
            self.stream.write("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H")
            self.stream.flush()

            if self.intent and self.report is None:
                self.start_search(self.intent)

            reader = _read_key_windows if sys.platform == "win32" else _read_key_posix
            while self._running:
                self._draw()
                key = reader(0.12)
                if key:
                    if not self.handle_key(key):
                        break
        finally:
            if old_termios is not None:
                import termios

                termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, old_termios)
            self.stream.write("\x1b[?25h\x1b[?1049l")
            self.stream.flush()
        return 0


def run_tui(
    *,
    settings: Settings | None = None,
    report: Report | None = None,
    initial_intent: str = "",
    find_kwargs: dict[str, Any] | None = None,
) -> int:
    """Launch the interactive in-place terminal UI."""
    app = TuiApp(
        settings=settings,
        report=report,
        initial_intent=initial_intent,
        find_kwargs=find_kwargs,
    )
    return app.run()
