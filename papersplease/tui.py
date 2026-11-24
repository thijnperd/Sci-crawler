"""Interactive in-place terminal UI (zero dependencies, Windows + POSIX).

Uses the terminal alternate screen buffer (``\\x1b[?1049h`` / ``\\x1b[?1049l``)
and explicit cursor positioning (``\\x1b[{row};1H``) so the interface stays
pinned in the exact same place with zero scrollback spam or flicker. Every
screen is rendered by ``TuiApp.render_lines(cols, rows)`` as pure lines, keeping
all layout and interaction logic completely testable without a TTY.
"""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
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
    "shortlist": "Shortlist",
}
_SORT_MODES = ("score", "year", "citations", "oa")

_EXAMPLES = [
    "influence of bad diet on diabetes type 1",
    "optimal training for hypertrophy in younger adults since 2015",
    "muscle strength in older adults since 2015",
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
                    "S": "DEL",
                    "\x0f": "SHIFT_TAB",
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
            if ch == "\x15":
                return "CTRL_U"
            if ch == "\x17":
                return "CTRL_W"
            return ch
        time.sleep(0.015)
    return None


def _utf8_seq_len(first_byte: int) -> int:
    if first_byte < 0x80:
        return 1
    if (first_byte & 0xE0) == 0xC0:
        return 2
    if (first_byte & 0xF0) == 0xE0:
        return 3
    if (first_byte & 0xF8) == 0xF0:
        return 4
    return 1


def _read_key_posix(timeout: float = 0.12) -> str | None:
    import select

    fd = sys.stdin.fileno()
    ready, _, _ = select.select([fd], [], [], timeout)
    if not ready:
        return None
    raw = os.read(fd, 1)
    if not raw:
        return None

    first = raw[0]
    if first == 0x1B:
        buf = b""
        while len(buf) < 6:
            more, _, _ = select.select([fd], [], [], 0.02)
            if not more:
                break
            nxt = os.read(fd, 1)
            if not nxt:
                break
            buf += nxt
            if len(buf) >= 2 and (64 <= buf[-1] <= 126):
                break
        if not buf:
            return "ESC"
        seq = buf.decode("ascii", errors="ignore")
        return {
            "[A": "UP",
            "[B": "DOWN",
            "[C": "RIGHT",
            "[D": "LEFT",
            "[H": "HOME",
            "[F": "END",
            "[Z": "SHIFT_TAB",
            "[1~": "HOME",
            "[3~": "DEL",
            "[4~": "END",
            "[5~": "PGUP",
            "[6~": "PGDN",
            "OH": "HOME",
            "OF": "END",
        }.get(seq, "ESC")

    need = _utf8_seq_len(first)
    while len(raw) < need:
        more, _, _ = select.select([fd], [], [], 0.02)
        if not more:
            break
        nxt = os.read(fd, need - len(raw))
        if not nxt:
            break
        raw += nxt
    ch = raw.decode("utf-8", errors="ignore")
    if not ch:
        return None
    if ch in ("\r", "\n"):
        return "ENTER"
    if ch == "\t":
        return "TAB"
    if ch in ("\x08", "\x7f"):
        return "BACKSPACE"
    if ch == "\x03":
        return "CTRL_C"
    if ch == "\x15":
        return "CTRL_U"
    if ch == "\x17":
        return "CTRL_W"
    return ch


def _pad_visible(text: str, width: int, *, unicode: bool = True) -> str:
    if width <= 0:
        return ""
    vis = ui.visible_len(text)
    if vis > width:
        text = ui.truncate_ansi(text, width, unicode=unicode)
        vis = ui.visible_len(text)
    return text + " " * max(0, width - vis)


def _copy_to_clipboard(text: str, stream: TextIO, vt_ok: bool) -> bool:
    """Copy ``text`` via native clipboard CLI or OSC 52 fallback."""
    commands: list[list[str]] = []
    if sys.platform == "win32":
        commands.append(["clip"])
    elif sys.platform == "darwin":
        commands.append(["pbcopy"])
    else:
        commands.extend(
            [
                ["wl-copy"],
                ["xclip", "-selection", "clipboard"],
                ["xsel", "--clipboard", "--input"],
            ]
        )
    for cmd in commands:
        exe = shutil.which(cmd[0])
        if not exe:
            continue
        try:
            proc = subprocess.run(
                [exe, *cmd[1:]],
                input=text.encode("utf-8"),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.5,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            proc = None
        if proc is not None and proc.returncode == 0:
            return True
    if vt_ok:
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        stream.write(f"\x1b]52;c;{encoded}\x07")
        stream.flush()
        return True
    return False


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
        self.theme = ui.Theme.auto(self.stream)
        self.report: Report | None = report
        self.intent: str = (report.spec.intent if report else initial_intent).strip()
        self.find_kwargs: dict[str, Any] = dict(find_kwargs or {})

        self.bucket_index: int = 0
        self.shortlist_active: bool = False
        self.selected: dict[str, int] = {b: 0 for b in (*_BUCKETS, "shortlist")}
        self.scroll: dict[str, int] = {b: 0 for b in (*_BUCKETS, "shortlist")}
        self.focus_pane: str = "list"  # list | detail
        self.detail_expanded: bool = False
        self.detail_scroll: int = 0
        self.sort_mode: str = "score"
        self.filter_text: str = ""
        self.starred: set[str] = set()
        self.search_history: list[str] = [self.intent] if self.intent else []
        self.history_idx: int = -1

        self.overlay: str | None = None  # None | "explain" | "mirrors" | "providers" | "help"
        self.overlay_lines: list[str] = []
        self.overlay_scroll: int = 0
        self._last_overlay_cols: int = 0
        self._last_body_height: int = 20
        self._mirror_statuses: list[Any] = []

        # Inline prompt state: None | "search" | "refine" | "export" | "filter"
        self.prompt_mode: str | None = None
        self.prompt_buffer: str = ""
        self.status_message: str = ""
        self.status_kind: str = "info"  # info | ok | warn | error

        # Thread synchronization + redraw state
        self._state_lock = threading.RLock()
        self._busy_lock = threading.Lock()
        self._busy_panel: ui.LivePanel | None = None
        self._busy_label: str = ""
        self._running: bool = True
        self._dirty: bool = True

        if self.report is not None:
            self._pick_initial_bucket()

    def _pick_initial_bucket(self) -> None:
        if self.report is None:
            return
        for idx, bucket in enumerate(_BUCKETS):
            if self.report.bucket(bucket):
                self.bucket_index = idx
                self.shortlist_active = False
                return
        self.bucket_index = 0

    @property
    def current_bucket(self) -> str:
        return "shortlist" if self.shortlist_active else _BUCKETS[self.bucket_index]

    def _raw_bucket_findings(self) -> list[Finding]:
        if self.report is None:
            return []
        if self.shortlist_active:
            return [f for f in self.report.findings if f.paper.key() in self.starred]
        return self.report.bucket(_BUCKETS[self.bucket_index])

    def _apply_filter_and_sort(self, items: list[Finding]) -> list[Finding]:
        if self.filter_text.strip():
            terms = self.filter_text.strip().lower().split()
            filtered: list[Finding] = []
            for f in items:
                p = f.paper
                has_oa = bool(p.is_oa or (f.access and f.access.open_access))
                blob = " ".join(
                    [
                        p.title or "",
                        " ".join(p.authors),
                        p.journal or "",
                        p.abstract or "",
                        p.doi or "",
                        " ".join(f.verdict.matched),
                    ]
                ).lower()
                ok = True
                for term in terms:
                    if term == "oa":
                        if not has_oa:
                            ok = False
                            break
                    elif term == "doi":
                        if not p.doi:
                            ok = False
                            break
                    elif term not in blob:
                        ok = False
                        break
                if ok:
                    filtered.append(f)
            items = filtered

        if self.sort_mode == "year":
            return sorted(items, key=lambda f: (-(f.paper.year or 0), -f.verdict.score))
        if self.sort_mode == "citations":
            return sorted(items, key=lambda f: (-(f.paper.cited_by or 0), -f.verdict.score))
        if self.sort_mode == "oa":
            return sorted(
                items,
                key=lambda f: (
                    0 if (f.paper.is_oa or (f.access and f.access.open_access)) else 1,
                    -f.verdict.score,
                ),
            )
        return list(items)

    def current_findings(self) -> list[Finding]:
        with self._state_lock:
            return self._apply_filter_and_sort(self._raw_bucket_findings())

    def current_finding(self) -> Finding | None:
        with self._state_lock:
            items = self.current_findings()
            if not items:
                return None
            b = self.current_bucket
            idx = max(0, min(self.selected.get(b, 0), len(items) - 1))
            self.selected[b] = idx
            return items[idx]

    def set_status(self, message: str, kind: str = "info") -> None:
        with self._state_lock:
            self.status_message = message
            self.status_kind = kind
            self._dirty = True

    @property
    def is_busy(self) -> bool:
        with self._busy_lock:
            return self._busy_panel is not None

    # ------------------------------------------------------------------ actions

    def start_search(self, intent: str) -> None:
        intent = " ".join(intent.split())
        if not intent:
            return
        with self._state_lock:
            self.intent = intent
            if not self.search_history or self.search_history[-1] != intent:
                self.search_history.append(intent)
            self.history_idx = -1
            self.overlay = None
            self.prompt_mode = None
            self.detail_expanded = False
            self.detail_scroll = 0
            self.shortlist_active = False
            self._dirty = True

        panel = ui.LivePanel(
            "live pipeline",
            f'"{intent}"',
            stream=self.stream,
            clear_on_exit=False,
            show_brand=False,
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
                with self._state_lock:
                    self.report = rep
                    self.selected = {b: 0 for b in (*_BUCKETS, "shortlist")}
                    self.scroll = {b: 0 for b in (*_BUCKETS, "shortlist")}
                    self.detail_scroll = 0
                    self._pick_initial_bucket()
                self.set_status(
                    f"Loaded {len(rep.in_scope)} in scope, {len(rep.adjacent)} adjacent, "
                    f"{len(rep.out_of_scope)} rejected.",
                    "ok",
                )
            except Exception as exc:  # noqa: BLE001 - worker thread surfaces failure to status bar
                self.set_status(f"Search failed: {exc}", "error")
            finally:
                with self._busy_lock:
                    self._busy_panel = None
                    self._busy_label = ""
                self._dirty = True

        threading.Thread(target=worker, daemon=True).start()

    def _open_selected_link(self) -> None:
        finding = self.current_finding()
        if not finding:
            self.set_status("No paper selected.", "warn")
            return
        url = self._best_url(finding)
        if not url:
            self.set_status("No URL available for this paper.", "warn")
            return
        try:
            webbrowser.open(url)
            self.set_status(f"Opened {ui.truncate(url, 60, unicode=self.theme.unicode)}", "ok")
        except (OSError, webbrowser.Error) as exc:
            self.set_status(f"Could not open browser: {exc}", "error")

    def _best_url(self, finding: Finding) -> str | None:
        if finding.access:
            if finding.access.open_access:
                return finding.access.open_access[0].get("url")
            if finding.access.scihub:
                ok_links = [lnk for lnk in finding.access.scihub if lnk.get("state") == "ok"]
                return (ok_links or finding.access.scihub)[0].get("url")
            if finding.access.doi_url:
                return finding.access.doi_url
        if finding.paper.oa_url:
            return finding.paper.oa_url
        if finding.paper.doi:
            return f"https://doi.org/{finding.paper.doi}"
        return finding.paper.url

    def _copy_selected(self) -> None:
        finding = self.current_finding()
        if not finding:
            self.set_status("No paper selected.", "warn")
            return
        value = finding.paper.doi or self._best_url(finding) or finding.paper.title
        if not value:
            self.set_status("Nothing to copy for this paper.", "warn")
            return
        ok = _copy_to_clipboard(value, self.stream, ui.vt_enabled(self.stream))
        if ok:
            self.set_status(f"Copied: {ui.truncate(value, 56, unicode=self.theme.unicode)}", "ok")
        else:
            self.set_status("Clipboard unavailable on this terminal.", "warn")

    def _download_selected(self) -> None:
        finding = self.current_finding()
        if not finding:
            self.set_status("No paper selected.", "warn")
            return
        panel = ui.LivePanel(
            "download",
            ui.truncate(finding.paper.title, 72, unicode=self.theme.unicode),
            stages=[("access", "Resolve PDF"), ("save", "Save file")],
            stream=self.stream,
            clear_on_exit=False,
            show_brand=False,
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
            except Exception as exc:  # noqa: BLE001 - worker thread surfaces failure to status bar
                self.set_status(f"Download error: {exc}", "error")
            finally:
                with self._busy_lock:
                    self._busy_panel = None
                    self._busy_label = ""
                self._dirty = True

        threading.Thread(target=worker, daemon=True).start()

    def _deep_read_selected(self) -> None:
        finding = self.current_finding()
        if not finding:
            self.set_status("No paper selected.", "warn")
            return
        target = finding.paper.doi or finding.paper.arxiv_id or finding.paper.title
        panel = ui.LivePanel(
            "reading paper",
            ui.truncate(finding.paper.title, 72, unicode=self.theme.unicode),
            stages=[("read", "Fetch full text")],
            stream=self.stream,
            clear_on_exit=False,
            show_brand=False,
        )
        panel.theme = self.theme
        panel.set_stage("read", "running", target)
        with self._busy_lock:
            self._busy_panel = panel
            self._busy_label = "reading"

        def worker() -> None:
            try:
                refreshed = read_one(target, settings=self.settings, deep=True)
                if refreshed:
                    with self._state_lock:
                        if refreshed.abstract:
                            finding.paper.abstract = refreshed.abstract
                        if refreshed.excerpt:
                            finding.paper.excerpt = refreshed.excerpt
                        finding.paper.read_note = refreshed.read_note
                        finding.paper.fulltext_chars = refreshed.fulltext_chars
                        self.detail_expanded = True
                        self.detail_scroll = 0
                    self.set_status(f"Read updated ({refreshed.read_note or 'abstract loaded'})", "ok")
                else:
                    self.set_status("No additional text found for this paper.", "warn")
            except Exception as exc:  # noqa: BLE001 - worker thread surfaces failure to status bar
                self.set_status(f"Read failed: {exc}", "error")
            finally:
                with self._busy_lock:
                    self._busy_panel = None
                    self._busy_label = ""
                self._dirty = True

        threading.Thread(target=worker, daemon=True).start()

    def _render_overlay_content(self, kind: str, cols: int) -> list[str]:
        width = max(48, min(96, cols - 4))
        t = self.theme
        if kind == "explain":
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
            return render_explain(payload, width=width, color=t.enabled).splitlines()
        if kind == "providers":
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
            return render_providers(rows, width=width, color=t.enabled).splitlines()
        if kind == "mirrors":
            return render_mirrors(self._mirror_statuses, width=width, color=t.enabled).splitlines()
        return [
            t.paint("PapersPlease Interactive Controls", "bold", "cyan"),
            ui.rule(min(68, width), t),
            "",
            f"  {t.cyan('↑/↓  j/k')}        Move in list (or scroll detail when focused)",
            f"  {t.cyan('←/→  h/l')}        Switch focus between Results and Detail panes",
            f"  {t.cyan('J/K  PgUp/PgDn')}  Scroll Detail pane anytime {t.sep} Home/End jump",
            f"  {t.cyan('1/2/3  Tab')}      Switch In Scope / Adjacent / Rejected buckets",
            f"  {t.cyan('s / Space  4')}    Star/unstar paper {t.sep} view {t.star} Shortlist",
            f"  {t.cyan('Enter')}           Toggle full-screen Detail view",
            f"  {t.cyan('/')}               Search a new intent (↑/↓ in prompt for history)",
            f"  {t.cyan('f')}               Refine scope (min=0.6 since=2018 +req -excl | reset)",
            f"  {t.cyan(chr(92))}               Filter loaded results (text, 'oa', 'doi')",
            f"  {t.cyan('S')}               Cycle sort order (score / year / citations / oa)",
            f"  {t.cyan('e')}               Inspect scope contract & direction (offline explain)",
            f"  {t.cyan('o / c')}           Open best link in browser / copy DOI or URL",
            f"  {t.cyan('d / r')}           Download PDF to ./papers / deep-read full text",
            f"  {t.cyan('x')}               Export starred (or in-scope) to bibtex/ris/md/json",
            f"  {t.cyan('p / m')}           View providers / probe Sci-Hub mirrors (r refreshes)",
            f"  {t.cyan('Esc / q')}         Close overlay or prompt / quit",
        ]

    def _open_explain_overlay(self) -> None:
        if not self.intent:
            self.set_status("Search an intent first (press /).", "warn")
            return
        cols = shutil.get_terminal_size((96, 28)).columns
        with self._state_lock:
            self.overlay = "explain"
            self.overlay_lines = self._render_overlay_content("explain", cols)
            self._last_overlay_cols = cols
            self.overlay_scroll = 0
            self._dirty = True

    def _open_providers_overlay(self) -> None:
        cols = shutil.get_terminal_size((96, 28)).columns
        with self._state_lock:
            self.overlay = "providers"
            self.overlay_lines = self._render_overlay_content("providers", cols)
            self._last_overlay_cols = cols
            self.overlay_scroll = 0
            self._dirty = True

    def _open_mirrors_overlay(self, *, force: bool = False) -> None:
        panel = ui.LivePanel(
            "Sci-Hub mirrors",
            "Checking mirror reachability",
            stages=[("probe", "Probe mirrors")],
            stream=self.stream,
            clear_on_exit=False,
            show_brand=False,
        )
        panel.theme = self.theme
        panel.set_stage("probe", "running", f"{len(self.settings.mirrors)} configured mirrors")
        with self._busy_lock:
            self._busy_panel = panel
            self._busy_label = "probing mirrors"

        def worker() -> None:
            try:
                statuses = mirror_health(self.settings, force=force)
                cols = shutil.get_terminal_size((96, 28)).columns
                with self._state_lock:
                    self._mirror_statuses = list(statuses)
                    self.overlay = "mirrors"
                    self.overlay_lines = self._render_overlay_content("mirrors", cols)
                    self._last_overlay_cols = cols
                    self.overlay_scroll = 0
            except Exception as exc:  # noqa: BLE001 - worker thread surfaces failure to status bar
                self.set_status(f"Mirror check failed: {exc}", "error")
            finally:
                with self._busy_lock:
                    self._busy_panel = None
                    self._busy_label = ""
                self._dirty = True

        threading.Thread(target=worker, daemon=True).start()

    def _open_help_overlay(self) -> None:
        cols = shutil.get_terminal_size((96, 28)).columns
        with self._state_lock:
            self.overlay = "help"
            self.overlay_lines = self._render_overlay_content("help", cols)
            self._last_overlay_cols = cols
            self.overlay_scroll = 0
            self._dirty = True

    def _active_refine_summary(self) -> str:
        parts: list[str] = []
        if self.find_kwargs.get("min_scope") is not None:
            parts.append(f"min={self.find_kwargs['min_scope']}")
        if self.find_kwargs.get("since"):
            parts.append(f"since={self.find_kwargs['since']}")
        if self.find_kwargs.get("until"):
            parts.append(f"until={self.find_kwargs['until']}")
        for req in self.find_kwargs.get("require") or []:
            parts.append(f"+{req}")
        for exc in self.find_kwargs.get("exclude") or []:
            parts.append(f"-{exc}")
        return " ".join(parts)

    def _apply_refine(self, raw: str) -> None:
        tokens = raw.strip().split()
        if not tokens:
            return
        if "reset" in tokens:
            for key in ("min_scope", "since", "until", "require", "exclude", "expand"):
                self.find_kwargs.pop(key, None)
            if self.intent:
                self.start_search(self.intent)
            else:
                self.set_status("Refinements cleared.", "ok")
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
                term = tok[1:]
                if term not in require:
                    require.append(term)
            elif tok.startswith("-") and len(tok) > 1:
                term = tok[1:]
                if term not in exclude:
                    exclude.append(term)
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
        fmt = raw.strip().lower() or "bibtex"
        ext_map = {"bibtex": "bib", "ris": "ris", "csljson": "csl.json", "md": "md", "json": "json"}
        if fmt not in ext_map:
            self.set_status("Supported formats: bibtex, ris, csljson, md, json", "warn")
            return
        targets = (
            [f for f in self.report.findings if f.paper.key() in self.starred]
            if self.starred
            else list(self.report.in_scope)
        )
        out_path = Path(f"papersplease_export.{ext_map[fmt]}")
        if fmt == "json":
            if self.starred:
                payload = {
                    "schema_version": 1,
                    "intent": self.report.spec.intent,
                    "results": [f.to_dict(abstract_chars=1000) for f in targets],
                }
            else:
                payload = self.report.to_dict(abstract_chars=1000)
            content = json.dumps(payload, ensure_ascii=False, indent=2)
        elif fmt == "md":
            if self.starred:
                sub = Report(
                    spec=self.report.spec,
                    findings=targets,
                    candidates=self.report.candidates,
                    source_counts=self.report.source_counts,
                )
                content = render_markdown(sub)
            else:
                content = render_markdown(self.report)
        else:
            content = export.render(targets, fmt)
        out_path.write_text(content, encoding="utf-8")
        kind_label = "starred" if self.starred else "in-scope"
        self.set_status(f"Exported {len(targets)} {kind_label} papers to {out_path}", "ok")

    # ------------------------------------------------------------ key handling

    def handle_key(self, key: str) -> bool:
        """Process one keypress. Returns False when the app should exit."""
        if key == "CTRL_C":
            self._running = False
            return False

        self._dirty = True

        # While typing in the bottom prompt bar
        if self.prompt_mode is not None:
            if key == "ESC":
                self.prompt_mode = None
                self.prompt_buffer = ""
                self.history_idx = -1
                return True
            if key == "ENTER":
                mode = self.prompt_mode
                val = self.prompt_buffer
                self.prompt_mode = None
                self.prompt_buffer = ""
                self.history_idx = -1
                if mode == "search":
                    self.start_search(val)
                elif mode == "refine":
                    self._apply_refine(val)
                elif mode == "export":
                    self._apply_export(val)
                elif mode == "filter":
                    with self._state_lock:
                        self.filter_text = val.strip()
                        for b in (*_BUCKETS, "shortlist"):
                            self.selected[b] = 0
                            self.scroll[b] = 0
                        self.detail_scroll = 0
                    self.set_status(
                        f"Filter: '{self.filter_text}'" if self.filter_text else "Filter cleared.",
                        "info",
                    )
                return True
            if key == "BACKSPACE":
                self.prompt_buffer = self.prompt_buffer[:-1]
                return True
            if key == "CTRL_U":
                self.prompt_buffer = ""
                return True
            if key == "CTRL_W":
                stripped = self.prompt_buffer.rstrip()
                idx = stripped.rfind(" ")
                self.prompt_buffer = stripped[: idx + 1] if idx >= 0 else ""
                return True
            if self.prompt_mode == "search" and self.search_history:
                if key == "UP":
                    if self.history_idx == -1:
                        self.history_idx = len(self.search_history) - 1
                    elif self.history_idx > 0:
                        self.history_idx -= 1
                    self.prompt_buffer = self.search_history[self.history_idx]
                    return True
                if key == "DOWN":
                    if self.history_idx != -1:
                        if self.history_idx < len(self.search_history) - 1:
                            self.history_idx += 1
                            self.prompt_buffer = self.search_history[self.history_idx]
                        else:
                            self.history_idx = -1
                            self.prompt_buffer = ""
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
            if key == "r" and self.overlay == "mirrors":
                self._open_mirrors_overlay(force=True)
                return True
            max_scroll = max(0, len(self.overlay_lines) - max(1, self._last_body_height - 2))
            if key in ("DOWN", "j"):
                self.overlay_scroll = min(max_scroll, self.overlay_scroll + 1)
            elif key in ("UP", "k"):
                self.overlay_scroll = max(0, self.overlay_scroll - 1)
            elif key == "PGDN":
                self.overlay_scroll = min(max_scroll, self.overlay_scroll + 8)
            elif key == "PGUP":
                self.overlay_scroll = max(0, self.overlay_scroll - 8)
            elif key == "HOME":
                self.overlay_scroll = 0
            elif key == "END":
                self.overlay_scroll = max_scroll
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
            self.history_idx = -1
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
        if key == "\\":
            self.prompt_mode = "filter"
            self.prompt_buffer = self.filter_text
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

        # Report navigation & pane controls
        if key in ("1", "2", "3"):
            self.shortlist_active = False
            self.bucket_index = int(key) - 1
            self.detail_expanded = False
            self.detail_scroll = 0
            return True
        if key == "4":
            self.shortlist_active = not self.shortlist_active
            self.detail_expanded = False
            self.detail_scroll = 0
            return True
        if key == "TAB":
            self.shortlist_active = False
            self.bucket_index = (self.bucket_index + 1) % len(_BUCKETS)
            self.detail_expanded = False
            self.detail_scroll = 0
            return True
        if key == "SHIFT_TAB":
            self.shortlist_active = False
            self.bucket_index = (self.bucket_index - 1) % len(_BUCKETS)
            self.detail_expanded = False
            self.detail_scroll = 0
            return True
        if key in ("LEFT", "h"):
            self.focus_pane = "list"
            return True
        if key in ("RIGHT", "l"):
            self.focus_pane = "detail"
            return True

        # Dedicated detail scroll keys work from anywhere
        if key in ("J", "PGDN"):
            step = 1 if key == "J" else 6
            self.detail_scroll += step
            return True
        if key in ("K", "PGUP"):
            step = 1 if key == "K" else 6
            self.detail_scroll = max(0, self.detail_scroll - step)
            return True

        # Up/Down moves list or scrolls detail depending on focus/expand state
        if key in ("DOWN", "j"):
            if self.detail_expanded or self.focus_pane == "detail":
                self.detail_scroll += 1
                return True
            items = self.current_findings()
            if items:
                b = self.current_bucket
                nxt = min(len(items) - 1, self.selected[b] + 1)
                if nxt != self.selected[b]:
                    self.selected[b] = nxt
                    self.detail_scroll = 0
            return True
        if key in ("UP", "k"):
            if self.detail_expanded or self.focus_pane == "detail":
                self.detail_scroll = max(0, self.detail_scroll - 1)
                return True
            b = self.current_bucket
            prv = max(0, self.selected[b] - 1)
            if prv != self.selected[b]:
                self.selected[b] = prv
                self.detail_scroll = 0
            return True
        if key == "HOME":
            if self.detail_expanded or self.focus_pane == "detail":
                self.detail_scroll = 0
            else:
                self.selected[self.current_bucket] = 0
                self.detail_scroll = 0
            return True
        if key == "END":
            if self.detail_expanded or self.focus_pane == "detail":
                self.detail_scroll += 999
            else:
                items = self.current_findings()
                if items:
                    self.selected[self.current_bucket] = len(items) - 1
                    self.detail_scroll = 0
            return True

        if key == "ENTER":
            self.detail_expanded = not self.detail_expanded
            self.detail_scroll = 0
            return True
        if key == "ESC":
            if self.detail_expanded:
                self.detail_expanded = False
                self.detail_scroll = 0
            elif self.filter_text:
                self.filter_text = ""
                self.set_status("Filter cleared.", "info")
            return True
        if key in ("s", " "):
            finding = self.current_finding()
            if finding:
                k = finding.paper.key()
                if k in self.starred:
                    self.starred.remove(k)
                    self.set_status(f"Removed from shortlist ({len(self.starred)} starred).", "info")
                else:
                    self.starred.add(k)
                    self.set_status(f"Starred ({len(self.starred)} in shortlist).", "ok")
            return True
        if key == "S":
            idx = _SORT_MODES.index(self.sort_mode)
            self.sort_mode = _SORT_MODES[(idx + 1) % len(_SORT_MODES)]
            self.selected[self.current_bucket] = 0
            self.scroll[self.current_bucket] = 0
            self.detail_scroll = 0
            self.set_status(f"Sorted by: {self.sort_mode}", "info")
            return True
        if key == "c":
            self._copy_selected()
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
        cols = max(40, cols)
        rows = max(12, rows)
        t = self.theme
        lines: list[str] = []

        # 1. Top Header Bar (2 lines)
        brand = t.paint(f" PapersPlease {__version__} ", "bold", "cyan")
        right_parts: list[str] = []
        if self.intent:
            right_parts.append(f'"{self.intent}"')
        else:
            right_parts.append("in-place literature screener")
        with self._state_lock:
            rep = self.report
            if rep and rep.cache_stats:
                hits = int(rep.cache_stats.get("verdict_hits", 0))
                if hits:
                    right_parts.append(f"cached {hits}")
        right_plain = f" {t.sep} ".join(right_parts)
        avail_right = max(10, cols - ui.visible_len(brand) - 2)
        right = t.dim(ui.truncate(right_plain, avail_right, unicode=t.unicode))
        gap = max(1, cols - ui.visible_len(brand) - ui.visible_len(right) - 1)
        lines.append(_pad_visible(brand + " " * gap + right, cols, unicode=t.unicode))
        lines.append(ui.rule(cols, t))

        # Reserve bottom lines (prompt/status + 1-or-2-line keybind bar)
        keybar_lines = [] if self.prompt_mode is not None else self._keybar_lines(cols)
        footer_h = 2 if self.prompt_mode is not None else 1 + len(keybar_lines)
        body_height = max(4, rows - 2 - footer_h)
        self._last_body_height = body_height

        with self._busy_lock:
            busy_panel = self._busy_panel

        if busy_panel is not None:
            panel_w = max(36, min(cols - 2, 92))
            panel_lines = busy_panel.render_snapshot(width=panel_w)
            pad_top = max(0, (body_height - len(panel_lines)) // 2)
            pad_left = " " * max(0, (cols - panel_w) // 2)
            body: list[str] = [""] * pad_top
            for pl in panel_lines:
                body.append(pad_left + pl)
        elif self.overlay is not None:
            if cols != self._last_overlay_cols:
                self.overlay_lines = self._render_overlay_content(self.overlay, cols)
                self._last_overlay_cols = cols
            body = self._render_overlay(cols, body_height)
        elif rep is None:
            body = self._render_welcome(cols, body_height)
        else:
            with self._state_lock:
                body = self._render_report(cols, body_height)

        body = body[:body_height]
        while len(body) < body_height:
            body.append("")
        lines.extend(_pad_visible(line, cols, unicode=t.unicode) for line in body)

        # 3. Bottom Prompt / Status Bar + wrapped Keybar
        if self.prompt_mode is not None:
            prompt_labels = {
                "search": "Search intent",
                "refine": "Refine (min=0.6 since=2018 +req -excl | reset)",
                "export": "Export format (bibtex|ris|csljson|md|json)",
                "filter": "Filter (text, 'oa', 'doi'; empty clears)",
            }
            lbl = t.paint(f" {prompt_labels[self.prompt_mode]}: ", "bold", "cyan")
            cursor = t.paint(" ", "reverse") if t.enabled else "_"
            avail_buf = max(8, cols - ui.visible_len(lbl) - 2)
            shown_buf = (
                self.prompt_buffer[-avail_buf:]
                if len(self.prompt_buffer) > avail_buf
                else self.prompt_buffer
            )
            lines.append(_pad_visible(lbl + shown_buf + cursor, cols, unicode=t.unicode))

            hint_bits = ["Enter submit", "Esc cancel"]
            if self.prompt_mode == "search" and self.search_history:
                hint_bits.append("↑↓ history" if t.unicode else "^v history")
            elif self.prompt_mode == "refine":
                active = self._active_refine_summary()
                if active:
                    hint_bits.append(f"active: {active}")
            lines.append(
                _pad_visible(t.dim("  " + f" {t.sep} ".join(hint_bits)), cols, unicode=t.unicode)
            )
        else:
            if self.status_message:
                painter = {
                    "ok": t.green,
                    "warn": t.yellow,
                    "error": t.red,
                }.get(self.status_kind, t.dim)
                lines.append(
                    _pad_visible(
                        " " + painter(ui.truncate(self.status_message, max(12, cols - 2), unicode=t.unicode)),
                        cols,
                        unicode=t.unicode,
                    )
                )
            else:
                lines.append(ui.rule(cols, t))
            for kline in keybar_lines:
                lines.append(_pad_visible(kline, cols, unicode=t.unicode))

        return lines[:rows]

    def _keybar_items(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        t = self.theme
        arr_ud = "↑↓" if t.unicode else "^v"
        arr_lr = "←→" if t.unicode else "<>"
        if self.overlay is not None:
            head = [(arr_ud, "scroll"), ("PgUp/Dn", "page")]
            if self.overlay == "mirrors":
                head.insert(0, ("r", "refresh"))
            return head, [("Esc/q", "close")]
        if self.report is None:
            return [
                ("/", "search"),
                ("1-3", "examples"),
                ("p", "providers"),
                ("m", "mirrors"),
            ], [("?", "help"), ("q", "quit")]
        return [
            (arr_ud, "move"),
            (arr_lr, "focus"),
            ("J/K", "scroll"),
            ("Tab", "bucket"),
            ("s", "star"),
            ("/", "search"),
            ("\\", "filter"),
            ("S", "sort"),
            ("f", "refine"),
            ("e", "explain"),
            ("o", "open"),
            ("c", "copy"),
            ("d", "pdf"),
            ("x", "export"),
        ], [("?", "help"), ("q", "quit")]

    def _keybar_lines(self, cols: int, *, max_lines: int = 2) -> list[str]:
        t = self.theme
        head, tail = self._keybar_items()
        head_chunks = [f"{t.cyan(k)} {t.dim(v)}" for k, v in head]
        tail_chunks = [f"{t.cyan(k)} {t.dim(v)}" for k, v in tail]
        rows: list[list[str]] = [[]]
        for chunk in head_chunks:
            cand = " " + "  ".join([*rows[-1], chunk])
            if ui.visible_len(cand) <= cols - 1 or not rows[-1]:
                rows[-1].append(chunk)
            elif len(rows) < max_lines:
                rows.append([chunk])
            else:
                break

        tail_joined = "  ".join(tail_chunks)
        if tail_joined:
            while rows[-1]:
                cand = " " + "  ".join([*rows[-1], tail_joined])
                if ui.visible_len(cand) <= cols - 1:
                    rows[-1].extend(tail_chunks)
                    break
                rows[-1].pop()
            else:
                if len(rows) < max_lines:
                    rows.append(list(tail_chunks))
                else:
                    rows[-1] = list(tail_chunks)
        return [" " + "  ".join(r) for r in rows if r] or [" " + tail_joined]

    def _keybar(self, cols: int) -> str:
        return self._keybar_lines(cols, max_lines=1)[0]

    def _render_welcome(self, cols: int, height: int) -> list[str]:
        t = self.theme
        box_w = min(cols - 2, 78)
        inner = max(24, box_w - 4)
        tl, tr, bl, br, h, v = t.box_chars

        def box_row(content: str) -> str:
            content = _pad_visible(content, inner, unicode=t.unicode)
            return f"{t.dim(v)} {content} {t.dim(v)}"

        footer = ui.truncate(
            f"Zero dependencies  {t.sep}  Europe PMC, Crossref, arXiv, OpenAlex, S2",
            inner,
            unicode=t.unicode,
        )
        card = [
            t.dim(tl + h * (box_w - 2) + tr),
            box_row(t.paint(ui.truncate("Search, read, and screen the literature in one place.", inner, unicode=t.unicode), "bold", "cyan")),
            box_row(t.dim(ui.truncate("Press / or Enter to type a research intent, or pick an example:", inner, unicode=t.unicode))),
            box_row(""),
        ]
        for idx, example in enumerate(_EXAMPLES, start=1):
            card.append(box_row(f"  {t.cyan(f'[{idx}]')} {ui.truncate(example, max(10, inner - 6), unicode=t.unicode)}"))
        card.append(box_row(""))
        card.append(box_row(t.dim(footer)))
        card.append(t.dim(bl + h * (box_w - 2) + br))

        pad_top = max(0, (height - len(card)) // 2)
        pad_left = " " * max(0, (cols - box_w) // 2)
        return [""] * pad_top + [pad_left + line for line in card]

    def _render_overlay(self, cols: int, height: int) -> list[str]:
        t = self.theme
        title = f" {self.overlay.upper()} " if self.overlay else ""
        total = len(self.overlay_lines)
        view_h = max(1, height - 2)
        max_scroll = max(0, total - view_h)
        self.overlay_scroll = max(0, min(self.overlay_scroll, max_scroll))
        pos_info = f" ({self.overlay_scroll + 1}-{min(total, self.overlay_scroll + view_h)}/{total})" if total > view_h else ""
        header = t.paint(title, "bold", "cyan") + t.dim(f"  Esc to close{pos_info}")
        visible = self.overlay_lines[self.overlay_scroll : self.overlay_scroll + view_h]
        return [header, ""] + ["  " + line for line in visible]

    def _frame_box(
        self,
        title: str,
        body_lines: list[str],
        width: int,
        height: int,
        *,
        active: bool = False,
        total_items: int = 0,
        view_offset: int = 0,
    ) -> list[str]:
        """Wrap ``body_lines`` inside a rounded/ASCII panel with optional scrollbar."""
        t = self.theme
        width = max(16, width)
        height = max(3, height)
        inner_w = max(8, width - 4)
        inner_h = max(1, height - 2)

        tl, tr, bl, br, h, v = t.box_chars
        border_paint = t.cyan if active else t.dim
        title_paint = (lambda s: t.paint(s, "bold", "cyan")) if active else t.bold

        title_clean = ui.truncate(title, max(4, width - 6), unicode=t.unicode)
        title_seg = f" {title_paint(title_clean)} "
        fill_top = max(1, width - 3 - ui.visible_len(title_seg))
        top = border_paint(tl + h) + title_seg + border_paint(h * fill_top + tr)

        has_scroll = total_items > inner_h
        thumb_start = 0
        thumb_size = inner_h
        if has_scroll and total_items > 0:
            thumb_size = max(1, round(inner_h * inner_h / total_items))
            max_off = max(1, total_items - inner_h)
            thumb_start = round((view_offset / max_off) * max(0, inner_h - thumb_size))

        out = [top]
        thumb_glyph = t.scroll_thumb
        for row in range(inner_h):
            content = body_lines[row] if row < len(body_lines) else ""
            padded = _pad_visible(content, inner_w, unicode=t.unicode)
            if has_scroll:
                is_thumb = thumb_start <= row < thumb_start + thumb_size
                bar_char = (
                    (t.cyan(thumb_glyph) if active else t.dim(thumb_glyph))
                    if is_thumb
                    else border_paint(v)
                )
                out.append(f"{border_paint(v)} {padded} {bar_char}")
            else:
                out.append(f"{border_paint(v)} {padded} {border_paint(v)}")

        out.append(border_paint(bl + h * (width - 2) + br))
        return out

    def _render_report(self, cols: int, height: int) -> list[str]:
        assert self.report is not None
        t = self.theme
        rep = self.report
        out: list[str] = []

        # Tab bar + active mode pills
        tabs: list[str] = []
        for idx, bucket in enumerate(_BUCKETS, start=1):
            count = len(rep.bucket(bucket))
            glyph = {"in_scope": t.dot, "adjacent": t.half, "out_of_scope": t.ring}[bucket]
            raw = f" [{idx}] {glyph} {_BUCKET_LABELS[bucket]} ({count}) "
            if not self.shortlist_active and idx - 1 == self.bucket_index:
                color = {"in_scope": "green", "adjacent": "yellow", "out_of_scope": "red"}[bucket]
                tabs.append(t.paint(raw, "bold", color, "reverse"))
            else:
                tabs.append(t.dim(raw))

        if self.starred or self.shortlist_active:
            sraw = f" [4] {t.star} Shortlist ({len(self.starred)}) "
            if self.shortlist_active:
                tabs.append(t.paint(sraw, "bold", "magenta", "reverse"))
            else:
                tabs.append(t.magenta(sraw))

        pills: list[str] = [f"{rep.candidates} cands"]
        if rep.spec.year_from or rep.spec.year_to:
            pills.append(f"{rep.spec.year_from or 'any'}-{rep.spec.year_to or 'any'}")
        if self.sort_mode != "score":
            pills.append(f"sort:{self.sort_mode}")
        if self.filter_text:
            pills.append(f"filter:{ui.truncate(self.filter_text, 14, unicode=t.unicode)}")
        tab_line = " " + " ".join(tabs) + "  " + t.dim(f" {t.sep} ".join(pills))
        out.append(tab_line)

        avail = max(4, height - 1)
        findings = self._apply_filter_and_sort(self._raw_bucket_findings())
        b = self.current_bucket
        sel = 0
        if findings:
            sel = max(0, min(self.selected.get(b, 0), len(findings) - 1))
            self.selected[b] = sel

        current = findings[sel] if findings else None

        if not findings:
            empty_lines = [
                "",
                t.yellow(
                    f"No papers in {_BUCKET_LABELS.get(b, b)}"
                    + (f" matching '{self.filter_text}'" if self.filter_text else "")
                    + "."
                ),
                t.dim("Press Tab to switch buckets, f to refine scope, \\ to clear filter, or / for a new search."),
            ]
            return out + empty_lines

        # Full-screen detail mode (Enter)
        if self.detail_expanded:
            inner_h = max(2, avail - 2)
            inner_w = max(20, cols - 4)
            all_detail = self._render_detail_lines(current, inner_w)
            max_scroll = max(0, len(all_detail) - inner_h)
            self.detail_scroll = max(0, min(self.detail_scroll, max_scroll))
            shown = all_detail[self.detail_scroll : self.detail_scroll + inner_h]
            panel = self._frame_box(
                f"Detail ({sel + 1}/{len(findings)})  {t.sep}  ↑↓ scroll  {t.sep}  Enter back",
                shown,
                cols,
                avail,
                active=True,
                total_items=len(all_detail),
                view_offset=self.detail_scroll,
            )
            return out + panel

        # Wide terminal -> side-by-side framed split pane
        if cols >= 96:
            left_w = max(40, int(cols * 0.46))
            right_w = cols - left_w - 1
            inner_h = max(2, avail - 2)

            list_lines, list_total_rows, list_row_offset = self._render_list_lines(
                findings, max(16, left_w - 4), inner_h
            )
            left_panel = self._frame_box(
                f"Results ({sel + 1}/{len(findings)})",
                list_lines,
                left_w,
                avail,
                active=(self.focus_pane == "list"),
                total_items=list_total_rows,
                view_offset=list_row_offset,
            )

            detail_lines_all = self._render_detail_lines(current, max(20, right_w - 4))
            max_d_scroll = max(0, len(detail_lines_all) - inner_h)
            self.detail_scroll = max(0, min(self.detail_scroll, max_d_scroll))
            detail_shown = detail_lines_all[self.detail_scroll : self.detail_scroll + inner_h]
            right_title = "Paper Detail"
            if len(detail_lines_all) > inner_h:
                right_title += f"  (J/K scroll {self.detail_scroll + 1}/{len(detail_lines_all)})"
            right_panel = self._frame_box(
                right_title,
                detail_shown,
                right_w,
                avail,
                active=(self.focus_pane == "detail"),
                total_items=len(detail_lines_all),
                view_offset=self.detail_scroll,
            )
            for idx in range(avail):
                out.append(f"{left_panel[idx]} {right_panel[idx]}")
            return out

        # Narrow terminal -> stacked framed panels
        list_h = max(4, avail // 2)
        detail_h = max(4, avail - list_h)
        list_inner_h = max(2, list_h - 2)
        det_inner_h = max(2, detail_h - 2)

        list_lines, list_total_rows, list_row_offset = self._render_list_lines(
            findings, max(16, cols - 4), list_inner_h
        )
        top_panel = self._frame_box(
            f"Results ({sel + 1}/{len(findings)})",
            list_lines,
            cols,
            list_h,
            active=(self.focus_pane == "list"),
            total_items=list_total_rows,
            view_offset=list_row_offset,
        )
        detail_lines_all = self._render_detail_lines(current, max(16, cols - 4))
        max_d_scroll = max(0, len(detail_lines_all) - det_inner_h)
        self.detail_scroll = max(0, min(self.detail_scroll, max_d_scroll))
        detail_shown = detail_lines_all[self.detail_scroll : self.detail_scroll + det_inner_h]
        bot_panel = self._frame_box(
            "Paper Detail (J/K scroll)",
            detail_shown,
            cols,
            detail_h,
            active=(self.focus_pane == "detail"),
            total_items=len(detail_lines_all),
            view_offset=self.detail_scroll,
        )
        return out + top_panel + bot_panel

    def _render_list_lines(
        self, findings: list[Finding], width: int, inner_h: int
    ) -> tuple[list[str], int, int]:
        t = self.theme
        b = self.current_bucket
        sel = max(0, min(self.selected.get(b, 0), len(findings) - 1))
        self.selected[b] = sel

        per_page = max(1, inner_h // 2)
        top = self.scroll.get(b, 0)
        if sel < top:
            top = sel
        elif sel >= top + per_page:
            top = sel - per_page + 1
        self.scroll[b] = top

        lines: list[str] = []
        star_on = t.star
        for idx in range(top, min(len(findings), top + per_page)):
            f = findings[idx]
            is_sel = idx == sel
            is_star = f.paper.key() in self.starred
            ptr = t.paint(t.pointer, "bold", "cyan") if is_sel else " "
            star = t.magenta(star_on) if is_star else " "
            title = ui.truncate(f.paper.title, max(12, width - 8), unicode=t.unicode)
            title_s = t.paint(title, "bold", "cyan") if is_sel else t.bold(title)
            lines.append(f"{ptr}{star} {t.dim(f'{idx + 1:>2}.')} {title_s}")

            bar = t.bar(f.verdict.score, width=6)
            year = str(f.paper.year or "n.d.")
            cites = f"c:{f.paper.cited_by}" if f.paper.cited_by else ""
            pills: list[str] = []
            if f.paper.is_oa or (f.access and f.access.open_access):
                pills.append(t.green("OA"))
            if f.access and f.access.scihub:
                pills.append(t.cyan("SH"))
            if f.verdict.direction == "aligned":
                pills.append(t.green(t.arrow))
            elif f.verdict.direction == "descriptive":
                pills.append(t.yellow("~"))
            elif f.verdict.direction == "reversed":
                pills.append(t.red("←" if t.unicode else "<-"))
            meta = f"      {bar} {f.verdict.score:.2f} {t.dim(year)}"
            if cites:
                meta += f" {t.dim(cites)}"
            if pills:
                meta += " " + " ".join(pills)
            lines.append(meta)
        return lines, len(findings) * 2, top * 2

    def _render_detail_lines(self, finding: Finding | None, width: int) -> list[str]:
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

        # Badges + Identifier row
        id_parts: list[str] = []
        if p.doi:
            doi_url = f"https://doi.org/{p.doi}"
            id_parts.append(f"{t.dim('DOI')} {ui.osc8_link(doi_url, p.doi, t.enabled)}")
        elif p.arxiv_id:
            ax_url = f"https://arxiv.org/abs/{p.arxiv_id}"
            id_parts.append(f"{t.dim('arXiv')} {ui.osc8_link(ax_url, p.arxiv_id, t.enabled)}")
        if p.sources:
            id_parts.append(t.dim("[" + ",".join(p.sources) + "]"))
        if p.key() in self.starred:
            id_parts.append(t.magenta(f"{t.star} starred"))
        if (Path("papers") / filename_for(p)).exists():
            id_parts.append(t.green("PDF saved"))
        if id_parts:
            out.append(f"  {t.sep}  ".join(id_parts))

        out.append("")
        score_row = (
            f"{ui.badge(v.bucket, t)}  {t.bar(v.score, 8)} {v.score:.2f}  "
            f"{t.dim(f'cov {v.coverage:.2f} {t.sep} title {v.title_coverage:.2f}')}"
        )
        out.append(score_row)
        if v.direction == "aligned":
            out.append(t.green(f"{t.check} direction aligned"))
        elif v.direction == "descriptive":
            out.append(t.yellow(f"{t.half} descriptive only (not causal X -> Y)"))
        elif v.direction == "reversed":
            out.append(t.red(f"{t.cross} reversed causal direction"))

        if v.matched:
            out.extend(
                ui.wrap(", ".join(v.matched), width, initial=f"{t.green('matched')} ", subsequent="        ")
            )
        if v.missing:
            out.extend(
                ui.wrap(", ".join(v.missing), width, initial=f"{t.yellow('missing')} ", subsequent="        ")
            )
        reason = v.excluded_reason or v.reason
        if reason:
            out.extend(ui.wrap(reason, width, initial=f"{t.dim('why')} ", subsequent="    "))
        if v.llm:
            llm_msg = f"{v.llm.get('verdict', '')} ({v.llm.get('specificity', '')}): {v.llm.get('reason', '')}"
            out.extend(ui.wrap(llm_msg, width, initial=f"{t.cyan('llm')} ", subsequent="    "))

        if v.matched_snippet and v.matched_snippet.lower() != (p.title or "").strip().lower():
            out.append("")
            out.append(t.dim("BEST MATCH SENTENCE"))
            quote_prefix = t.dim("│ " if t.unicode else "| ")
            out.extend(ui.wrap(v.matched_snippet, width, initial=quote_prefix, subsequent=quote_prefix))

        if finding.access:
            out.append("")
            for item in finding.access.open_access[:2]:
                url = item["url"]
                label = ui.truncate(url, max(16, width - 8), unicode=t.unicode)
                out.append(f"{t.green('OA')}      {ui.osc8_link(url, label, t.enabled)}")
            for link in finding.access.scihub[:2]:
                url = link["url"]
                st = link.get("state", "unchecked")
                label = ui.truncate(url, max(16, width - 18), unicode=t.unicode)
                out.append(f"{t.cyan('Sci-Hub')} {ui.osc8_link(url, label, t.enabled)} {t.dim(f'[{st}]')}")

        if p.abstract:
            out.append("")
            out.append(t.paint("ABSTRACT", "bold", "cyan"))
            out.extend(ui.wrap(p.abstract, width))

        if p.excerpt:
            out.append("")
            note = f" ({p.read_note})" if p.read_note else ""
            out.append(t.paint(f"FULL-TEXT EXCERPT{note}", "bold", "cyan"))
            out.extend(ui.wrap(p.excerpt, width))
        return out

    # --------------------------------------------------------------- main loop

    def _draw(self) -> None:
        size = shutil.get_terminal_size((100, 30))
        lines = self.render_lines(size.columns, size.lines)
        if lines == getattr(self, "_last_drawn_lines", None):
            return
        self._last_drawn_lines = list(lines)
        frame = (
            "\x1b[?2026h"
            + "".join(f"\x1b[{row};1H{line}\x1b[K" for row, line in enumerate(lines, start=1))
            + "\x1b[?2026l"
        )
        self.stream.write(frame)
        self.stream.flush()

    def run(self) -> int:
        if not ui.vt_enabled(self.stream) or not getattr(sys.stdin, "isatty", lambda: False)():
            return 1

        old_termios = None
        if sys.platform != "win32":
            import termios
            import tty

            fd = sys.stdin.fileno()
            old_termios = termios.tcgetattr(fd)
            tty.setraw(fd)

        try:
            self.stream.write("\x1b[?1049h\x1b[?25l\x1b[2J\x1b[H")
            self.stream.flush()

            if self.intent and self.report is None:
                self.start_search(self.intent)

            reader = _read_key_windows if sys.platform == "win32" else _read_key_posix
            last_size = shutil.get_terminal_size((100, 30))
            last_busy_draw = 0.0
            self._last_drawn_lines = None
            self._dirty = True
            while self._running:
                cur_size = shutil.get_terminal_size((100, 30))
                now = time.monotonic()
                if self._dirty or cur_size != last_size or (self.is_busy and now - last_busy_draw >= 0.25):
                    self._draw()
                    last_size = cur_size
                    last_busy_draw = now
                    self._dirty = False
                key = reader(0.12)
                if key and not self.handle_key(key):
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
