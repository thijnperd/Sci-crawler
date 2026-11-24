"""Tests for the interactive in-place TUI and LivePanel."""

import io
import unittest
from unittest.mock import patch

from papersplease import cli, ui
from papersplease.intent import parse_intent
from papersplease.models import Paper, Verdict
from papersplease.pipeline import Finding, Report
from papersplease.resolver import Access
from papersplease.tui import TuiApp


def _sample_report() -> Report:
    spec = parse_intent("optimal training for hypertrophy in younger adults since 2015")
    p1 = Paper(
        title="Resistance training volume and muscle hypertrophy in young adults",
        doi="10.1000/hypertrophy1",
        year=2021,
        authors=["Schoenfeld Brad", "Grgic Jozo"],
        journal="Sports Medicine",
        cited_by=85,
        abstract="We examined weekly set volume for muscle hypertrophy in younger adults.",
    )
    v1 = Verdict(
        doi=p1.doi,
        title=p1.title,
        score=0.92,
        coverage=1.0,
        title_coverage=1.0,
        matched=["young adults", "exercise", "hypertrophy"],
        bucket="in_scope",
        in_scope=True,
        reason="matched all facets",
    )
    a1 = Access(
        doi=p1.doi,
        doi_url=f"https://doi.org/{p1.doi}",
        open_access=[{"url": "https://example.org/paper1.pdf", "kind": "oa", "license": "CC BY"}],
        scihub=[{"mirror": "https://sci-hub.ru", "url": "https://sci-hub.ru/10.1000/hypertrophy1", "state": "ok"}],
    )
    p2 = Paper(
        title="Training in older adults",
        doi="10.1000/older1",
        year=2020,
        authors=["Doe Jane"],
        journal="Ageing Research",
    )
    v2 = Verdict(
        doi=p2.doi,
        title=p2.title,
        score=0.58,
        coverage=0.60,
        title_coverage=0.5,
        matched=["exercise"],
        missing=["young adults"],
        bucket="adjacent",
        in_scope=False,
        reason="wrong population",
    )
    return Report(
        spec=spec,
        findings=[Finding(p1, v1, a1), Finding(p2, v2, None)],
        candidates=12,
        source_counts={"europepmc": 7, "crossref": 5},
        in_scope_total=1,
    )


class LivePanelTests(unittest.TestCase):
    def test_live_panel_is_silent_off_terminal(self):
        stream = io.StringIO()
        with ui.live_panel("live pipeline", '"hypertrophy"', stream=stream) as panel:
            self.assertFalse(panel.enabled)
            panel.event("parse", {"facets": ["hypertrophy"], "year_from": 2015, "year_to": None})
            panel.event("search_start", {"sources": ["europepmc", "crossref"]})
            panel.event("source_done", {"source": "europepmc", "count": 12, "ok": True})
            panel.event("search_done", {"candidates": 12})
            panel.event("read", {"done": 2, "total": 2})
            panel.event("screen_done", {"in_scope": 3, "adjacent": 2, "out_of_scope": 7})
            panel.event("access", {"done": 5, "total": 5})
            panel.event("done", {})
        self.assertEqual(stream.getvalue(), "")

    def test_live_panel_renders_bounded_box_with_stage_updates(self):
        stream = io.StringIO()
        panel = ui.LivePanel("live pipeline", '"hypertrophy"', stream=stream)
        panel.theme = ui.Theme(enabled=False, unicode=True)
        panel.event("parse", {"facets": ["young adults", "hypertrophy"], "year_from": 2015})
        panel.event("search_start", {"sources": ["europepmc", "crossref"]})
        panel.event("source_done", {"source": "europepmc", "count": 14, "ok": True})
        lines = panel.render_lines(width=84)
        self.assertTrue(lines[0].startswith("╭"))
        self.assertTrue(lines[-1].startswith("╰"))
        joined = "\n".join(lines)
        self.assertIn("young adults + hypertrophy (2015-any)", joined)
        self.assertIn("epmc ✓14", joined)
        for line in lines:
            self.assertLessEqual(ui.visible_len(line), 84)


class TuiAppTests(unittest.TestCase):
    def test_welcome_screen_fits_exact_dimensions(self):
        app = TuiApp(stream=io.StringIO())
        lines = app.render_lines(100, 28)
        self.assertEqual(len(lines), 28)
        for line in lines:
            self.assertLessEqual(ui.visible_len(line), 100)
        text = "\n".join(lines)
        self.assertIn("PapersPlease", text)
        self.assertIn("optimal training for hypertrophy", text)

    def test_report_split_and_narrow_layouts(self):
        app = TuiApp(report=_sample_report(), stream=io.StringIO())
        wide = app.render_lines(112, 30)
        self.assertEqual(len(wide), 30)
        self.assertIn("In Scope (1)", "\n".join(wide))
        self.assertIn("Resistance training volume", "\n".join(wide))

        narrow = app.render_lines(76, 24)
        self.assertEqual(len(narrow), 24)
        for line in narrow:
            self.assertLessEqual(ui.visible_len(line), 76)

    def test_navigation_buckets_and_overlays(self):
        app = TuiApp(report=_sample_report(), stream=io.StringIO())
        self.assertEqual(app.current_bucket, "in_scope")

        # Switch to adjacent via TAB
        self.assertTrue(app.handle_key("TAB"))
        self.assertEqual(app.current_bucket, "adjacent")
        self.assertEqual(app.current_finding().paper.title, "Training in older adults")

        # Open explain overlay and close with ESC
        self.assertTrue(app.handle_key("e"))
        self.assertEqual(app.overlay, "explain")
        overlay_text = "\n".join(app.render_lines(96, 26))
        self.assertIn("EXPLAIN", overlay_text)
        self.assertTrue(app.handle_key("ESC"))
        self.assertIsNone(app.overlay)

        # Open help overlay and close
        self.assertTrue(app.handle_key("?"))
        self.assertEqual(app.overlay, "help")
        self.assertTrue(app.handle_key("ESC"))
        self.assertIsNone(app.overlay)

        # Prompt typing + backspace + cancel
        self.assertTrue(app.handle_key("f"))
        self.assertEqual(app.prompt_mode, "refine")
        for ch in "min=0.6":
            app.handle_key(ch)
        self.assertEqual(app.prompt_buffer, "min=0.6")
        app.handle_key("BACKSPACE")
        self.assertEqual(app.prompt_buffer, "min=0.")
        app.handle_key("ESC")
        self.assertIsNone(app.prompt_mode)

        # Quit
        self.assertFalse(app.handle_key("q"))


class CliFindArgsTests(unittest.TestCase):
    def test_find_parser_has_require_in_and_interactive(self):
        parser = cli.build_parser()
        args = parser.parse_args(
            ["find", "optimal training for hypertrophy in younger adults since 2015"]
        )
        self.assertIsNone(args.require_in)
        self.assertFalse(args.interactive)

        args2 = parser.parse_args(
            ["find", "hypertrophy", "--require", "volume", "--require-in", "title", "-i"]
        )
        self.assertEqual(args2.require_in, "title")
        self.assertTrue(args2.interactive)

    def test_cmd_find_runs_without_attribute_error(self):
        fake_report = _sample_report()
        with patch("papersplease.cli.find", return_value=fake_report) as mock_find:
            buf = io.StringIO()
            with patch("sys.stdout", buf):
                rc = cli.main(["find", "optimal training for hypertrophy in younger adults since 2015"])
            self.assertEqual(rc, 0)
            mock_find.assert_called_once()
            self.assertIn("Resistance training volume", buf.getvalue())


class AdvancedTuiTests(unittest.TestCase):
    def test_detail_scrolling_pane_focus_and_narrow_bounds(self):
        rep = _sample_report()
        rep.findings[0].paper.abstract = " ".join(["Word"] * 220)
        app = TuiApp(report=rep, stream=io.StringIO())

        self.assertEqual(app.focus_pane, "list")
        self.assertTrue(app.handle_key("RIGHT"))
        self.assertEqual(app.focus_pane, "detail")
        self.assertTrue(app.handle_key("LEFT"))
        self.assertEqual(app.focus_pane, "list")

        self.assertEqual(app.detail_scroll, 0)
        app.handle_key("J")
        self.assertEqual(app.detail_scroll, 1)
        app.handle_key("PGDN")
        self.assertEqual(app.detail_scroll, 7)
        app.handle_key("K")
        self.assertEqual(app.detail_scroll, 6)
        app.handle_key("PGUP")
        self.assertEqual(app.detail_scroll, 0)

        narrow = app.render_lines(56, 18)
        self.assertEqual(len(narrow), 18)
        for line in narrow:
            self.assertLessEqual(ui.visible_len(line), 56)

    def test_starring_shortlist_sort_and_live_filter(self):
        app = TuiApp(report=_sample_report(), stream=io.StringIO())
        self.assertTrue(app.handle_key("s"))
        self.assertEqual(len(app.starred), 1)

        self.assertTrue(app.handle_key("4"))
        self.assertEqual(app.current_bucket, "shortlist")
        rendered = "\n".join(app.render_lines(112, 30))
        self.assertIn("Shortlist (1)", rendered)

        self.assertTrue(app.handle_key("S"))
        self.assertEqual(app.sort_mode, "year")
        self.assertTrue(app.handle_key("S"))
        self.assertEqual(app.sort_mode, "citations")

        self.assertTrue(app.handle_key("\\"))
        for ch in "older":
            app.handle_key(ch)
        app.handle_key("ENTER")
        self.assertEqual(app.filter_text, "older")
        self.assertTrue(app.handle_key("TAB"))
        self.assertEqual(app.current_bucket, "adjacent")
        self.assertEqual(app.current_finding().paper.title, "Training in older adults")

        app.handle_key("ESC")
        self.assertEqual(app.filter_text, "")

    def test_truncate_ansi_preserves_ansi_and_visible_width(self):
        styled = "\x1b[36mHello \x1b[1mWorld from PapersPlease\x1b[0m"
        cut = ui.truncate_ansi(styled, 16, unicode=False)
        self.assertLessEqual(ui.visible_len(cut), 16)
        self.assertIn("\x1b[36m", cut)
        self.assertIn("\x1b[0m", cut)
        self.assertTrue(ui._ANSI.sub("", cut).endswith("..."))


if __name__ == "__main__":
    unittest.main()
