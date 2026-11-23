"""The terminal renderer: hierarchy, colour gating and table states."""

import io
import os
import unittest
from unittest.mock import patch

from papersplease import ui
from papersplease.intent import parse_intent
from papersplease.models import Paper, Verdict
from papersplease.pipeline import Finding, Report, explain_intent
from papersplease.render import (
    render_downloads,
    render_explain,
    render_mirrors,
    render_providers,
    render_read,
    render_text,
)
from papersplease.resolver import Access, MirrorStatus


def finding(title, bucket, *, score=0.9, coverage=0.9, reason="matched volume, frequency"):
    paper = Paper(
        title=title,
        doi="10.1000/example",
        year=2024,
        authors=["Doe Jane", "Roe Richard"],
        journal="Journal of Testing",
        cited_by=12,
    )
    verdict = Verdict(
        doi=paper.doi,
        title=paper.title,
        score=score,
        coverage=coverage,
        title_coverage=coverage,
        matched=["volume", "frequency"],
        bucket=bucket,
        in_scope=bucket == "in_scope",
        reason=reason,
    )
    return Finding(paper=paper, verdict=verdict)


def report() -> Report:
    spec = parse_intent("sets and frequency for hypertrophy")
    return Report(
        spec=spec,
        findings=[
            finding("A trial of equal-volume training", "in_scope"),
            finding("A frequency trial", "in_scope"),
            finding("Creatine in older men", "adjacent", score=0.61, reason="title names no anchor"),
            finding("Vegan protein in older adults", "out_of_scope", score=0.2, reason="anchors missing"),
        ],
        candidates=71,
        source_counts={"crossref": 30, "europepmc": 24},
        in_scope_total=4,
    )


class TextReportTests(unittest.TestCase):
    def test_plain_output_has_hierarchy_but_no_ansi(self):
        text = render_text(report(), width=96, color=False)
        self.assertNotIn("\x1b[", text)
        self.assertIn("PapersPlease  ·  literature scope report", text)
        self.assertIn("● IN SCOPE", text)
        self.assertIn("◐ ADJACENT", text)
        self.assertIn("REJECTED", text)
        self.assertIn("A trial of equal-volume training", text)
        self.assertIn("4 in scope (showing 2)", text)
        self.assertIn("why", text)

    def test_forced_colour_emits_ansi(self):
        text = render_text(report(), width=96, color=True)
        self.assertIn("\x1b[", text)
        self.assertIn("IN SCOPE", text)

    def test_adjacent_bucket_is_capped_and_says_where_the_rest_is(self):
        base = report()
        base.findings = [item for item in base.findings if item.verdict.bucket != "adjacent"]
        for index in range(9):
            base.findings.append(finding(f"Adjacent paper {index}", "adjacent"))
        text = render_text(base, width=96, color=False, max_adjacent=3)
        self.assertIn("showing 3; use --format json for all", text)
        self.assertIn("Adjacent paper 0", text)
        self.assertIn("Adjacent paper 2", text)
        self.assertNotIn("Adjacent paper 3", text)

    def test_empty_report_has_an_actionable_message(self):
        spec = parse_intent("sets and frequency for hypertrophy")
        text = render_text(Report(spec=spec), width=96, color=False)
        self.assertIn("No papers matched.", text)
        self.assertIn("--min-scope", text)


class ExplainRenderTests(unittest.TestCase):
    def test_contract_shows_facets_guards_and_query(self):
        payload = explain_intent("sets and frequency for hypertrophy")
        text = render_explain(payload, width=96, color=False)
        self.assertIn("PapersPlease  ·  scope contract (offline)", text)
        self.assertIn("FACETS", text)
        self.assertIn("volume", text)
        self.assertIn("anchors volume, frequency", text)
        self.assertIn("Query", text)


class TableRenderTests(unittest.TestCase):
    def test_providers_marks_a_missing_key(self):
        rows = [
            {
                "source": "europepmc",
                "default": True,
                "key_env": None,
                "configured": True,
                "note": "abstracts and full text",
            },
            {
                "source": "openalex",
                "default": True,
                "key_env": "PAPERSPLEASE_OPENALEX_KEY",
                "configured": False,
                "note": "broad coverage",
            },
        ]
        text = render_providers(rows, width=96, color=False)
        self.assertIn("SOURCE", text)
        self.assertIn("PAPERSPLEASE_OPENALEX_KEY !", text)
        self.assertNotIn("\x1b[", text)

    def test_mirror_states_are_named(self):
        statuses = [
            MirrorStatus("https://sci-hub.ru", "ok", 200, 120, "reachable"),
            MirrorStatus("https://sci-hub.st", "blocked", 403, None, "refused"),
            MirrorStatus("https://sci-hub.se", "unreachable", None, None, "no DNS"),
        ]
        text = render_mirrors(statuses, width=96, color=False)
        self.assertIn("ok", text)
        self.assertIn("blocked", text)
        self.assertIn("unreachable", text)


class PaperRenderTests(unittest.TestCase):
    def test_read_shows_abstract_and_access(self):
        paper = Paper(
            title="A trial",
            doi="10.1000/x",
            year=2024,
            authors=["Doe Jane"],
            journal="Journal",
            abstract="An abstract about training volume.",
        )
        access = Access(
            doi=paper.doi,
            doi_url="https://doi.org/10.1000/x",
            open_access=[{"url": "https://example.org/x.pdf", "kind": "oa", "license": None}],
            scihub=[],
        )
        text = render_read(paper, access, width=96, color=False)
        self.assertIn("ABSTRACT", text)
        self.assertIn("ACCESS", text)
        self.assertIn("https://example.org/x.pdf", text)

    def test_downloads_report_both_outcomes(self):
        text = render_downloads(
            [
                {"saved": True, "title": "One", "path": "/tmp/one.pdf", "source": "arxiv"},
                {"saved": False, "title": "Two", "note": "not a PDF"},
            ],
            width=96,
            color=False,
        )
        self.assertIn("saved", text)
        self.assertIn("failed", text)
        self.assertIn("1 saved · 1 failed", text)


class ColourModeTests(unittest.TestCase):
    def test_no_color_and_forced_color(self):
        stream = io.StringIO()
        with patch.dict(os.environ, {"NO_COLOR": "1"}, clear=False):
            self.assertFalse(ui.color_enabled(stream))
        with patch.dict(os.environ, {"PAPERSPLEASE_COLOR": "always"}, clear=False):
            self.assertTrue(ui.color_enabled(stream))
        with patch.dict(os.environ, {"PAPERSPLEASE_COLOR": "never"}, clear=False):
            self.assertFalse(ui.color_enabled(stream))

    def test_ansi_is_not_counted_as_visible_width(self):
        self.assertEqual(ui.visible_len("\x1b[31mred\x1b[0m"), 3)

    def test_wrap_measures_coloured_indents_without_the_escapes(self):
        lines = ui.wrap("one two three four five six", 24, initial="\x1b[2mwhy\x1b[0m ")
        self.assertTrue(all(ui.visible_len(line) <= 24 for line in lines))
        self.assertGreater(len(lines), 1)


class ProgressTests(unittest.TestCase):
    def test_progress_is_silent_off_terminal(self):
        stream = io.StringIO()
        with ui.progress("working", stream=stream) as bar:
            self.assertFalse(bar.enabled)
        self.assertEqual(stream.getvalue(), "")


if __name__ == "__main__":
    unittest.main()
